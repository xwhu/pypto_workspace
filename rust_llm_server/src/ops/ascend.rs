//! Ascend NPU backend — implements `ComputeOps`, `CommOps`, `QuantOps`
//! using the `ascend` crate (CANN `aclnn*` operators).
//!
//! This module is only compiled when the `ascend` feature is enabled:
//! ```bash
//! cargo build --features ascend
//! ```

use crate::model::tensor::{DType, Tensor};
use super::stubs::{ComputeOps, CommOps, QuantOps};

use ascend::{Device, Stream, DeviceBuffer, AclTensor};
use aclnn_sys::common::AclDataType;

/// Convert our `DType` to CANN's `AclDataType`.
fn to_acl_dtype(dtype: DType) -> AclDataType {
    match dtype {
        DType::Float16 => AclDataType::Float16,
        DType::BFloat16 => AclDataType::BFloat16,
        DType::Float32 => AclDataType::Float,
        DType::Int32 => AclDataType::Int32,
        DType::Uint32 => AclDataType::Uint32,
        DType::Int8 => AclDataType::Int8,
        DType::Int4 => AclDataType::Int4,
    }
}

/// Convert our `Tensor` shape (Vec<usize>) to i64 for aclnn.
fn shape_i64(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|&s| s as i64).collect()
}

// ─── AscendComputeOps ──────────────────────────────────────────────────

/// Real NPU compute backend using CANN `aclnn*` operators.
///
/// Holds the device guard and a compute stream. All operator calls
/// are enqueued on this stream.
pub struct AscendComputeOps {
    #[allow(dead_code)]
    device: Device,
    stream: Stream,
    device_id: i32,
}

impl AscendComputeOps {
    /// Initialize the Ascend backend on the given device.
    ///
    /// Reads `ASCEND_DEVICE_ID` env var if `device_id` is `None`.
    pub fn new(device_id: Option<i32>) -> Result<Self, ascend::AscendError> {
        let id = device_id.unwrap_or_else(|| {
            std::env::var("ASCEND_DEVICE_ID")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        });
        let device = Device::init(id)?;
        let stream = Stream::new()?;

        let (free, total) = device.memory_info().unwrap_or((0, 0));
        tracing::info!(
            "Ascend NPU device {} initialized: {:.2} GB free / {:.2} GB total",
            device.id(),
            free as f64 / 1e9,
            total as f64 / 1e9,
        );

        Ok(Self { device, stream, device_id: id })
    }

    /// Synchronize the compute stream (wait for all enqueued ops to finish).
    pub fn synchronize(&self) -> Result<(), ascend::AscendError> {
        self.stream.synchronize()
    }

    /// Ensure the CANN device context is set for the current thread.
    ///
    /// CANN's device context is thread-local. When using tokio's multi-thread
    /// runtime, HTTP handlers may run on different worker threads that don't
    /// have the device context set.
    fn ensure_device_context(&self) {
        unsafe {
            ascendcl_sys::aclrtSetDevice(self.device_id);
        }
    }

    /// Helper: create an AclTensor from our Tensor.
    ///
    /// - If tensor has `data_ptr` (loaded weight on device) → wrap existing pointer, no allocation.
    /// - Otherwise → allocate a new DeviceBuffer (for intermediate/output tensors).
    ///
    /// Returns (Optional owned buffer, AclTensor descriptor).
    fn make_acl_tensor(
        &self,
        tensor: &Tensor,
    ) -> Result<(Option<DeviceBuffer>, AclTensor), ascend::AscendError> {
        // Ensure device context on this thread (CANN context is thread-local)
        self.ensure_device_context();

        let shape = shape_i64(&tensor.shape);
        let dtype = to_acl_dtype(tensor.dtype);
        let byte_size = tensor.size_bytes();

        tracing::debug!(
            "make_acl_tensor: name={}, shape={:?}, dtype={:?}, size={} bytes, has_data_ptr={}",
            tensor.name, tensor.shape, tensor.dtype, byte_size, tensor.data_ptr.is_some()
        );

        if let Some(ptr) = tensor.data_ptr {
            // Weight tensor: device memory already allocated and filled.
            let device_ptr = ptr as *mut std::os::raw::c_void;
            let acl_t = AclTensor::from_ptr(&shape, dtype, device_ptr)?;
            Ok((None, acl_t))
        } else {
            // Intermediate / output tensor: allocate fresh device memory.
            if byte_size == 0 {
                return Err(ascend::AscendError::InvalidArgument(format!(
                    "Cannot allocate 0-size tensor: name={}, shape={:?}",
                    tensor.name, tensor.shape
                )));
            }
            let buf = DeviceBuffer::alloc(byte_size)?;
            let acl_t = AclTensor::new(&shape, dtype, &buf)?;
            Ok((Some(buf), acl_t))
        }
    }
}

impl ComputeOps for AscendComputeOps {
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) {
        self.ensure_device_context();

        let (_buf_a, acl_a) = self.make_acl_tensor(a)
            .expect("AscendComputeOps::matmul: failed to create tensor A");

        // Weight B is stored as [out_features, in_features] (PyTorch convention).
        // aclnnMatmul expects B = [K, N] = [in_features, out_features].
        // Create a transposed view for 2D weights.
        let (_buf_b, acl_b) = if b.shape.len() == 2 && b.data_ptr.is_some() {
            let shape = shape_i64(&b.shape);
            let dtype = to_acl_dtype(b.dtype);
            let device_ptr = b.data_ptr.unwrap() as *mut std::os::raw::c_void;
            let acl_t = AclTensor::from_ptr_transposed_2d(&shape, dtype, device_ptr)
                .expect("AscendComputeOps::matmul: failed to create transposed tensor B");
            (None, acl_t)
        } else {
            self.make_acl_tensor(b)
                .expect("AscendComputeOps::matmul: failed to create tensor B")
        };

        let (_buf_out, mut acl_out) = self.make_acl_tensor(out)
            .expect("AscendComputeOps::matmul: failed to create tensor Out");

        ascend::ops::matmul::matmul(&self.stream, &acl_a, &acl_b, &mut acl_out)
            .expect("AscendComputeOps::matmul: aclnnMatmul failed");

        tracing::trace!("ascend::matmul({} @ {} -> {})", a, b, out);
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor) {
        tracing::debug!(
            "rms_norm: input={} shape={:?} dtype={:?} ptr={:?}, weight={} shape={:?} dtype={:?} ptr={:?}",
            input.name, input.shape, input.dtype, input.data_ptr,
            weight.name, weight.shape, weight.dtype, weight.data_ptr,
        );

        let (_buf_x, acl_x) = self.make_acl_tensor(input)
            .expect("AscendComputeOps::rms_norm: tensor x");
        let (_buf_w, acl_w) = self.make_acl_tensor(weight)
            .expect("AscendComputeOps::rms_norm: tensor w");
        let (_buf_y, mut acl_y) = self.make_acl_tensor(out)
            .expect("AscendComputeOps::rms_norm: tensor y");

        ascend::ops::rmsnorm::rmsnorm(&self.stream, &acl_x, &acl_w, eps as f64, &mut acl_y)
            .expect("AscendComputeOps::rms_norm: aclnnRmsNorm failed");

        tracing::trace!("ascend::rms_norm({}, eps={})", input, eps);
    }

    fn rotary_embedding(&self, q: &mut Tensor, k: &mut Tensor, positions: &[u32], rope_theta: f64) {
        // TODO: Implement via aclnnRotaryPosEmb
        // Requires building cos/sin tables from positions + rope_theta
        tracing::debug!(
            "ascend::rotary_embedding({}, {}, pos_len={}, theta={}) [NOT YET IMPLEMENTED]",
            q, k, positions.len(), rope_theta
        );
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        // TODO: Implement via aclnnFlashAttentionScore
        tracing::debug!(
            "ascend::attention(q={}, k={}, v={} -> {}, h={}, kv={}, d={}) [NOT YET IMPLEMENTED]",
            q, k, v, out, num_heads, num_kv_heads, head_dim
        );
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor, out: &mut Tensor) {
        self.ensure_device_context();

        // Step 1: silu_out = silu(gate)
        let (_buf_gate, acl_gate) = self.make_acl_tensor(gate)
            .expect("silu_mul: gate tensor");
        let silu_buf = DeviceBuffer::alloc(gate.size_bytes())
            .expect("silu_mul: alloc silu_out");
        let mut acl_silu = AclTensor::new(
            &shape_i64(&gate.shape), to_acl_dtype(gate.dtype), &silu_buf,
        ).expect("silu_mul: silu_out tensor");

        ascend::ops::activation::silu(&self.stream, &acl_gate, &mut acl_silu)
            .expect("silu_mul: aclnnSilu failed");

        // Step 2: out = silu_out * up
        let (_buf_up, acl_up) = self.make_acl_tensor(up)
            .expect("silu_mul: up tensor");
        let (_buf_out, mut acl_out) = self.make_acl_tensor(out)
            .expect("silu_mul: output tensor");

        ascend::ops::elementwise::mul(&self.stream, &acl_silu, &acl_up, &mut acl_out)
            .expect("silu_mul: aclnnMul failed");

        tracing::trace!("ascend::silu_mul({}, {} -> {})", gate, up, out);
    }

    fn embedding(&self, ids: &[u32], table: &Tensor, out: &mut Tensor) {
        self.ensure_device_context();

        // Convert u32 ids to i64 and upload to device
        let ids_i64: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let ids_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                ids_i64.as_ptr() as *const u8,
                ids_i64.len() * std::mem::size_of::<i64>(),
            )
        };
        let mut ids_buf = DeviceBuffer::alloc(ids_bytes.len())
            .expect("embedding: failed to alloc ids buffer");
        ids_buf.copy_from_host(ids_bytes)
            .expect("embedding: failed to copy ids to device");

        // Create AclTensor for ids — shape must match output prefix dims.
        // Output is [1, seq_len, embed_dim], so indices must be [1, seq_len].
        let ids_shape = [1i64, ids.len() as i64];
        let ids_acl = AclTensor::from_ptr(
            &ids_shape,
            aclnn_sys::common::AclDataType::Int64,
            ids_buf.ptr(),
        ).expect("embedding: failed to create ids tensor");

        // Table weight (already on device)
        let (_buf_table, acl_table) = self.make_acl_tensor(table)
            .expect("embedding: table tensor");

        // Output
        let (_buf_out, mut acl_out) = self.make_acl_tensor(out)
            .expect("embedding: output tensor");

        ascend::ops::embedding::embedding(&self.stream, &acl_table, &ids_acl, &mut acl_out)
            .expect("embedding: aclnnEmbedding failed");

        tracing::trace!("ascend::embedding(ids_len={}, {} -> {})", ids.len(), table, out);
    }

    fn softmax(&self, input: &Tensor, out: &mut Tensor) {
        self.ensure_device_context();

        let (_buf_in, acl_in) = self.make_acl_tensor(input)
            .expect("softmax: input tensor");
        let (_buf_out, mut acl_out) = self.make_acl_tensor(out)
            .expect("softmax: output tensor");

        // Softmax along last dimension
        let dim = (input.shape.len() as i64) - 1;
        ascend::ops::reduction::softmax(&self.stream, &acl_in, dim, &mut acl_out)
            .expect("softmax: aclnnSoftmax failed");

        tracing::trace!("ascend::softmax({} -> {})", input, out);
    }

    fn add(&self, a: &mut Tensor, b: &Tensor) {
        self.ensure_device_context();

        let (_buf_a, acl_a) = self.make_acl_tensor(a)
            .expect("add: tensor a");
        let (_buf_b, acl_b) = self.make_acl_tensor(b)
            .expect("add: tensor b");

        ascend::ops::elementwise::inplace_add(&self.stream, &acl_a, &acl_b, 1.0)
            .expect("add: aclnnInplaceAdd failed");

        tracing::trace!("ascend::add({} += {})", a, b);
    }

    fn sample_argmax(&self, logits: &Tensor) -> u32 {
        self.ensure_device_context();

        let (_buf_logits, acl_logits) = self.make_acl_tensor(logits)
            .expect("sample_argmax: logits tensor");

        // ArgMax output shape: input shape with last dim removed.
        // e.g., [1, 4, vocab_size] → [1, 4]
        let last_dim = logits.shape.len() - 1;
        let out_shape: Vec<i64> = logits.shape[..last_dim]
            .iter()
            .map(|&d| d as i64)
            .collect();
        let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
        let out_bytes = out_numel * std::mem::size_of::<i64>();

        let out_buf = DeviceBuffer::alloc(out_bytes)
            .expect("sample_argmax: alloc output");
        let mut acl_out = AclTensor::from_ptr(
            &out_shape,
            aclnn_sys::common::AclDataType::Int64,
            out_buf.ptr(),
        ).expect("sample_argmax: output tensor");

        // ArgMax along last dimension
        ascend::ops::reduction::argmax(
            &self.stream, &acl_logits, last_dim as i64, false, &mut acl_out,
        ).expect("sample_argmax: aclnnArgMax failed");

        // Synchronize and copy all results back to host
        self.stream.synchronize().expect("sample_argmax: sync failed");
        let mut results = vec![0i64; out_numel];
        out_buf.copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(
                results.as_mut_ptr() as *mut u8,
                out_bytes,
            )
        }).expect("sample_argmax: copy result to host failed");

        // Return the last position's argmax (for autoregressive generation)
        let token_id = *results.last().unwrap_or(&0);
        tracing::trace!("ascend::sample_argmax({}) = {}", logits, token_id);
        token_id as u32
    }
}

// ─── AscendCommOps ─────────────────────────────────────────────────────

/// Communication ops for Ascend multi-NPU (HCCL).
///
/// Placeholder — will be implemented using HCCL library.
pub struct AscendCommOps;

impl CommOps for AscendCommOps {
    fn all_reduce_sum(&self, tensor: &mut Tensor) {
        tracing::debug!("ascend::all_reduce_sum({}) [NOT YET IMPLEMENTED]", tensor);
    }
    fn all_gather(&self, input: &Tensor, out: &mut Tensor) {
        tracing::debug!("ascend::all_gather({} -> {}) [NOT YET IMPLEMENTED]", input, out);
    }
    fn send(&self, tensor: &Tensor, dst_rank: usize) {
        tracing::debug!("ascend::send({} -> rank {}) [NOT YET IMPLEMENTED]", tensor, dst_rank);
    }
    fn recv(&self, out: &mut Tensor, src_rank: usize) {
        tracing::debug!("ascend::recv({} <- rank {}) [NOT YET IMPLEMENTED]", out, src_rank);
    }
}

// ─── AscendQuantOps ────────────────────────────────────────────────────

/// Quantization ops for Ascend NPU.
///
/// Placeholder — will be implemented using CANN quant operators.
pub struct AscendQuantOps;

impl QuantOps for AscendQuantOps {
    fn matmul_quantized(
        &self,
        input: &Tensor,
        weight: &Tensor,
        scales: &Tensor,
        _zeros: Option<&Tensor>,
        out: &mut Tensor,
    ) {
        tracing::debug!(
            "ascend::matmul_quantized({} @ {} [s={}] -> {}) [NOT YET IMPLEMENTED]",
            input, weight, scales, out
        );
    }

    fn dequantize(
        &self,
        quantized: &Tensor,
        scales: &Tensor,
        _zeros: Option<&Tensor>,
        out: &mut Tensor,
    ) {
        tracing::debug!(
            "ascend::dequantize({} [s={}] -> {}) [NOT YET IMPLEMENTED]",
            quantized, scales, out
        );
    }
}

// ─── OpsBundle Constructor ─────────────────────────────────────────────

use super::stubs::OpsBundle;

impl OpsBundle {
    /// Create an OpsBundle using the Ascend NPU backend.
    ///
    /// `device_id`: NPU device to use. `None` reads `ASCEND_DEVICE_ID` env var.
    pub fn ascend(device_id: Option<i32>) -> Result<Self, ascend::AscendError> {
        let compute = AscendComputeOps::new(device_id)?;
        Ok(Self {
            compute: Box::new(compute),
            comm: Box::new(AscendCommOps),
            quant: Box::new(AscendQuantOps),
        })
    }
}
