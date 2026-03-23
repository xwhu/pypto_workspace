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
}

impl AscendComputeOps {
    /// Initialize the Ascend backend on the given device.
    ///
    /// Reads `ASCEND_DEVICE_ID` env var if `device_id` is `None`.
    pub fn new(device_id: Option<i32>) -> Result<Self, ascend::AscendError> {
        let device = match device_id {
            Some(id) => Device::init(id)?,
            None => Device::from_env()?,
        };
        let stream = Stream::new()?;

        let (free, total) = device.memory_info().unwrap_or((0, 0));
        tracing::info!(
            "Ascend NPU device {} initialized: {:.2} GB free / {:.2} GB total",
            device.id(),
            free as f64 / 1e9,
            total as f64 / 1e9,
        );

        Ok(Self { device, stream })
    }

    /// Synchronize the compute stream (wait for all enqueued ops to finish).
    pub fn synchronize(&self) -> Result<(), ascend::AscendError> {
        self.stream.synchronize()
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
        let shape = shape_i64(&tensor.shape);
        let dtype = to_acl_dtype(tensor.dtype);

        if let Some(ptr) = tensor.data_ptr {
            // Weight tensor: device memory already allocated and filled.
            // Create AclTensor pointing to existing device memory.
            let device_ptr = ptr as *mut std::os::raw::c_void;
            let acl_t = AclTensor::from_ptr(&shape, dtype, device_ptr)?;
            Ok((None, acl_t)) // No owned buffer — memory is managed by model's device_buf
        } else {
            // Intermediate / output tensor: allocate fresh device memory.
            let byte_size = tensor.size_bytes();
            let buf = DeviceBuffer::alloc(byte_size)?;
            let acl_t = AclTensor::new(&shape, dtype, &buf)?;
            Ok((Some(buf), acl_t)) // We own this buffer
        }
    }
}

impl ComputeOps for AscendComputeOps {
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) {
        // Allocate device memory
        let (_buf_a, acl_a) = self.make_acl_tensor(a)
            .expect("AscendComputeOps::matmul: failed to create tensor A");
        let (_buf_b, acl_b) = self.make_acl_tensor(b)
            .expect("AscendComputeOps::matmul: failed to create tensor B");
        let (_buf_out, mut acl_out) = self.make_acl_tensor(out)
            .expect("AscendComputeOps::matmul: failed to create tensor Out");

        ascend::ops::matmul::matmul(&self.stream, &acl_a, &acl_b, &mut acl_out)
            .expect("AscendComputeOps::matmul: aclnnMatmul failed");

        tracing::trace!("ascend::matmul({} @ {} -> {})", a, b, out);
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor) {
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
        // TODO: Implement via aclnnSilu + aclnnMul or aclnnSwiGlu
        tracing::debug!(
            "ascend::silu_mul({}, {} -> {}) [NOT YET IMPLEMENTED]",
            gate, up, out
        );
    }

    fn embedding(&self, ids: &[u32], table: &Tensor, out: &mut Tensor) {
        // TODO: Implement via aclnnEmbedding
        tracing::debug!(
            "ascend::embedding(ids_len={}, table={} -> {}) [NOT YET IMPLEMENTED]",
            ids.len(), table, out
        );
    }

    fn softmax(&self, input: &Tensor, out: &mut Tensor) {
        // TODO: Implement via aclnnSoftmax
        tracing::debug!(
            "ascend::softmax({} -> {}) [NOT YET IMPLEMENTED]",
            input, out
        );
    }

    fn add(&self, a: &mut Tensor, b: &Tensor) {
        // TODO: Implement via aclnnInplaceAdd
        tracing::debug!(
            "ascend::add({} += {}) [NOT YET IMPLEMENTED]",
            a, b
        );
    }

    fn sample_argmax(&self, logits: &Tensor) -> u32 {
        // TODO: Implement via aclnnArgMax
        tracing::debug!(
            "ascend::sample_argmax({}) [NOT YET IMPLEMENTED, returning 0]",
            logits
        );
        0
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
