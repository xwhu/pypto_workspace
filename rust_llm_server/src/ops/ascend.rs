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

/// Greatest common divisor (used to infer head_dim from Q/K hidden sizes).
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Persist an output buffer so subsequent operations can access the data.
///
/// Frees the old device memory (if any) to prevent device memory exhaustion.
/// Weight tensors (which have device_buf set) are not freed.
fn persist_output(tensor: &mut Tensor, buf: Option<DeviceBuffer>) {
    if let Some(b) = buf {
        // Free old intermediate device memory before replacing
        if let Some(old_ptr) = tensor.data_ptr {
            if tensor.device_buf.is_none() {
                unsafe {
                    ascendcl_sys::aclrtFree(old_ptr as *mut std::os::raw::c_void);
                }
            }
        }
        tensor.data_ptr = Some(b.ptr() as usize);
        std::mem::forget(b);
    }
}

/// Debug helper: synchronize stream, dump first N FP16 values from a device tensor.
/// Only active when RUST_LOG=debug or trace.
fn debug_dump_fp16(stream: &ascend::Stream, tensor: &Tensor, label: &str, n: usize) {
    if !tracing::enabled!(tracing::Level::DEBUG) { return; }
    if let Some(ptr) = tensor.data_ptr {
        stream.synchronize().ok();
        let total_elems: usize = tensor.shape.iter().product();
        let n_vals = n.min(total_elems);
        let byte_len = n_vals * 2; // FP16 = 2 bytes
        let mut host_buf = vec![0u8; byte_len];
        let device_ptr = ptr as *mut std::os::raw::c_void;
        // Use ascend's DeviceBuffer wrapper for safe copy
        unsafe {
            ascendcl_sys::aclrtMemcpy(
                host_buf.as_mut_ptr() as *mut std::os::raw::c_void,
                byte_len,
                device_ptr,
                byte_len,
                ascendcl_sys::AclrtMemcpyKind::DeviceToHost,
            );
        }
        let vals: Vec<f32> = host_buf.chunks(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect();
        tracing::debug!("{}: shape={:?} first_{}={:?}", label, tensor.shape, n, vals);
    } else {
        tracing::debug!("{}: shape={:?} NO DATA_PTR", label, tensor.shape);
    }
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

        persist_output(out, _buf_out);
        debug_dump_fp16(&self.stream, out, "matmul_out", 8);
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

        persist_output(out, _buf_y);
        debug_dump_fp16(&self.stream, out, "rms_norm_out", 8);
        tracing::trace!("ascend::rms_norm({}, eps={})", input, eps);
    }

    fn qk_norm(&self, qk: &mut Tensor, weight: &Tensor, num_heads: usize, head_dim: usize, eps: f32) {
        self.ensure_device_context();

        // qk shape is [B, S, num_heads * head_dim]
        // We need to reshape to [1, B*S*num_heads, head_dim] for per-head RMS norm
        // (CANN aclnnRmsNorm requires at least 3D input)
        let total_hidden: usize = qk.shape.iter().product();
        let n_groups = total_hidden / head_dim; // = B * S * num_heads

        tracing::debug!(
            "qk_norm: qk_shape={:?}, weight_shape={:?}, total_hidden={}, n_groups={}, head_dim={}, num_heads={}",
            qk.shape, weight.shape, total_hidden, n_groups, head_dim, num_heads
        );

        // Use 3D shape: [1, n_groups, head_dim]
        let reshaped_shape = [1i64, n_groups as i64, head_dim as i64];
        let ptr = qk.data_ptr.expect("qk_norm: qk has no data_ptr");
        let device_ptr = ptr as *mut std::os::raw::c_void;

        let acl_x = AclTensor::from_ptr(
            &reshaped_shape, to_acl_dtype(qk.dtype), device_ptr,
        ).expect("qk_norm: input tensor");

        // Weight is [head_dim]
        let (_buf_w, acl_w) = self.make_acl_tensor(weight)
            .expect("qk_norm: weight tensor");

        // Output buffer (same size as input)
        let out_bytes = total_hidden * 2; // FP16
        let out_buf = DeviceBuffer::alloc(out_bytes)
            .expect("qk_norm: alloc output");
        let mut acl_y = AclTensor::new(
            &reshaped_shape, to_acl_dtype(qk.dtype), &out_buf,
        ).expect("qk_norm: output tensor");

        ascend::ops::rmsnorm::rmsnorm(&self.stream, &acl_x, &acl_w, eps as f64, &mut acl_y)
            .unwrap_or_else(|e| {
                panic!(
                    "qk_norm: aclnnRmsNorm failed: {:?}\n  qk_shape={:?}, weight_shape={:?}, reshaped={:?}, n_groups={}, head_dim={}",
                    e, qk.shape, weight.shape, reshaped_shape, n_groups, head_dim
                );
            });

        // Persist output (free old device memory first)
        persist_output(qk, Some(out_buf));

        tracing::trace!("ascend::qk_norm(heads={}, d={}, eps={})", num_heads, head_dim, eps);
    }

    fn rotary_embedding(&self, q: &mut Tensor, k: &mut Tensor, positions: &[u32], rope_theta: f64, head_dim: usize) {
        self.ensure_device_context();

        let seq_len = positions.len();

        // Use explicit head_dim to compute num_heads from Q/K hidden sizes
        let q_hidden = *q.shape.last().unwrap();
        let k_hidden = *k.shape.last().unwrap();
        let num_q_heads = q_hidden / head_dim;
        let num_kv_heads = k_hidden / head_dim;
        let batch = q.shape[0];

        // ── 1. Precompute cos/sin tables on host ──
        // Shape: [1, seq, 1, head_dim] for broadcasting with [batch, seq, heads, dim]
        // mode=0 (half): first half_dim values are cos/sin of the frequencies
        let half_dim = head_dim / 2;
        let table_size = seq_len * head_dim;
        let mut cos_table = vec![0.0f32; table_size];
        let mut sin_table = vec![0.0f32; table_size];

        for (s, &pos) in positions.iter().enumerate() {
            for i in 0..half_dim {
                let freq = (pos as f64) / rope_theta.powf(2.0 * i as f64 / head_dim as f64);
                let cos_val = freq.cos() as f32;
                let sin_val = freq.sin() as f32;
                // mode=0 (half): [cos_0..cos_{d/2-1}, cos_0..cos_{d/2-1}]
                cos_table[s * head_dim + i] = cos_val;
                cos_table[s * head_dim + half_dim + i] = cos_val;
                sin_table[s * head_dim + i] = sin_val;
                sin_table[s * head_dim + half_dim + i] = sin_val;
            }
        }

        // Convert to FP16 and upload
        let cos_f16: Vec<u16> = cos_table.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let sin_f16: Vec<u16> = sin_table.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();

        let cos_bytes = unsafe {
            std::slice::from_raw_parts(cos_f16.as_ptr() as *const u8, cos_f16.len() * 2)
        };
        let sin_bytes = unsafe {
            std::slice::from_raw_parts(sin_f16.as_ptr() as *const u8, sin_f16.len() * 2)
        };

        let mut cos_buf = DeviceBuffer::alloc(cos_bytes.len()).expect("rope: alloc cos");
        cos_buf.copy_from_host(cos_bytes).expect("rope: upload cos");
        let mut sin_buf = DeviceBuffer::alloc(sin_bytes.len()).expect("rope: alloc sin");
        sin_buf.copy_from_host(sin_bytes).expect("rope: upload sin");

        // cos/sin shape: [1, seq, 1, head_dim] — broadcasts to [batch, seq, heads, dim]
        let cs_shape = [1i64, seq_len as i64, 1i64, head_dim as i64];
        let acl_cos = AclTensor::from_ptr(&cs_shape, to_acl_dtype(DType::Float16), cos_buf.ptr())
            .expect("rope: cos tensor");
        let acl_sin = AclTensor::from_ptr(&cs_shape, to_acl_dtype(DType::Float16), sin_buf.ptr())
            .expect("rope: sin tensor");

        // ── 2. Create 4D views of Q/K ──
        let make_4d_view = |t: &Tensor, n_heads: usize| -> (Option<DeviceBuffer>, AclTensor) {
            let shape_4d = [batch as i64, seq_len as i64, n_heads as i64, head_dim as i64];
            let strides = [
                (seq_len * n_heads * head_dim) as i64,
                (n_heads * head_dim) as i64,
                head_dim as i64,
                1i64,
            ];
            if let Some(ptr) = t.data_ptr {
                let device_ptr = ptr as *mut std::os::raw::c_void;
                let acl_t = AclTensor::from_ptr_with_strides(
                    &shape_4d, &strides, to_acl_dtype(t.dtype), device_ptr,
                ).expect("rope: 4D view");
                (None, acl_t)
            } else {
                let buf = DeviceBuffer::alloc(t.size_bytes()).expect("rope: alloc");
                let acl_t = AclTensor::new(&shape_4d, to_acl_dtype(t.dtype), &buf)
                    .expect("rope: tensor");
                (Some(buf), acl_t)
            }
        };

        let (_buf_q, acl_q) = make_4d_view(q, num_q_heads);
        let (_buf_k, acl_k) = make_4d_view(k, num_kv_heads);

        // ── 3. Allocate output tensors ──
        let q_out_buf = DeviceBuffer::alloc(q.size_bytes()).expect("rope: alloc q_out");
        let q_out_shape = [batch as i64, seq_len as i64, num_q_heads as i64, head_dim as i64];
        let mut acl_q_out = AclTensor::new(&q_out_shape, to_acl_dtype(q.dtype), &q_out_buf)
            .expect("rope: q_out");
        let k_out_buf = DeviceBuffer::alloc(k.size_bytes()).expect("rope: alloc k_out");
        let k_out_shape = [batch as i64, seq_len as i64, num_kv_heads as i64, head_dim as i64];
        let mut acl_k_out = AclTensor::new(&k_out_shape, to_acl_dtype(k.dtype), &k_out_buf)
            .expect("rope: k_out");

        // ── 4. Apply RoPE to Q and K separately ──
        // mode=0: half (standard HuggingFace RoPE)
        ascend::ops::rope::rotary_position_embedding(
            &self.stream, &acl_q, &acl_cos, &acl_sin, 0, &mut acl_q_out,
        ).expect("rope: Q aclnnRotaryPositionEmbedding failed");

        ascend::ops::rope::rotary_position_embedding(
            &self.stream, &acl_k, &acl_cos, &acl_sin, 0, &mut acl_k_out,
        ).expect("rope: K aclnnRotaryPositionEmbedding failed");

        // Update Q/K data_ptr (leak buffers; TODO: TensorPool)
        persist_output(q, Some(q_out_buf));
        persist_output(k, Some(k_out_buf));

        tracing::trace!(
            "ascend::rotary_embedding(q={}, k={}, seq={}, theta={})",
            q, k, seq_len, rope_theta
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
        self.ensure_device_context();

        let batch = q.shape[0];
        let seq_len = q.shape[1];

        tracing::debug!(
            "attention: batch={}, seq_len={}, q_shape={:?}, k_shape={:?}, v_shape={:?}, heads={}, kv_heads={}, head_dim={}",
            batch, seq_len, q.shape, k.shape, v.shape, num_heads, num_kv_heads, head_dim
        );

        // ── 1. Create ACL tensors for Q/K/V in BSH layout ──
        let (_buf_q, acl_q) = self.make_acl_tensor(q).expect("attention: Q tensor");
        let (_buf_k, acl_k) = self.make_acl_tensor(k).expect("attention: K tensor");
        let (_buf_v, acl_v) = self.make_acl_tensor(v).expect("attention: V tensor");

        // ── 2. Create causal attention mask ──
        // Synchronize stream first to ensure all prior ops finished and released temps
        self.stream.synchronize().ok();

        let mask_shape = [1i64, 1, seq_len as i64, seq_len as i64];
        let mask_numel = seq_len * seq_len;
        let mut host_mask = vec![0u8; mask_numel];
        for row in 0..seq_len {
            for col in (row + 1)..seq_len {
                host_mask[row * seq_len + col] = 1;
            }
        }
        let mut mask_buf = DeviceBuffer::alloc(mask_numel).expect("attention: alloc mask");
        mask_buf.copy_from_host(&host_mask).expect("attention: upload mask");
        let acl_mask = AclTensor::from_ptr(
            &mask_shape,
            aclnn_sys::common::AclDataType::Bool,
            mask_buf.ptr(),
        ).expect("attention: mask tensor");

        // ── 3. Allocate auxiliary softmax tensors (Float32) ──
        let aux_shape = [batch as i64, num_heads as i64, seq_len as i64, 8i64];
        let aux_bytes = batch * num_heads * seq_len * 8 * std::mem::size_of::<f32>();

        let sm_max_buf = DeviceBuffer::alloc(aux_bytes).expect("attention: alloc softmax_max");
        let acl_sm_max = AclTensor::new(
            &aux_shape, aclnn_sys::common::AclDataType::Float, &sm_max_buf,
        ).expect("attention: softmax_max tensor");

        let sm_sum_buf = DeviceBuffer::alloc(aux_bytes).expect("attention: alloc softmax_sum");
        let acl_sm_sum = AclTensor::new(
            &aux_shape, aclnn_sys::common::AclDataType::Float, &sm_sum_buf,
        ).expect("attention: softmax_sum tensor");

        // ── 4. Allocate output tensor ──
        let (_buf_out, mut acl_out) = self.make_acl_tensor(out).expect("attention: output tensor");

        // ── 5. Call FlashAttentionScore with explicit causal mask ──
        let scale = 1.0 / (head_dim as f64).sqrt();

        ascend::ops::attention::flash_attention_score_with_mask(
            &self.stream,
            &acl_q, &acl_k, &acl_v,
            &acl_mask,
            scale,
            num_heads as i64,
            "BSH",
            65536,
            &acl_sm_max, &acl_sm_sum,
            &mut acl_out,
        ).unwrap_or_else(|e| {
            // Log device memory before panicking
            let (free, total) = self.device.memory_info().unwrap_or((0, 0));
            panic!(
                "attention: aclnnFlashAttentionScore failed: {:?}\n  batch={}, seq_len={}, heads={}, kv_heads={}, head_dim={}, scale={:.6}\n  device memory: {:.2} MB free / {:.2} MB total",
                e, batch, seq_len, num_heads, num_kv_heads, head_dim, scale,
                free as f64 / 1e6, total as f64 / 1e6,
            );
        });

        if let Some(buf) = _buf_out {
            persist_output(out, Some(buf));
        }

        tracing::trace!(
            "ascend::attention(q={}, k={}, v={} -> {}, h={}, kv={}, d={})",
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

        persist_output(out, _buf_out);
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

        persist_output(out, _buf_out);
        debug_dump_fp16(&self.stream, out, "embedding_out", 8);
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

        persist_output(out, _buf_out);
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

        // In-place add: result is in a's buffer
        persist_output(a, _buf_a);
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
