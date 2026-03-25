//! Ascend NPU backend — typed compute operators with RAII device memory.
//!
//! All device memory is managed via `DeviceTensor` (RAII wrapper over
//! `DeviceBuffer`). No `std::mem::forget` or manual `aclrtFree` calls.
//!
//! This module is only compiled when the `ascend` feature is enabled:
//! ```bash
//! cargo build --features ascend
//! ```

use crate::model::device_tensor::{DeviceTensor, WeightTensor};
use crate::model::tensor::DType;

use aclnn_sys::common::AclDataType;
use ascend::{AclTensor, Device, DeviceBuffer, Stream};

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

/// Convert shape `&[usize]` to `Vec<i64>` for aclnn.
fn shape_i64(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|&s| s as i64).collect()
}

// ─── Decode Pre-allocated Buffers ──────────────────────────────────────

/// Pre-allocated device buffers for the decode hot path.
///
/// Eliminates per-layer per-step `aclrtMalloc`/`aclrtFree` calls by
/// reusing these buffers across all layers and decode steps.
pub struct DecodeBuffers {
    /// Block table device buffer: [1, max_blocks_per_seq] INT32.
    /// Sized for the maximum sequence length; only `copy_from_host` per step.
    pub block_table_buf: DeviceBuffer,
    /// Maximum blocks per sequence this buffer was sized for.
    pub max_blocks_capacity: usize,
}

// ─── AscendComputeOps ──────────────────────────────────────────────────

/// NPU compute backend using CANN `aclnn*` operators.
///
/// Holds the device guard and a compute stream. All operator calls
/// are enqueued on this stream and use typed DeviceTensor/WeightTensor
/// for RAII-safe device memory management.
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

        Ok(Self {
            device,
            stream,
            device_id: id,
        })
    }

    /// Create pre-allocated decode buffers sized for this model.
    ///
    /// Call once at Engine init; reuse across all decode steps.
    pub fn init_decode_buffers(&self, max_blocks_per_seq: usize) -> DecodeBuffers {
        self.ensure_device_context();

        // Block table: max_blocks_per_seq × 4 bytes (INT32)
        let bt_bytes = max_blocks_per_seq * 4;
        let block_table_buf =
            DeviceBuffer::alloc(bt_bytes).expect("init_decode_buffers: alloc block_table");

        DecodeBuffers {
            block_table_buf,
            max_blocks_capacity: max_blocks_per_seq,
        }
    }

    /// Synchronize the compute stream (wait for all enqueued ops to finish).
    pub fn synchronize(&self) -> Result<(), ascend::AscendError> {
        self.stream.synchronize()
    }

    /// Ensure the CANN device context is set for the current thread.
    fn ensure_device_context(&self) {
        unsafe {
            ascendcl_sys::aclrtSetDevice(self.device_id);
        }
    }

    // ── Helpers ──

    /// Wrap a DeviceTensor as an AclTensor view (no allocation).
    fn wrap_device(dt: &DeviceTensor) -> AclTensor {
        let shape = shape_i64(dt.shape());
        let dtype = to_acl_dtype(dt.dtype());
        AclTensor::from_ptr(&shape, dtype, dt.ptr())
            .expect("wrap_device: failed to create AclTensor")
    }

    /// Wrap a WeightTensor as an AclTensor view (no allocation).
    fn wrap_weight(wt: &WeightTensor) -> AclTensor {
        let shape = shape_i64(wt.shape());
        let dtype = to_acl_dtype(wt.dtype());
        AclTensor::from_ptr(&shape, dtype, wt.ptr())
            .expect("wrap_weight: failed to create AclTensor")
    }

    /// Wrap a WeightTensor as a transposed 2D AclTensor view.
    fn wrap_weight_transposed(wt: &WeightTensor) -> AclTensor {
        let shape = shape_i64(wt.shape());
        let dtype = to_acl_dtype(wt.dtype());
        AclTensor::from_ptr_transposed_2d(&shape, dtype, wt.ptr())
            .expect("wrap_weight_transposed: failed to create AclTensor")
    }

    // ── Operators ──

    pub fn matmul(&self, a: &DeviceTensor, b: &WeightTensor) -> DeviceTensor {
        self.ensure_device_context();

        let acl_a = Self::wrap_device(a);
        let acl_b = if b.shape().len() == 2 {
            Self::wrap_weight_transposed(b)
        } else {
            Self::wrap_weight(b)
        };

        // Output shape: [batch, seq_len, out_features]
        let mut out_shape = a.shape().to_vec();
        if let Some(last) = out_shape.last_mut() {
            *last = b.shape()[0]; // [out_features, in_features] → out_features
        }
        let out =
            DeviceTensor::alloc(out_shape, a.dtype(), "matmul_out").expect("matmul: alloc output");
        let mut acl_out = Self::wrap_device(&out);

        ascend::ops::matmul::matmul(&self.stream, &acl_a, &acl_b, &mut acl_out).unwrap_or_else(
            |e| {
                panic!(
                    "matmul: aclnnMatmul failed: {:?}\n  a={:?}, b={:?}, out={:?}",
                    e,
                    a.shape(),
                    b.shape(),
                    out.shape(),
                );
            },
        );

        out
    }

    pub fn rms_norm(&self, input: &DeviceTensor, weight: &WeightTensor, eps: f32) -> DeviceTensor {
        self.ensure_device_context();

        let acl_x = Self::wrap_device(input);
        let acl_w = Self::wrap_weight(weight);

        let out = DeviceTensor::alloc(input.shape().to_vec(), input.dtype(), "norm_out")
            .expect("rms_norm: alloc output");
        let mut acl_y = Self::wrap_device(&out);

        ascend::ops::rmsnorm::rmsnorm(&self.stream, &acl_x, &acl_w, eps as f64, &mut acl_y)
            .unwrap_or_else(|e| {
                let (free, total) = self.device.memory_info().unwrap_or((0, 0));
                panic!(
                    "rms_norm failed: {:?}\n  input: shape={:?}\n  weight: shape={:?}\n  device memory: {:.2} MB free / {:.2} MB total",
                    e, input.shape(), weight.shape(), free as f64 / 1e6, total as f64 / 1e6,
                );
            });

        out
    }

    pub fn qk_norm(
        &self,
        qk: DeviceTensor,
        weight: &WeightTensor,
        _num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> DeviceTensor {
        self.ensure_device_context();

        let total_hidden: usize = qk.shape().iter().product();
        let n_groups = total_hidden / head_dim;

        let reshaped_shape = [1i64, n_groups as i64, head_dim as i64];
        let acl_x = AclTensor::from_ptr(&reshaped_shape, to_acl_dtype(qk.dtype()), qk.ptr())
            .expect("qk_norm: input tensor");
        let acl_w = Self::wrap_weight(weight);

        let out_bytes = total_hidden * 2; // FP16
        let out_buf = DeviceBuffer::alloc(out_bytes).expect("qk_norm: alloc output");
        let mut acl_y = AclTensor::new(&reshaped_shape, to_acl_dtype(qk.dtype()), &out_buf)
            .expect("qk_norm: output tensor");

        ascend::ops::rmsnorm::rmsnorm(&self.stream, &acl_x, &acl_w, eps as f64, &mut acl_y)
            .unwrap_or_else(|e| {
                panic!(
                    "qk_norm: aclnnRmsNorm failed: {:?}\n  shape={:?}, reshaped={:?}, n_groups={}, head_dim={}",
                    e, qk.shape(), reshaped_shape, n_groups, head_dim
                );
            });

        // qk is consumed (dropped here), freeing its old buffer.
        DeviceTensor::from_buf(qk.shape().to_vec(), qk.dtype(), "qk_norm_out", out_buf)
    }

    pub fn rotary_embedding(
        &self,
        q: DeviceTensor,
        k: DeviceTensor,
        positions: &[u32],
        rope_theta: f64,
        head_dim: usize,
    ) -> (DeviceTensor, DeviceTensor) {
        self.ensure_device_context();

        let seq_len = positions.len();
        let q_hidden = *q.shape().last().unwrap();
        let k_hidden = *k.shape().last().unwrap();
        let num_q_heads = q_hidden / head_dim;
        let num_kv_heads = k_hidden / head_dim;
        let batch = q.shape()[0];

        // Precompute cos/sin tables
        let half_dim = head_dim / 2;
        let table_size = seq_len * head_dim;
        let mut cos_table = vec![0.0f32; table_size];
        let mut sin_table = vec![0.0f32; table_size];

        for (s, &pos) in positions.iter().enumerate() {
            for i in 0..half_dim {
                let freq = (pos as f64) / rope_theta.powf(2.0 * i as f64 / head_dim as f64);
                let cos_val = freq.cos() as f32;
                let sin_val = freq.sin() as f32;
                cos_table[s * head_dim + i] = cos_val;
                cos_table[s * head_dim + half_dim + i] = cos_val;
                sin_table[s * head_dim + i] = sin_val;
                sin_table[s * head_dim + half_dim + i] = sin_val;
            }
        }

        let cos_f16: Vec<u16> = cos_table
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let sin_f16: Vec<u16> = sin_table
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();

        let cos_bytes =
            unsafe { std::slice::from_raw_parts(cos_f16.as_ptr() as *const u8, cos_f16.len() * 2) };
        let sin_bytes =
            unsafe { std::slice::from_raw_parts(sin_f16.as_ptr() as *const u8, sin_f16.len() * 2) };

        let mut cos_buf = DeviceBuffer::alloc(cos_bytes.len()).expect("rope: alloc cos");
        cos_buf.copy_from_host(cos_bytes).expect("rope: upload cos");
        let mut sin_buf = DeviceBuffer::alloc(sin_bytes.len()).expect("rope: alloc sin");
        sin_buf.copy_from_host(sin_bytes).expect("rope: upload sin");

        let cs_shape = [1i64, seq_len as i64, 1i64, head_dim as i64];
        let acl_cos = AclTensor::from_ptr(&cs_shape, to_acl_dtype(DType::Float16), cos_buf.ptr())
            .expect("rope: cos tensor");
        let acl_sin = AclTensor::from_ptr(&cs_shape, to_acl_dtype(DType::Float16), sin_buf.ptr())
            .expect("rope: sin tensor");

        // Create 4D views of input Q/K
        let make_4d = |dt: &DeviceTensor, n_heads: usize| -> AclTensor {
            let shape_4d = [
                batch as i64,
                seq_len as i64,
                n_heads as i64,
                head_dim as i64,
            ];
            let strides = [
                (seq_len * n_heads * head_dim) as i64,
                (n_heads * head_dim) as i64,
                head_dim as i64,
                1i64,
            ];
            AclTensor::from_ptr_with_strides(
                &shape_4d,
                &strides,
                to_acl_dtype(dt.dtype()),
                dt.ptr(),
            )
            .expect("rope: 4D view")
        };

        let acl_q = make_4d(&q, num_q_heads);
        let acl_k = make_4d(&k, num_kv_heads);

        // Allocate outputs
        let q_out_buf = DeviceBuffer::alloc(q.size_bytes()).expect("rope: alloc q_out");
        let q_out_shape = [
            batch as i64,
            seq_len as i64,
            num_q_heads as i64,
            head_dim as i64,
        ];
        let mut acl_q_out =
            AclTensor::new(&q_out_shape, to_acl_dtype(q.dtype()), &q_out_buf).expect("rope: q_out");

        let k_out_buf = DeviceBuffer::alloc(k.size_bytes()).expect("rope: alloc k_out");
        let k_out_shape = [
            batch as i64,
            seq_len as i64,
            num_kv_heads as i64,
            head_dim as i64,
        ];
        let mut acl_k_out =
            AclTensor::new(&k_out_shape, to_acl_dtype(k.dtype()), &k_out_buf).expect("rope: k_out");

        ascend::ops::rope::rotary_position_embedding(
            &self.stream,
            &acl_q,
            &acl_cos,
            &acl_sin,
            0,
            &mut acl_q_out,
        )
        .expect("rope: Q failed");

        ascend::ops::rope::rotary_position_embedding(
            &self.stream,
            &acl_k,
            &acl_cos,
            &acl_sin,
            0,
            &mut acl_k_out,
        )
        .expect("rope: K failed");

        // q and k consumed (dropped), cos/sin buffers dropped → aclrtFree ✓
        let q_out = DeviceTensor::from_buf(q.shape().to_vec(), q.dtype(), "q_rope", q_out_buf);
        let k_out = DeviceTensor::from_buf(k.shape().to_vec(), k.dtype(), "k_rope", k_out_buf);

        (q_out, k_out)
    }

    pub fn attention(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> DeviceTensor {
        self.ensure_device_context();

        let batch = q.shape()[0];
        let seq_len = q.shape()[1];

        let acl_q = Self::wrap_device(q);
        let acl_k = Self::wrap_device(k);
        let acl_v = Self::wrap_device(v);

        // Sync stream to ensure all prior ops finished
        self.stream.synchronize().ok();

        // Causal mask
        let mask_shape = [1i64, 1, seq_len as i64, seq_len as i64];
        let mask_numel = seq_len * seq_len;
        let mut host_mask = vec![0u8; mask_numel];
        for row in 0..seq_len {
            for col in (row + 1)..seq_len {
                host_mask[row * seq_len + col] = 1;
            }
        }
        let mut mask_buf = DeviceBuffer::alloc(mask_numel).expect("attention: alloc mask");
        mask_buf
            .copy_from_host(&host_mask)
            .expect("attention: upload mask");
        let acl_mask = AclTensor::from_ptr(
            &mask_shape,
            aclnn_sys::common::AclDataType::Bool,
            mask_buf.ptr(),
        )
        .expect("attention: mask tensor");

        // Auxiliary softmax buffers
        let aux_shape = [batch as i64, num_heads as i64, seq_len as i64, 8i64];
        let aux_bytes = batch * num_heads * seq_len * 8 * std::mem::size_of::<f32>();

        let sm_max_buf = DeviceBuffer::alloc(aux_bytes).expect("attention: alloc sm_max");
        let acl_sm_max = AclTensor::new(
            &aux_shape,
            aclnn_sys::common::AclDataType::Float,
            &sm_max_buf,
        )
        .expect("attention: sm_max tensor");

        let sm_sum_buf = DeviceBuffer::alloc(aux_bytes).expect("attention: alloc sm_sum");
        let acl_sm_sum = AclTensor::new(
            &aux_shape,
            aclnn_sys::common::AclDataType::Float,
            &sm_sum_buf,
        )
        .expect("attention: sm_sum tensor");

        // Output
        let out = DeviceTensor::alloc(q.shape().to_vec(), q.dtype(), "attn_out")
            .expect("attention: alloc output");
        let mut acl_out = Self::wrap_device(&out);

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
            let (free, total) = self.device.memory_info().unwrap_or((0, 0));
            panic!(
                "attention failed: {:?}\n  batch={}, seq_len={}, heads={}, kv_heads={}, head_dim={}\n  device memory: {:.2} MB free / {:.2} MB total",
                e, batch, seq_len, num_heads, num_kv_heads, head_dim,
                free as f64 / 1e6, total as f64 / 1e6,
            );
        });

        // mask_buf, sm_max_buf, sm_sum_buf dropped here → aclrtFree ✓
        out
    }

    pub fn silu_mul(&self, gate: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor {
        self.ensure_device_context();

        let acl_gate = Self::wrap_device(gate);
        let silu_buf = DeviceBuffer::alloc(gate.size_bytes()).expect("silu_mul: alloc silu");
        let mut acl_silu = AclTensor::new(
            &shape_i64(gate.shape()),
            to_acl_dtype(gate.dtype()),
            &silu_buf,
        )
        .expect("silu_mul: silu tensor");

        ascend::ops::activation::silu(&self.stream, &acl_gate, &mut acl_silu)
            .expect("silu_mul: aclnnSilu failed");

        let acl_up = Self::wrap_device(up);
        let out = DeviceTensor::alloc(gate.shape().to_vec(), gate.dtype(), "silu_out")
            .expect("silu_mul: alloc output");
        let mut acl_out = Self::wrap_device(&out);

        ascend::ops::elementwise::mul(&self.stream, &acl_silu, &acl_up, &mut acl_out)
            .expect("silu_mul: aclnnMul failed");

        // silu_buf dropped here → aclrtFree ✓
        out
    }

    pub fn embedding(&self, ids: &[u32], table: &WeightTensor) -> DeviceTensor {
        self.ensure_device_context();

        let ids_i64: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let ids_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(ids_i64.as_ptr() as *const u8, ids_i64.len() * 8) };
        let mut ids_buf = DeviceBuffer::alloc(ids_bytes.len()).expect("embedding: alloc ids");
        ids_buf
            .copy_from_host(ids_bytes)
            .expect("embedding: upload ids");

        let ids_shape = [1i64, ids.len() as i64];
        let ids_acl = AclTensor::from_ptr(
            &ids_shape,
            aclnn_sys::common::AclDataType::Int64,
            ids_buf.ptr(),
        )
        .expect("embedding: ids tensor");

        let acl_table = Self::wrap_weight(table);

        let embed_dim = table.shape()[1];
        let out = DeviceTensor::alloc(vec![1, ids.len(), embed_dim], DType::Float16, "embed_out")
            .expect("embedding: alloc output");
        let mut acl_out = Self::wrap_device(&out);

        ascend::ops::embedding::embedding(&self.stream, &acl_table, &ids_acl, &mut acl_out)
            .expect("embedding: aclnnEmbedding failed");

        // ids_buf dropped here → aclrtFree ✓
        out
    }

    pub fn add(&self, a: DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        self.ensure_device_context();

        let acl_a = Self::wrap_device(&a);
        let acl_b = Self::wrap_device(b);

        // In-place add on a's buffer
        ascend::ops::elementwise::inplace_add(&self.stream, &acl_a, &acl_b, 1.0)
            .expect("add: aclnnInplaceAdd failed");

        // a is consumed and returned (same buffer, no new allocation)
        a
    }

    pub fn sample_argmax(&self, logits: &DeviceTensor) -> u32 {
        self.ensure_device_context();

        let acl_logits = Self::wrap_device(logits);

        let last_dim = logits.shape().len() - 1;
        let out_shape: Vec<i64> = logits.shape()[..last_dim]
            .iter()
            .map(|&d| d as i64)
            .collect();
        let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
        let out_bytes = out_numel * std::mem::size_of::<i64>();

        let out_buf = DeviceBuffer::alloc(out_bytes).expect("sample_argmax: alloc");
        let mut acl_out = AclTensor::from_ptr(
            &out_shape,
            aclnn_sys::common::AclDataType::Int64,
            out_buf.ptr(),
        )
        .expect("sample_argmax: output tensor");

        ascend::ops::reduction::argmax(
            &self.stream,
            &acl_logits,
            last_dim as i64,
            false,
            &mut acl_out,
        )
        .expect("sample_argmax: aclnnArgMax failed");

        self.stream.synchronize().expect("sample_argmax: sync");
        let mut results = vec![0i64; out_numel];
        out_buf
            .copy_to_host(unsafe {
                std::slice::from_raw_parts_mut(results.as_mut_ptr() as *mut u8, out_bytes)
            })
            .expect("sample_argmax: copy to host");

        // out_buf dropped here → aclrtFree ✓
        *results.last().unwrap_or(&0) as u32
    }

    /// Write K/V tokens into the paged KV cache at the specified slot positions.
    ///
    /// For each token i in [0..num_tokens), copies the K/V data from
    /// the projection output into the cache at slot `slot_mapping[i]`.
    ///
    /// K/V shape: [1, num_tokens, num_kv_heads * head_dim] (BSH layout)
    /// Cache shape: [num_blocks * block_size, num_kv_heads, head_dim] (flattened)
    pub fn reshape_and_cache(
        &self,
        k: &DeviceTensor,           // [1, seq_len, num_kv_heads * head_dim]
        v: &DeviceTensor,           // same
        key_cache: &DeviceBuffer,   // [num_blocks * block_size * num_kv_heads * head_dim] FP16
        value_cache: &DeviceBuffer, // same
        slot_mapping: &[i32],       // [num_tokens]
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.ensure_device_context();

        let token_size = num_kv_heads * head_dim * 2; // FP16 = 2 bytes per element

        // Device-to-Device async copy: scatter K/V tokens into cache slots.
        // All copies are enqueued on the same stream as the preceding QKV/RotaryEmb
        // operations, so data dependencies are automatically satisfied — no
        // stream.synchronize() needed.
        for (token_idx, &slot) in slot_mapping.iter().enumerate() {
            let src_offset = token_idx * token_size;
            let dst_offset = slot as usize * token_size;

            // K: source → cache slot
            let k_src = unsafe { (k.ptr() as usize + src_offset) as *const std::os::raw::c_void };
            let k_dst =
                unsafe { (key_cache.ptr() as usize + dst_offset) as *mut std::os::raw::c_void };
            let status = unsafe {
                ascendcl_sys::aclrtMemcpyAsync(
                    k_dst,
                    token_size,
                    k_src,
                    token_size,
                    ascendcl_sys::AclrtMemcpyKind::DeviceToDevice,
                    self.stream.raw(),
                )
            };
            assert_eq!(
                status, 0,
                "reshape_and_cache: K aclrtMemcpyAsync failed (slot={})",
                slot
            );

            // V: source → cache slot
            let v_src = unsafe { (v.ptr() as usize + src_offset) as *const std::os::raw::c_void };
            let v_dst =
                unsafe { (value_cache.ptr() as usize + dst_offset) as *mut std::os::raw::c_void };
            let status = unsafe {
                ascendcl_sys::aclrtMemcpyAsync(
                    v_dst,
                    token_size,
                    v_src,
                    token_size,
                    ascendcl_sys::AclrtMemcpyKind::DeviceToDevice,
                    self.stream.raw(),
                )
            };
            assert_eq!(
                status, 0,
                "reshape_and_cache: V aclrtMemcpyAsync failed (slot={})",
                slot
            );
        }
    }

    /// Paged decode attention using IncreFlashAttentionV4.
    ///
    /// Uses pre-allocated `DecodeBuffers` to avoid per-call malloc/free.
    /// Only lightweight operations remain: block_table `copy_from_host`,
    /// `aclCreateIntArray`, and list-owned tensor descriptor creation.
    pub fn paged_decode_attention(
        &self,
        q: &DeviceTensor,
        key_cache: &DeviceBuffer,
        value_cache: &DeviceBuffer,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
        block_table_host: &[i32],
        max_blocks_per_seq: usize,
        actual_seq_len: usize,
        decode_bufs: &mut DecodeBuffers,
    ) -> DeviceTensor {
        self.ensure_device_context();

        // BSH cache shape: [1, num_blocks * block_size, num_kv_heads * head_dim]
        let total_slots = num_blocks * block_size;
        let kv_hidden = num_kv_heads * head_dim;
        let cache_shape_bsh = [1i64, total_slots as i64, kv_hidden as i64];

        let acl_q = Self::wrap_device(q);

        // Upload block_table to pre-allocated device buffer (no malloc/free)
        let bt_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                block_table_host.as_ptr() as *const u8,
                block_table_host.len() * 4,
            )
        };
        assert!(
            max_blocks_per_seq <= decode_bufs.max_blocks_capacity,
            "block_table exceeds pre-allocated capacity: {} > {}",
            max_blocks_per_seq,
            decode_bufs.max_blocks_capacity,
        );
        decode_bufs
            .block_table_buf
            .copy_from_host(bt_bytes)
            .expect("paged_attn: upload block_table");
        let bt_shape = [1i64, max_blocks_per_seq as i64];
        let acl_bt = AclTensor::from_ptr(
            &bt_shape,
            AclDataType::Int32,
            decode_bufs.block_table_buf.ptr(),
        )
        .expect("paged_attn: block_table tensor");

        // actual_seq_lengths IntArray (lightweight host metadata)
        let seq_lens = [actual_seq_len as i64];
        let actual_seq_arr = unsafe { aclnn_sys::common::aclCreateIntArray(seq_lens.as_ptr(), 1) };

        // Allocate attention output (pool takes ownership, can't pre-alloc)
        let out = DeviceTensor::alloc(q.shape().to_vec(), q.dtype(), "paged_attn_out")
            .expect("paged_attn: alloc output");
        let mut acl_out = Self::wrap_device(&out);

        let scale = 1.0 / (head_dim as f64).sqrt();

        ascend::ops::paged_attention::paged_attention_decode(
            &self.stream,
            &acl_q,
            key_cache.ptr(),
            value_cache.ptr(),
            &cache_shape_bsh,
            actual_seq_arr,
            num_heads as i64,
            num_kv_heads as i64,
            scale,
            block_size as i64,
            &acl_bt,
            &mut acl_out,
        ).unwrap_or_else(|e| {
            panic!(
                "paged_decode_attention failed: {:?}\n  Q={:?}, KV_cache=[1,{},{}], heads={}, kv_heads={}, head_dim={}, block_size={}, num_blocks={}, seq_len={}",
                e, q.shape(), total_slots, kv_hidden, num_heads, num_kv_heads, head_dim, block_size, num_blocks, actual_seq_len,
            );
        });

        unsafe {
            aclnn_sys::common::aclDestroyIntArray(actual_seq_arr);
        }

        out
    }
}

// ─── AscendCommOps ─────────────────────────────────────────────────────

use super::stubs::{CommOps, QuantOps};
use crate::model::tensor::Tensor;

/// Communication ops for Ascend multi-NPU (HCCL).
///
/// Placeholder — will be implemented using HCCL library.
pub struct AscendCommOps;

impl CommOps for AscendCommOps {
    fn all_reduce_sum(&self, tensor: &mut Tensor) {
        tracing::debug!("ascend::all_reduce_sum({}) [NOT YET IMPLEMENTED]", tensor);
    }
    fn all_gather(&self, input: &Tensor, out: &mut Tensor) {
        tracing::debug!(
            "ascend::all_gather({} -> {}) [NOT YET IMPLEMENTED]",
            input,
            out
        );
    }
    fn send(&self, tensor: &Tensor, dst_rank: usize) {
        tracing::debug!(
            "ascend::send({} -> rank {}) [NOT YET IMPLEMENTED]",
            tensor,
            dst_rank
        );
    }
    fn recv(&self, out: &mut Tensor, src_rank: usize) {
        tracing::debug!(
            "ascend::recv({} <- rank {}) [NOT YET IMPLEMENTED]",
            out,
            src_rank
        );
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
            input,
            weight,
            scales,
            out
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
            quantized,
            scales,
            out
        );
    }
}

// ─── OpsBundle Constructor ─────────────────────────────────────────────

use super::stubs::{ComputeOps, OpsBundle};

/// Stub ComputeOps impl (required for OpsBundle, but v2 path uses typed methods directly).
impl ComputeOps for AscendComputeOps {
    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) {
        panic!("Use typed matmul() instead of ComputeOps::matmul for Ascend v2 path");
    }
    fn rms_norm(&self, _input: &Tensor, _weight: &Tensor, _eps: f32, _out: &mut Tensor) {
        panic!("Use typed rms_norm() instead of ComputeOps::rms_norm for Ascend v2 path");
    }
    fn rotary_embedding(
        &self,
        _q: &mut Tensor,
        _k: &mut Tensor,
        _positions: &[u32],
        _rope_theta: f64,
        _head_dim: usize,
    ) {
        panic!("Use typed rotary_embedding() instead for Ascend v2 path");
    }
    fn qk_norm(
        &self,
        _qk: &mut Tensor,
        _weight: &Tensor,
        _num_heads: usize,
        _head_dim: usize,
        _eps: f32,
    ) {
        panic!("Use typed qk_norm() instead for Ascend v2 path");
    }
    fn attention(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _out: &mut Tensor,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) {
        panic!("Use typed attention() instead for Ascend v2 path");
    }
    fn silu_mul(&self, _gate: &Tensor, _up: &Tensor, _out: &mut Tensor) {
        panic!("Use typed silu_mul() instead for Ascend v2 path");
    }
    fn embedding(&self, _ids: &[u32], _table: &Tensor, _out: &mut Tensor) {
        panic!("Use typed embedding() instead for Ascend v2 path");
    }
    fn softmax(&self, _input: &Tensor, _out: &mut Tensor) {
        panic!("Use typed softmax() instead for Ascend v2 path");
    }
    fn add(&self, _a: &mut Tensor, _b: &Tensor) {
        panic!("Use typed add() instead for Ascend v2 path");
    }
    fn sample_argmax(&self, _logits: &Tensor) -> u32 {
        panic!("Use typed sample_argmax() instead for Ascend v2 path");
    }
}

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
