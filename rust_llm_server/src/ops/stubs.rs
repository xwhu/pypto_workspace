use crate::model::tensor::Tensor;

/// Core compute operations for LLM inference.
///
/// This trait covers pure mathematical operations that run on a single
/// device (GPU/NPU). It does NOT include communication or quantization
/// operations — those are in separate traits.
///
/// Each method here maps to one or more `aclnn*` / CUDA kernel calls
/// in a real implementation.
pub trait ComputeOps: Send + Sync {
    /// Matrix multiplication: out = a @ b.
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor);

    /// RMSNorm: out = x * weight / rms(x, eps).
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor);

    /// Apply rotary position embeddings in-place to Q and K.
    fn rotary_embedding(&self, q: &mut Tensor, k: &mut Tensor, positions: &[u32], rope_theta: f64, head_dim: usize);

    /// Grouped-Query Attention.
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    );

    /// SwiGLU activation: out = silu(gate) * up.
    fn silu_mul(&self, gate: &Tensor, up: &Tensor, out: &mut Tensor);

    /// Token embedding lookup: out[i] = table[ids[i]].
    fn embedding(&self, ids: &[u32], table: &Tensor, out: &mut Tensor);

    /// Softmax over the last dimension.
    fn softmax(&self, input: &Tensor, out: &mut Tensor);

    /// Residual add: out = a + b (in-place on a).
    fn add(&self, a: &mut Tensor, b: &Tensor);

    /// Argmax sampling from logits. Returns the token ID.
    fn sample_argmax(&self, logits: &Tensor) -> u32;
}

/// Communication operations for multi-device parallelism.
///
/// Used when tensor parallelism (TP) or pipeline parallelism (PP) is active.
/// In single-device mode, these are no-ops.
pub trait CommOps: Send + Sync {
    /// All-reduce (sum) a tensor across all TP ranks in-place.
    fn all_reduce_sum(&self, tensor: &mut Tensor);

    /// All-gather a tensor across all TP ranks.
    /// Input shape on each rank: [N, local_dim]
    /// Output shape: [N, local_dim * tp_size]
    fn all_gather(&self, input: &Tensor, out: &mut Tensor);

    /// Point-to-point send to a specific rank (for PP).
    fn send(&self, tensor: &Tensor, dst_rank: usize);

    /// Point-to-point receive from a specific rank (for PP).
    fn recv(&self, out: &mut Tensor, src_rank: usize);
}

/// Quantization-aware operations.
///
/// Used when model weights are stored in reduced precision (INT8/INT4).
/// These operations handle dequantization and mixed-precision compute.
pub trait QuantOps: Send + Sync {
    /// Quantized matrix multiplication.
    /// Input is FP16/BF16 activation, weight is quantized.
    /// Handles dequantization internally for best performance.
    fn matmul_quantized(
        &self,
        input: &Tensor,
        weight: &Tensor,
        scales: &Tensor,
        zeros: Option<&Tensor>,
        out: &mut Tensor,
    );

    /// Explicit dequantization: convert quantized tensor to FP16.
    fn dequantize(
        &self,
        quantized: &Tensor,
        scales: &Tensor,
        zeros: Option<&Tensor>,
        out: &mut Tensor,
    );
}

/// Bundle of all ops implementations for a given backend.
///
/// Holds one implementation for each ops trait. The execution plan's
/// compiled closures capture references to this bundle.
pub struct OpsBundle {
    pub compute: Box<dyn ComputeOps>,
    pub comm: Box<dyn CommOps>,
    pub quant: Box<dyn QuantOps>,
}

// ─── Stub Implementations ──────────────────────────────────────────────

/// Stub compute ops — all operations are no-ops with logging.
pub struct StubComputeOps;

impl ComputeOps for StubComputeOps {
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) {
        tracing::debug!("stub::matmul({} @ {} -> {})", a, b, out);
    }
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor) {
        tracing::debug!("stub::rms_norm({}, w={}, eps={eps}) -> {}", input, weight, out);
    }
    fn rotary_embedding(&self, q: &mut Tensor, k: &mut Tensor, positions: &[u32], rope_theta: f64, head_dim: usize) {
        tracing::debug!("stub::rotary_embedding({}, {}, pos_len={}, theta={rope_theta}, d={head_dim})", q, k, positions.len());
    }
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, out: &mut Tensor, num_heads: usize, num_kv_heads: usize, head_dim: usize) {
        tracing::debug!("stub::attention(q={}, k={}, v={} -> {}, h={num_heads}, kv={num_kv_heads}, d={head_dim})", q, k, v, out);
    }
    fn silu_mul(&self, gate: &Tensor, up: &Tensor, out: &mut Tensor) {
        tracing::debug!("stub::silu_mul({}, {} -> {})", gate, up, out);
    }
    fn embedding(&self, ids: &[u32], table: &Tensor, out: &mut Tensor) {
        tracing::debug!("stub::embedding(ids_len={}, table={} -> {})", ids.len(), table, out);
    }
    fn softmax(&self, input: &Tensor, out: &mut Tensor) {
        tracing::debug!("stub::softmax({} -> {})", input, out);
    }
    fn add(&self, a: &mut Tensor, b: &Tensor) {
        tracing::debug!("stub::add({} += {})", a, b);
    }
    fn sample_argmax(&self, logits: &Tensor) -> u32 {
        tracing::debug!("stub::sample_argmax({}) -> 0", logits);
        0
    }
}

/// Stub communication ops — no-ops for single-device mode.
pub struct StubCommOps;

impl CommOps for StubCommOps {
    fn all_reduce_sum(&self, tensor: &mut Tensor) {
        tracing::debug!("stub::all_reduce_sum({})", tensor);
    }
    fn all_gather(&self, input: &Tensor, out: &mut Tensor) {
        tracing::debug!("stub::all_gather({} -> {})", input, out);
    }
    fn send(&self, tensor: &Tensor, dst_rank: usize) {
        tracing::debug!("stub::send({} -> rank {})", tensor, dst_rank);
    }
    fn recv(&self, out: &mut Tensor, src_rank: usize) {
        tracing::debug!("stub::recv({} <- rank {})", out, src_rank);
    }
}

/// Stub quantization ops — no-ops.
pub struct StubQuantOps;

impl QuantOps for StubQuantOps {
    fn matmul_quantized(&self, input: &Tensor, weight: &Tensor, scales: &Tensor, _zeros: Option<&Tensor>, out: &mut Tensor) {
        tracing::debug!("stub::matmul_quantized({} @ {} [s={}] -> {})", input, weight, scales, out);
    }
    fn dequantize(&self, quantized: &Tensor, scales: &Tensor, _zeros: Option<&Tensor>, out: &mut Tensor) {
        tracing::debug!("stub::dequantize({} [s={}] -> {})", quantized, scales, out);
    }
}

impl OpsBundle {
    /// Create a bundle of all-stub implementations.
    pub fn stub() -> Self {
        Self {
            compute: Box::new(StubComputeOps),
            comm: Box::new(StubCommOps),
            quant: Box::new(StubQuantOps),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tensor::DType;

    #[test]
    fn test_stub_compute_ops_no_panic() {
        let ops = StubComputeOps;
        let a = Tensor::new(vec![1, 4096], DType::Float16, "a");
        let b = Tensor::new(vec![4096, 4096], DType::Float16, "b");
        let mut out = Tensor::new(vec![1, 4096], DType::Float16, "out");
        ops.matmul(&a, &b, &mut out);
        ops.rms_norm(&a, &Tensor::new(vec![4096], DType::Float16, "w"), 1e-6, &mut out);
        assert_eq!(ops.sample_argmax(&a), 0);
    }

    #[test]
    fn test_stub_comm_ops_no_panic() {
        let ops = StubCommOps;
        let mut t = Tensor::new(vec![1, 4096], DType::Float16, "t");
        ops.all_reduce_sum(&mut t);
        ops.send(&t, 1);
    }

    #[test]
    fn test_ops_bundle() {
        let bundle = OpsBundle::stub();
        let a = Tensor::new(vec![1, 4096], DType::Float16, "a");
        let b = Tensor::new(vec![4096, 4096], DType::Float16, "b");
        let mut out = Tensor::new(vec![1, 4096], DType::Float16, "out");
        bundle.compute.matmul(&a, &b, &mut out);
        bundle.comm.all_reduce_sum(&mut out);
    }
}
