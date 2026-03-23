use crate::model::tensor::Tensor;

/// Trait defining all compute operations needed for LLM inference.
///
/// This is the swappable operator layer. The `StubOps` implementation
/// does nothing (no-ops with logging). Real implementations would call
/// AscendCL/CUDA/etc. via FFI.
///
/// Design note: This trait mirrors the operator set that `torch_npu` and
/// vLLM Ascend use via the `aclnn*` two-stage APIs. Each method here
/// corresponds to one or more `aclnn*` calls in a real implementation.
pub trait Ops: Send + Sync {
    /// Matrix multiplication: out = a @ b.
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor);

    /// RMSNorm: out = x * weight / rms(x, eps).
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor);

    /// Apply rotary position embeddings in-place to Q and K.
    fn rotary_embedding(&self, q: &mut Tensor, k: &mut Tensor, positions: &[u32], rope_theta: f64);

    /// Grouped-Query Attention.
    ///
    /// Computes scaled dot-product attention with GQA layout:
    /// - q: [batch, seq_len, num_heads, head_dim]
    /// - k: [batch, kv_len, num_kv_heads, head_dim]
    /// - v: [batch, kv_len, num_kv_heads, head_dim]
    /// - out: [batch, seq_len, num_heads, head_dim]
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

/// Stub operator implementation — all operations are no-ops.
///
/// Used for framework development and testing without a real accelerator.
/// Each method logs the operation name and tensor shapes.
pub struct StubOps;

impl Ops for StubOps {
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) {
        tracing::debug!(
            "stub::matmul({} @ {} -> {})",
            a, b, out
        );
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor) {
        tracing::debug!(
            "stub::rms_norm({}, w={}, eps={eps}) -> {}",
            input, weight, out
        );
    }

    fn rotary_embedding(&self, q: &mut Tensor, k: &mut Tensor, positions: &[u32], rope_theta: f64) {
        tracing::debug!(
            "stub::rotary_embedding({}, {}, pos_len={}, theta={rope_theta})",
            q, k, positions.len()
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
        tracing::debug!(
            "stub::attention(q={}, k={}, v={} -> {}, heads={num_heads}, kv_heads={num_kv_heads}, d={head_dim})",
            q, k, v, out
        );
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor, out: &mut Tensor) {
        tracing::debug!("stub::silu_mul({}, {} -> {})", gate, up, out);
    }

    fn embedding(&self, ids: &[u32], table: &Tensor, out: &mut Tensor) {
        tracing::debug!(
            "stub::embedding(ids_len={}, table={} -> {})",
            ids.len(), table, out
        );
    }

    fn softmax(&self, input: &Tensor, out: &mut Tensor) {
        tracing::debug!("stub::softmax({} -> {})", input, out);
    }

    fn add(&self, a: &mut Tensor, b: &Tensor) {
        tracing::debug!("stub::add({} += {})", a, b);
    }

    fn sample_argmax(&self, logits: &Tensor) -> u32 {
        tracing::debug!("stub::sample_argmax({}) -> 0", logits);
        // Return a fixed token (e.g., EOS-like) for stub
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tensor::DType;

    #[test]
    fn test_stub_ops_no_panic() {
        let ops = StubOps;
        let a = Tensor::new(vec![1, 4096], DType::Float16, "a");
        let b = Tensor::new(vec![4096, 4096], DType::Float16, "b");
        let mut out = Tensor::new(vec![1, 4096], DType::Float16, "out");

        ops.matmul(&a, &b, &mut out);
        ops.rms_norm(&a, &Tensor::new(vec![4096], DType::Float16, "w"), 1e-6, &mut out);
        let token = ops.sample_argmax(&a);
        assert_eq!(token, 0);
    }
}
