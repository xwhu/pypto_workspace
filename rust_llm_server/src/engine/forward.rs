use crate::model::config::Qwen3Config;
use crate::model::network::Qwen3Model;
use crate::model::tensor::{DType, Tensor};
use crate::ops::Ops;
use super::kv_cache::SequenceKVCache;

/// Forward pass through the Qwen3 model.
///
/// Walks the model network graph and calls operators in sequence:
/// 1. Token embedding lookup
/// 2. For each transformer layer:
///    a. input_layernorm(x) → normed
///    b. Q/K/V projection → q, k, v
///    c. RoPE on q, k
///    d. Attention(q, k, v) → attn_out
///    e. o_proj(attn_out) → attn_out
///    f. x = x + attn_out (residual)
///    g. post_attention_layernorm(x) → normed
///    h. gate_proj(normed), up_proj(normed) → SwiGLU → down_proj → ffn_out
///    i. x = x + ffn_out (residual)
/// 3. Final RMSNorm
/// 4. LM head → logits
pub struct ForwardPass<'a> {
    model: &'a Qwen3Model,
    ops: &'a dyn Ops,
}

impl<'a> ForwardPass<'a> {
    pub fn new(model: &'a Qwen3Model, ops: &'a dyn Ops) -> Self {
        Self { model, ops }
    }

    /// Run the forward pass for a batch of token IDs.
    ///
    /// Returns logits for the last token position (for next-token prediction).
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs: [seq_len]
    /// * `positions` - Position indices: [seq_len]
    /// * `kv_cache` - KV cache for this sequence (updated in place)
    pub fn forward(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        kv_cache: &mut SequenceKVCache,
    ) -> Tensor {
        let cfg = &self.model.config;
        let batch = 1;
        let seq_len = input_ids.len();

        // 1. Token embedding
        let mut hidden = Tensor::new(
            vec![batch, seq_len, cfg.hidden_size],
            DType::Float16,
            "hidden_states",
        );
        self.ops.embedding(input_ids, &self.model.embed_tokens, &mut hidden);

        // 2. Transformer layers
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            tracing::trace!("Forward: layer {layer_idx}/{}", self.model.num_layers());

            // 2a. Input layernorm
            let mut normed = hidden.with_shape(
                vec![batch, seq_len, cfg.hidden_size],
                format!("layer.{layer_idx}.normed"),
            );
            self.ops.rms_norm(
                &hidden,
                &layer.input_layernorm.weight,
                cfg.rms_norm_eps as f32,
                &mut normed,
            );

            // 2b. Q/K/V projection
            let mut q = Tensor::new(
                vec![batch, seq_len, cfg.num_attention_heads, cfg.head_dim],
                DType::Float16,
                format!("layer.{layer_idx}.q"),
            );
            let mut k = Tensor::new(
                vec![batch, seq_len, cfg.num_key_value_heads, cfg.head_dim],
                DType::Float16,
                format!("layer.{layer_idx}.k"),
            );
            let mut v = Tensor::new(
                vec![batch, seq_len, cfg.num_key_value_heads, cfg.head_dim],
                DType::Float16,
                format!("layer.{layer_idx}.v"),
            );

            self.ops.matmul(&normed, &layer.self_attn.q_proj, &mut q);
            self.ops.matmul(&normed, &layer.self_attn.k_proj, &mut k);
            self.ops.matmul(&normed, &layer.self_attn.v_proj, &mut v);

            // 2c. Rotary position embeddings
            self.ops.rotary_embedding(&mut q, &mut k, positions, cfg.rope_theta);

            // (Stub: In a real impl, k and v would be appended to kv_cache here)

            // 2d. Grouped-Query Attention
            let mut attn_out = Tensor::new(
                vec![batch, seq_len, cfg.num_attention_heads, cfg.head_dim],
                DType::Float16,
                format!("layer.{layer_idx}.attn_out"),
            );
            self.ops.attention(
                &q,
                &k,
                &v,
                &mut attn_out,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
            );

            // 2e. Output projection
            let mut proj_out = Tensor::new(
                vec![batch, seq_len, cfg.hidden_size],
                DType::Float16,
                format!("layer.{layer_idx}.o_proj_out"),
            );
            self.ops.matmul(&attn_out, &layer.self_attn.o_proj, &mut proj_out);

            // 2f. Residual connection
            self.ops.add(&mut hidden, &proj_out);

            // 2g. Post-attention layernorm
            let mut normed2 = hidden.with_shape(
                vec![batch, seq_len, cfg.hidden_size],
                format!("layer.{layer_idx}.normed2"),
            );
            self.ops.rms_norm(
                &hidden,
                &layer.post_attention_layernorm.weight,
                cfg.rms_norm_eps as f32,
                &mut normed2,
            );

            // 2h. MLP (SwiGLU)
            let mut gate = Tensor::new(
                vec![batch, seq_len, cfg.intermediate_size],
                DType::Float16,
                format!("layer.{layer_idx}.gate"),
            );
            let mut up = Tensor::new(
                vec![batch, seq_len, cfg.intermediate_size],
                DType::Float16,
                format!("layer.{layer_idx}.up"),
            );
            self.ops.matmul(&normed2, &layer.mlp.gate_proj, &mut gate);
            self.ops.matmul(&normed2, &layer.mlp.up_proj, &mut up);

            let mut silu_out = Tensor::new(
                vec![batch, seq_len, cfg.intermediate_size],
                DType::Float16,
                format!("layer.{layer_idx}.silu_out"),
            );
            self.ops.silu_mul(&gate, &up, &mut silu_out);

            let mut ffn_out = Tensor::new(
                vec![batch, seq_len, cfg.hidden_size],
                DType::Float16,
                format!("layer.{layer_idx}.ffn_out"),
            );
            self.ops.matmul(&silu_out, &layer.mlp.down_proj, &mut ffn_out);

            // 2i. Residual connection
            self.ops.add(&mut hidden, &ffn_out);
        }

        // Update KV cache length
        kv_cache.append(seq_len);

        // 3. Final RMSNorm
        let mut final_normed = hidden.with_shape(
            vec![batch, seq_len, cfg.hidden_size],
            "final_normed",
        );
        self.ops.rms_norm(
            &hidden,
            &self.model.norm.weight,
            cfg.rms_norm_eps as f32,
            &mut final_normed,
        );

        // 4. LM head → logits (only for last token position)
        let mut logits = Tensor::new(
            vec![batch, 1, cfg.vocab_size],
            DType::Float16,
            "logits",
        );
        self.ops.matmul(&final_normed, &self.model.lm_head, &mut logits);

        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::StubOps;

    #[test]
    fn test_forward_pass_no_panic() {
        let config = Qwen3Config::qwen3_0_6b(); // use small model for test
        let model = Qwen3Model::new(config.clone());
        let ops = StubOps;
        let fwd = ForwardPass::new(&model, &ops);

        let mut kv_cache = SequenceKVCache::new(&config, 2048);
        let input_ids = vec![1u32, 2, 3, 4, 5];
        let positions: Vec<u32> = (0..5).collect();

        let logits = fwd.forward(&input_ids, &positions, &mut kv_cache);

        assert_eq!(logits.shape, vec![1, 1, config.vocab_size]);
        assert_eq!(kv_cache.current_len(), 5);
    }
}
