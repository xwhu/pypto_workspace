use super::kv_cache::SequenceKVCache;
use crate::model::network::Qwen3Model;
use crate::model::tensor::{DType, Tensor};
use crate::ops::ComputeOps;

/// Forward pass through the Qwen3 model (legacy direct-execution mode).
///
/// This walks the model network graph and calls operators in sequence.
/// It is preserved as a reference implementation and for simple testing.
/// For production, use `CompiledPlan` from `plan.rs` instead.
#[allow(dead_code)]
pub struct ForwardPass<'a> {
    model: &'a Qwen3Model,
    ops: &'a dyn ComputeOps,
}

#[allow(dead_code)]
impl<'a> ForwardPass<'a> {
    pub fn new(model: &'a Qwen3Model, ops: &'a dyn ComputeOps) -> Self {
        Self { model, ops }
    }

    /// Run the forward pass for a batch of token IDs.
    /// Returns logits for next-token prediction.
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
        self.ops
            .embedding(input_ids, &self.model.embed_tokens, &mut hidden);

        // 2. Transformer layers
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            tracing::trace!("Forward: layer {layer_idx}/{}", self.model.num_layers());

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
            self.ops
                .rotary_embedding(&mut q, &mut k, positions, cfg.rope_theta, cfg.head_dim);

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

            let mut proj_out = Tensor::new(
                vec![batch, seq_len, cfg.hidden_size],
                DType::Float16,
                format!("layer.{layer_idx}.o_proj_out"),
            );
            self.ops
                .matmul(&attn_out, &layer.self_attn.o_proj, &mut proj_out);
            self.ops.add(&mut hidden, &proj_out);

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
            self.ops
                .matmul(&silu_out, &layer.mlp.down_proj, &mut ffn_out);
            self.ops.add(&mut hidden, &ffn_out);
        }

        kv_cache.append(seq_len);

        let mut final_normed =
            hidden.with_shape(vec![batch, seq_len, cfg.hidden_size], "final_normed");
        self.ops.rms_norm(
            &hidden,
            &self.model.norm.weight,
            cfg.rms_norm_eps as f32,
            &mut final_normed,
        );

        let mut logits = Tensor::new(vec![batch, 1, cfg.vocab_size], DType::Float16, "logits");
        self.ops
            .matmul(&final_normed, &self.model.lm_head, &mut logits);

        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::Qwen3Config;
    use crate::ops::stubs::StubComputeOps;

    #[test]
    fn test_forward_pass_no_panic() {
        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config.clone());
        let ops = StubComputeOps;
        let fwd = ForwardPass::new(&model, &ops);

        let mut kv_cache = SequenceKVCache::new(&config, 2048);
        let input_ids = vec![1u32, 2, 3, 4, 5];
        let positions: Vec<u32> = (0..5).collect();

        let logits = fwd.forward(&input_ids, &positions, &mut kv_cache);

        assert_eq!(logits.shape, vec![1, 1, config.vocab_size]);
        assert_eq!(kv_cache.current_len(), 5);
    }
}
