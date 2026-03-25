use super::config::Qwen3Config;
use super::tensor::{DType, Tensor};

// ─── Weight Descriptors ────────────────────────────────────────────────

/// Weights for RMSNorm.
#[derive(Debug)]
pub struct RMSNormWeights {
    pub weight: Tensor, // [hidden_size]
}

/// Weights for Qwen3 attention (Q/K/V projections + output projection + QK norm).
/// Shape convention: PyTorch nn.Linear [out_features, in_features].
#[derive(Debug)]
pub struct Qwen3AttentionWeights {
    pub q_proj: Tensor, // [num_heads * head_dim, hidden_size]
    pub k_proj: Tensor, // [num_kv_heads * head_dim, hidden_size]
    pub v_proj: Tensor, // [num_kv_heads * head_dim, hidden_size]
    pub o_proj: Tensor, // [hidden_size, num_heads * head_dim]
    pub q_norm: Tensor, // [head_dim] — per-head RMS norm on Q
    pub k_norm: Tensor, // [head_dim] — per-head RMS norm on K
}

/// Weights for Qwen3 MLP (SwiGLU: gate_proj, up_proj, down_proj).
/// Shape convention: PyTorch nn.Linear [out_features, in_features].
#[derive(Debug)]
pub struct Qwen3MLPWeights {
    pub gate_proj: Tensor, // [intermediate_size, hidden_size]
    pub up_proj: Tensor,   // [intermediate_size, hidden_size]
    pub down_proj: Tensor, // [hidden_size, intermediate_size]
}

/// Weights for one transformer block.
#[derive(Debug)]
pub struct TransformerBlockWeights {
    pub input_layernorm: RMSNormWeights,
    pub self_attn: Qwen3AttentionWeights,
    pub post_attention_layernorm: RMSNormWeights,
    pub mlp: Qwen3MLPWeights,
}

// ─── Network Graph ─────────────────────────────────────────────────────

/// Complete Qwen3 model network graph.
///
/// This defines the model structure (embedding → N×TransformerBlock →
/// norm → lm_head) with weight shape descriptors. No actual weight data
/// is held — this is used to drive the forward pass through stub operators.
#[derive(Debug)]
pub struct Qwen3Model {
    pub config: Qwen3Config,

    /// Token embedding table: [vocab_size, hidden_size].
    pub embed_tokens: Tensor,

    /// Transformer layers.
    pub layers: Vec<TransformerBlockWeights>,

    /// Final RMSNorm before the LM head.
    pub norm: RMSNormWeights,

    /// Language model head: [hidden_size, vocab_size].
    pub lm_head: Tensor,
}

impl Qwen3Model {
    /// Build the Qwen3 model network from configuration.
    ///
    /// Creates weight descriptors (shape + dtype) for all parameters.
    /// No actual memory is allocated.
    pub fn new(config: Qwen3Config) -> Self {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let vocab = config.vocab_size;

        let embed_tokens = Tensor::embedding_table(vocab, h, "model.embed_tokens.weight");

        let layers: Vec<TransformerBlockWeights> = (0..config.num_hidden_layers)
            .map(|i| {
                let prefix = format!("model.layers.{i}");
                TransformerBlockWeights {
                    input_layernorm: RMSNormWeights {
                        weight: Tensor::new(
                            vec![h],
                            DType::Float16,
                            format!("{prefix}.input_layernorm.weight"),
                        ),
                    },
                    self_attn: Qwen3AttentionWeights {
                        q_proj: Tensor::weight(
                            n_heads * head_dim,
                            h,
                            format!("{prefix}.self_attn.q_proj.weight"),
                        ),
                        k_proj: Tensor::weight(
                            n_kv_heads * head_dim,
                            h,
                            format!("{prefix}.self_attn.k_proj.weight"),
                        ),
                        v_proj: Tensor::weight(
                            n_kv_heads * head_dim,
                            h,
                            format!("{prefix}.self_attn.v_proj.weight"),
                        ),
                        o_proj: Tensor::weight(
                            h,
                            n_heads * head_dim,
                            format!("{prefix}.self_attn.o_proj.weight"),
                        ),
                        q_norm: Tensor::new(
                            vec![head_dim],
                            DType::Float16,
                            format!("{prefix}.self_attn.q_norm.weight"),
                        ),
                        k_norm: Tensor::new(
                            vec![head_dim],
                            DType::Float16,
                            format!("{prefix}.self_attn.k_norm.weight"),
                        ),
                    },
                    post_attention_layernorm: RMSNormWeights {
                        weight: Tensor::new(
                            vec![h],
                            DType::Float16,
                            format!("{prefix}.post_attention_layernorm.weight"),
                        ),
                    },
                    mlp: Qwen3MLPWeights {
                        gate_proj: Tensor::weight(
                            inter,
                            h,
                            format!("{prefix}.mlp.gate_proj.weight"),
                        ),
                        up_proj: Tensor::weight(inter, h, format!("{prefix}.mlp.up_proj.weight")),
                        down_proj: Tensor::weight(
                            h,
                            inter,
                            format!("{prefix}.mlp.down_proj.weight"),
                        ),
                    },
                }
            })
            .collect();

        let norm = RMSNormWeights {
            weight: Tensor::new(vec![h], DType::Float16, "model.norm.weight"),
        };

        let lm_head = Tensor::weight(vocab, h, "lm_head.weight");

        Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        }
    }

    /// Total number of parameters (approximate, for info display).
    pub fn param_count(&self) -> usize {
        let mut total = 0usize;
        total += self.embed_tokens.numel();
        for layer in &self.layers {
            total += layer.input_layernorm.weight.numel();
            total += layer.self_attn.q_proj.numel();
            total += layer.self_attn.k_proj.numel();
            total += layer.self_attn.v_proj.numel();
            total += layer.self_attn.o_proj.numel();
            total += layer.post_attention_layernorm.weight.numel();
            total += layer.mlp.gate_proj.numel();
            total += layer.mlp.up_proj.numel();
            total += layer.mlp.down_proj.numel();
        }
        total += self.norm.weight.numel();
        total += self.lm_head.numel();
        total
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Return mutable references to all weight tensors in the model.
    ///
    /// Used by the weight loader to set host_data and data_ptr on each tensor.
    /// Order: embed_tokens, then per-layer (norm, qkvo, norm, gate/up/down), then final norm, lm_head.
    pub fn weight_tensors_mut(&mut self) -> Vec<&mut Tensor> {
        let mut tensors: Vec<&mut Tensor> = Vec::new();
        tensors.push(&mut self.embed_tokens);
        for layer in &mut self.layers {
            tensors.push(&mut layer.input_layernorm.weight);
            tensors.push(&mut layer.self_attn.q_proj);
            tensors.push(&mut layer.self_attn.k_proj);
            tensors.push(&mut layer.self_attn.v_proj);
            tensors.push(&mut layer.self_attn.o_proj);
            tensors.push(&mut layer.self_attn.q_norm);
            tensors.push(&mut layer.self_attn.k_norm);
            tensors.push(&mut layer.post_attention_layernorm.weight);
            tensors.push(&mut layer.mlp.gate_proj);
            tensors.push(&mut layer.mlp.up_proj);
            tensors.push(&mut layer.mlp.down_proj);
        }
        tensors.push(&mut self.norm.weight);
        tensors.push(&mut self.lm_head);
        tensors
    }

    /// Count how many weight tensors have data loaded.
    pub fn loaded_count(&self) -> usize {
        let mut count = 0;
        if self.embed_tokens.is_loaded() {
            count += 1;
        }
        for layer in &self.layers {
            if layer.input_layernorm.weight.is_loaded() {
                count += 1;
            }
            if layer.self_attn.q_proj.is_loaded() {
                count += 1;
            }
            if layer.self_attn.k_proj.is_loaded() {
                count += 1;
            }
            if layer.self_attn.v_proj.is_loaded() {
                count += 1;
            }
            if layer.self_attn.o_proj.is_loaded() {
                count += 1;
            }
            if layer.self_attn.q_norm.is_loaded() {
                count += 1;
            }
            if layer.self_attn.k_norm.is_loaded() {
                count += 1;
            }
            if layer.post_attention_layernorm.weight.is_loaded() {
                count += 1;
            }
            if layer.mlp.gate_proj.is_loaded() {
                count += 1;
            }
            if layer.mlp.up_proj.is_loaded() {
                count += 1;
            }
            if layer.mlp.down_proj.is_loaded() {
                count += 1;
            }
        }
        if self.norm.weight.is_loaded() {
            count += 1;
        }
        if self.lm_head.is_loaded() {
            count += 1;
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_model_construction() {
        let config = Qwen3Config::qwen3_8b();
        let model = Qwen3Model::new(config);

        assert_eq!(model.num_layers(), 36);
        assert_eq!(model.embed_tokens.shape, vec![151936, 4096]);
        assert_eq!(model.lm_head.shape, vec![151936, 4096]);

        // Check first layer shapes — PyTorch [out_features, in_features]
        let layer0 = &model.layers[0];
        assert_eq!(layer0.self_attn.q_proj.shape, vec![4096, 4096]); // [32*128, 4096]
        assert_eq!(layer0.self_attn.k_proj.shape, vec![1024, 4096]); // [8*128, 4096]
        assert_eq!(layer0.self_attn.v_proj.shape, vec![1024, 4096]); // [8*128, 4096]
        assert_eq!(layer0.self_attn.o_proj.shape, vec![4096, 4096]);
        assert_eq!(layer0.mlp.gate_proj.shape, vec![12288, 4096]);
        assert_eq!(layer0.mlp.up_proj.shape, vec![12288, 4096]);
        assert_eq!(layer0.mlp.down_proj.shape, vec![4096, 12288]);
    }

    #[test]
    fn test_qwen3_param_count() {
        let config = Qwen3Config::qwen3_8b();
        let model = Qwen3Model::new(config);
        let params = model.param_count();
        // Qwen3-8B should be roughly 8B (8 billion) parameters
        // Exact: embed(151936*4096) + 36 layers * per_layer + norm + lm_head
        assert!(params > 7_000_000_000, "Expected >7B params, got {params}");
        assert!(params < 9_000_000_000, "Expected <9B params, got {params}");
    }

    #[test]
    fn test_qwen3_0_6b_model() {
        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config);
        assert_eq!(model.num_layers(), 28);
        assert_eq!(model.layers[0].self_attn.q_proj.shape, vec![2048, 1024]); // [16*128, 1024]
    }
}
