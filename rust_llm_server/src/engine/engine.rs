use std::sync::Arc;

use crate::model::config::Qwen3Config;
use crate::model::network::Qwen3Model;
use crate::ops::Ops;
use super::forward::ForwardPass;
use super::kv_cache::{KVCacheManager, SequenceKVCache};

/// Generation configuration for a single request.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Stop generation if this token ID is produced.
    pub eos_token_id: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            eos_token_id: 151643, // Qwen3 EOS token
        }
    }
}

/// Result of text generation.
#[derive(Debug)]
pub struct GenerationResult {
    /// Generated token IDs (not including the prompt).
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens processed.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub completion_tokens: usize,
}

/// The inference engine — glues model, operators, and KV cache together.
///
/// Accepts a prompt (as token IDs), runs the forward pass in an
/// autoregressive loop, and returns generated token IDs.
pub struct Engine {
    model: Arc<Qwen3Model>,
    ops: Arc<dyn Ops>,
    kv_cache_manager: KVCacheManager,
}

impl Engine {
    pub fn new(model: Qwen3Model, ops: Arc<dyn Ops>) -> Self {
        let config = model.config.clone();
        let kv_cache_manager = KVCacheManager::new(
            config.clone(),
            config.max_position_embeddings,
        );
        Self {
            model: Arc::new(model),
            ops,
            kv_cache_manager,
        }
    }

    /// Generate tokens from a prompt.
    ///
    /// Runs the autoregressive generation loop:
    /// 1. Prefill: process all prompt tokens at once
    /// 2. Decode: generate one token at a time until EOS or max_new_tokens
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_config: &GenerationConfig,
    ) -> GenerationResult {
        let fwd = ForwardPass::new(&self.model, self.ops.as_ref());
        let mut kv_cache = self.kv_cache_manager.allocate();
        let mut generated_tokens = Vec::new();

        // 1. Prefill phase: process all prompt tokens
        let positions: Vec<u32> = (0..prompt_ids.len() as u32).collect();
        let logits = fwd.forward(prompt_ids, &positions, &mut kv_cache);
        let next_token = self.ops.sample_argmax(&logits);

        if next_token == gen_config.eos_token_id {
            return GenerationResult {
                token_ids: generated_tokens,
                prompt_tokens: prompt_ids.len(),
                completion_tokens: 0,
            };
        }
        generated_tokens.push(next_token);

        // 2. Decode phase: generate tokens one at a time
        for step in 0..gen_config.max_new_tokens.saturating_sub(1) {
            let pos = (prompt_ids.len() + step + 1) as u32;
            let logits = fwd.forward(
                &[*generated_tokens.last().unwrap()],
                &[pos],
                &mut kv_cache,
            );

            let next_token = self.ops.sample_argmax(&logits);
            if next_token == gen_config.eos_token_id {
                break;
            }
            generated_tokens.push(next_token);

            if kv_cache.remaining() == 0 {
                tracing::warn!("KV cache full, stopping generation");
                break;
            }
        }

        GenerationResult {
            prompt_tokens: prompt_ids.len(),
            completion_tokens: generated_tokens.len(),
            token_ids: generated_tokens,
        }
    }

    /// Get model configuration.
    pub fn config(&self) -> &Qwen3Config {
        &self.model.config
    }

    /// Get model info string.
    pub fn model_info(&self) -> String {
        format!(
            "{} ({}B params, {} layers, hidden={}, heads={}/{})",
            self.model.config.model_type,
            self.model.param_count() as f64 / 1e9,
            self.model.num_layers(),
            self.model.config.hidden_size,
            self.model.config.num_attention_heads,
            self.model.config.num_key_value_heads,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::StubOps;

    #[test]
    fn test_engine_generate() {
        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config);
        let ops = Arc::new(StubOps);
        let engine = Engine::new(model, ops);

        let prompt = vec![1u32, 2, 3, 4, 5];
        let gen_config = GenerationConfig {
            max_new_tokens: 10,
            eos_token_id: 151643,
        };

        let result = engine.generate(&prompt, &gen_config);

        assert_eq!(result.prompt_tokens, 5);
        // StubOps.sample_argmax always returns 0, which is not EOS (151643),
        // so we should get max_new_tokens generated
        assert_eq!(result.completion_tokens, 10);
        assert_eq!(result.token_ids.len(), 10);
    }

    #[test]
    fn test_engine_model_info() {
        let config = Qwen3Config::qwen3_8b();
        let model = Qwen3Model::new(config);
        let ops = Arc::new(StubOps);
        let engine = Engine::new(model, ops);

        let info = engine.model_info();
        assert!(info.contains("qwen3"));
        assert!(info.contains("36 layers"));
    }
}
