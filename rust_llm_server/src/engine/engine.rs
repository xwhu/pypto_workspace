use crate::model::config::Qwen3Config;
use crate::model::network::Qwen3Model;
use crate::model::parallel::ParallelConfig;
use crate::model::quantize::QuantConfig;
use crate::ops::OpsBundle;
use super::kv_cache::KVCacheManager;
use super::plan::{compile_plan, CompiledPlan, TensorPool};

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
/// Uses a compiled execution plan (方案B) for minimal dispatch overhead.
/// The plan is compiled once at construction time and reused for every
/// inference step.
pub struct Engine {
    config: Qwen3Config,
    ops: OpsBundle,
    compiled_plan: CompiledPlan,
    kv_cache_manager: KVCacheManager,
    model_info: String,
    param_count: usize,
}

impl Engine {
    /// Create a new engine with compiled execution plan.
    ///
    /// # Arguments
    /// * `model` - Logical model definition (consumed)
    /// * `ops` - Operator bundle (compute + comm + quant)
    /// * `parallel` - Parallel execution config
    /// * `quant` - Quantization config
    pub fn new(
        model: Qwen3Model,
        ops: OpsBundle,
        parallel: ParallelConfig,
        quant: QuantConfig,
    ) -> Self {
        let config = model.config.clone();
        let model_info = format!(
            "{} ({}B params, {} layers, hidden={}, heads={}/{}, tp={}, pp={})",
            config.model_type,
            model.param_count() as f64 / 1e9,
            model.num_layers(),
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            parallel.tp_size,
            parallel.pp_size,
        );
        let param_count = model.param_count();

        // Compile the execution plan at init time
        tracing::info!("Compiling execution plan...");
        let plan = compile_plan(&model, &parallel, &quant);
        tracing::info!(
            "Plan compiled: {} steps, {} buffers, {} weights",
            plan.num_steps(),
            plan.num_buffers,
            plan.weight_names.len(),
        );
        plan.dump();

        let compiled_plan = plan.compile(&ops);
        let kv_cache_manager = KVCacheManager::new(config.clone(), config.max_position_embeddings);

        Self {
            config,
            ops,
            compiled_plan,
            kv_cache_manager,
            model_info,
            param_count,
        }
    }

    /// Generate tokens from a prompt.
    ///
    /// Runs the autoregressive generation loop using the compiled plan:
    /// 1. Prefill: process all prompt tokens at once
    /// 2. Decode: generate one token at a time until EOS or max_new_tokens
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_config: &GenerationConfig,
    ) -> GenerationResult {
        let mut kv_cache = self.kv_cache_manager.allocate();
        let mut pool = TensorPool::new(self.compiled_plan.plan().num_buffers);
        let mut generated_tokens = Vec::new();

        // 1. Prefill phase
        let positions: Vec<u32> = (0..prompt_ids.len() as u32).collect();
        let next_token = self.compiled_plan.execute(
            &self.ops, &mut pool, prompt_ids, &positions, &mut kv_cache,
        );
        kv_cache.append(prompt_ids.len());

        if next_token == gen_config.eos_token_id {
            return GenerationResult {
                token_ids: generated_tokens,
                prompt_tokens: prompt_ids.len(),
                completion_tokens: 0,
            };
        }
        generated_tokens.push(next_token);

        // 2. Decode phase
        for step in 0..gen_config.max_new_tokens.saturating_sub(1) {
            let pos = (prompt_ids.len() + step + 1) as u32;
            let next_token = self.compiled_plan.execute(
                &self.ops,
                &mut pool,
                &[*generated_tokens.last().unwrap()],
                &[pos],
                &mut kv_cache,
            );
            kv_cache.append(1);

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
    pub fn config(&self) -> &Qwen3Config { &self.config }

    /// Get model info string.
    pub fn model_info(&self) -> &str { &self.model_info }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_generate() {
        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config);
        let engine = Engine::new(
            model,
            OpsBundle::stub(),
            ParallelConfig::single_device(),
            QuantConfig::none(),
        );

        let prompt = vec![1u32, 2, 3, 4, 5];
        let gen_config = GenerationConfig {
            max_new_tokens: 10,
            eos_token_id: 151643,
        };

        let result = engine.generate(&prompt, &gen_config);
        assert_eq!(result.prompt_tokens, 5);
        assert_eq!(result.completion_tokens, 10);
        assert_eq!(result.token_ids.len(), 10);
    }

    #[test]
    fn test_engine_model_info() {
        let config = Qwen3Config::qwen3_8b();
        let model = Qwen3Model::new(config);
        let engine = Engine::new(
            model,
            OpsBundle::stub(),
            ParallelConfig::single_device(),
            QuantConfig::none(),
        );

        let info = engine.model_info();
        assert!(info.contains("qwen3"));
        assert!(info.contains("36 layers"));
    }

    #[test]
    fn test_engine_tp_config() {
        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config);
        let engine = Engine::new(
            model,
            OpsBundle::stub(),
            ParallelConfig::tensor_parallel(4, 0),
            QuantConfig::none(),
        );

        let info = engine.model_info();
        assert!(info.contains("tp=4"));
    }
}
