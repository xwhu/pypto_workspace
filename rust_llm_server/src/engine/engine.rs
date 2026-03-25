use crate::model::config::Qwen3Config;
use crate::model::network::Qwen3Model;
use crate::model::parallel::ParallelConfig;
use crate::model::quantize::QuantConfig;
use crate::ops::OpsBundle;
use super::kv_cache::PagedKvCacheManager;
use super::plan::{compile_plan, CompiledPlan};
use kv_cache::types::SeqId;

/// Generation configuration for a single request.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Stop generation if this token ID is produced.
    pub eos_token_id: u32,
    /// Sampling temperature (0.0 = greedy, higher = more random).
    pub temperature: f64,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            eos_token_id: 151645, // Qwen3 <|im_end|> token
            temperature: 0.0,
            top_p: 1.0,
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
///
/// The model is stored here to keep its `device_buf`s alive — the plan's
/// weight tensors hold `data_ptr` pointers into this model's device memory.
pub struct Engine {
    config: Qwen3Config,
    ops: OpsBundle,
    compiled_plan: CompiledPlan,
    kv_cache_manager: std::sync::Mutex<PagedKvCacheManager>,
    model_info: String,
    param_count: usize,
    /// Kept alive for device_buf ownership (plan's weight_tensors hold data_ptr into here).
    #[allow(dead_code)]
    _model: Qwen3Model,
    /// Concrete Ascend compute ops for v2 typed execution path.
    #[cfg(feature = "ascend")]
    ascend_ops: Option<crate::ops::ascend::AscendComputeOps>,
    /// Weight tensors for v2 path (non-owning views backed by _model's device_bufs).
    #[cfg(feature = "ascend")]
    weight_tensors_v2: Vec<crate::model::device_tensor::WeightTensor>,
}

impl Engine {
    /// Create a new engine with compiled execution plan.
    ///
    /// # Arguments
    /// * `model` - Logical model definition (kept alive for device memory)
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
        let kv_cache_manager = std::sync::Mutex::new(PagedKvCacheManager::new(&config));

        Self {
            config,
            ops,
            compiled_plan,
            kv_cache_manager,
            model_info,
            param_count,
            _model: model,
            #[cfg(feature = "ascend")]
            ascend_ops: None,
            #[cfg(feature = "ascend")]
            weight_tensors_v2: Vec::new(),
        }
    }

    /// Create a new engine using the Ascend NPU v2 typed execution path.
    ///
    /// Creates non-owning `WeightTensor` views from the plan's weight tensors
    /// and stores a concrete `AscendComputeOps` for direct method dispatch.
    #[cfg(feature = "ascend")]
    pub fn new_ascend(
        model: Qwen3Model,
        ascend_ops: crate::ops::ascend::AscendComputeOps,
        ops: OpsBundle,
        parallel: ParallelConfig,
        quant: QuantConfig,
    ) -> Self {
        use crate::model::device_tensor::WeightTensor;

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

        tracing::info!("Compiling execution plan...");
        let plan = compile_plan(&model, &parallel, &quant);
        tracing::info!(
            "Plan compiled: {} steps, {} buffers, {} weights",
            plan.num_steps(), plan.num_buffers, plan.weight_names.len(),
        );
        plan.dump();

        // Convert weight tensors to non-owning WeightTensor views.
        // The model's device_bufs keep the memory alive for the Engine's lifetime.
        let weight_tensors_v2: Vec<WeightTensor> = plan.weight_tensors.iter()
            .map(|t| {
                let ptr = t.data_ptr.expect("weight must have data_ptr");
                let buf = unsafe {
                    ascend::DeviceBuffer::from_raw_non_owning(
                        ptr as *mut std::os::raw::c_void,
                        t.size_bytes(),
                    )
                };
                WeightTensor::from_buf(t.shape.clone(), t.dtype, &t.name, buf)
            })
            .collect();
        tracing::info!("Created {} WeightTensor v2 views", weight_tensors_v2.len());

        let compiled_plan = plan.compile(&ops);
        let kv_cache_manager = std::sync::Mutex::new(PagedKvCacheManager::new(&config));

        Self {
            config,
            ops,
            compiled_plan,
            kv_cache_manager,
            model_info,
            param_count,
            _model: model,
            ascend_ops: Some(ascend_ops),
            weight_tensors_v2,
        }
    }

    /// Generate tokens from a prompt.
    ///
    /// Uses typed DeviceTensor/WeightTensor with full RAII device memory management.
    /// 1. Prefill: process all prompt tokens at once (prefix-cached tokens are skipped)
    /// 2. Decode: generate one token at a time until EOS or max_new_tokens
    #[cfg(feature = "ascend")]
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_config: &GenerationConfig,
    ) -> GenerationResult {
        let ascend_ops = self.ascend_ops.as_ref()
            .expect("Engine::generate requires ascend_ops (use Engine::new_ascend)");
        let weights = &self.weight_tensors_v2;
        let mut pool = crate::model::device_tensor::TensorPool::new(
            self.compiled_plan.plan().num_buffers,
        );
        let mut generated_tokens = Vec::new();

        // Allocate KV cache blocks with prefix matching
        static NEXT_SEQ_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let seq_id = SeqId(NEXT_SEQ_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        let cached_tokens = {
            let mut kv_mgr = self.kv_cache_manager.lock().unwrap();
            match kv_mgr.match_and_allocate(seq_id, prompt_ids) {
                Some(cached) => {
                    if cached > 0 {
                        tracing::info!(
                            "Prefix cache hit: {}/{} prompt tokens cached",
                            cached, prompt_ids.len(),
                        );
                    }
                    cached
                }
                None => {
                    tracing::error!(
                        "Failed to allocate KV cache for {} prompt tokens",
                        prompt_ids.len(),
                    );
                    return GenerationResult {
                        token_ids: Vec::new(),
                        prompt_tokens: prompt_ids.len(),
                        completion_tokens: 0,
                    };
                }
            }
        };

        // 1. Prefill — process all prompt tokens with dense attention
        // TODO: When paged attention is wired, skip first `cached_tokens` and
        //       only compute the uncached suffix.
        let positions: Vec<u32> = (0..prompt_ids.len() as u32).collect();
        let next_token = self.compiled_plan.execute(
            ascend_ops, &mut pool, weights, prompt_ids, &positions,
        );

        // Register newly computed blocks in the radix tree for future sharing
        {
            let mut kv_mgr = self.kv_cache_manager.lock().unwrap();
            kv_mgr.insert_computed_blocks(seq_id, prompt_ids, cached_tokens);
        }

        if next_token == gen_config.eos_token_id {
            self.kv_cache_manager.lock().unwrap().release_seq(seq_id);
            return GenerationResult {
                token_ids: generated_tokens,
                prompt_tokens: prompt_ids.len(),
                completion_tokens: 0,
            };
        }
        generated_tokens.push(next_token);
        self.kv_cache_manager.lock().unwrap().append_token(seq_id);

        // 2. Decode — generate one token at a time
        // TODO: Switch to paged attention using incre_flash_attention_v4
        // with block_table from kv_cache_manager.build_block_table()
        let mut all_tokens: Vec<u32> = prompt_ids.to_vec();
        all_tokens.push(next_token);

        for _step in 0..gen_config.max_new_tokens.saturating_sub(1) {
            let positions: Vec<u32> = (0..all_tokens.len() as u32).collect();
            let next_token = self.compiled_plan.execute(
                ascend_ops, &mut pool, weights, &all_tokens, &positions,
            );

            if next_token == gen_config.eos_token_id {
                break;
            }
            generated_tokens.push(next_token);
            {
                let mut kv_mgr = self.kv_cache_manager.lock().unwrap();
                kv_mgr.append_token(seq_id);
                if !kv_mgr.can_append(seq_id) {
                    tracing::warn!("KV cache full, stopping generation");
                    break;
                }
            }
            all_tokens.push(next_token);
        }

        // Release KV cache blocks — cached prefix blocks stay in tree
        self.kv_cache_manager.lock().unwrap().release_seq(seq_id);

        // pool dropped here → all DeviceTensors dropped → all DeviceBuffers freed ✓
        GenerationResult {
            prompt_tokens: prompt_ids.len(),
            completion_tokens: generated_tokens.len(),
            token_ids: generated_tokens,
        }
    }

    /// Stub generate for non-ascend backends (tests).
    #[cfg(not(feature = "ascend"))]
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_config: &GenerationConfig,
    ) -> GenerationResult {
        // Stub: return empty result for non-ascend builds
        GenerationResult {
            prompt_tokens: prompt_ids.len(),
            completion_tokens: 0,
            token_ids: Vec::new(),
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
