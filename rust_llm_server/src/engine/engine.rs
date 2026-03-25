use crate::model::config::Qwen3Config;
use crate::model::network::Qwen3Model;
use crate::model::parallel::ParallelConfig;
use crate::model::quantize::QuantConfig;
use crate::ops::OpsBundle;
use super::kv_cache::KVCacheManager;
use super::plan::{compile_plan, CompiledPlan};

// Paged KV Cache integration
use kv_cache::block_manager::{BlockManager, SequenceId};
use kv_cache::npu_memory::{KVCacheConfig, KVCachePool};
use std::sync::Mutex;

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
    kv_cache_manager: KVCacheManager,
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
    /// Paged KV Cache block manager for prefix caching and memory reuse.
    block_manager: Mutex<BlockManager>,
    /// Physical NPU KV cache memory pool configuration.
    kv_pool: KVCachePool,
    /// Per-layer key cache device buffers: [num_blocks, block_size, num_kv_heads, head_dim] FP16
    #[cfg(feature = "ascend")]
    kv_key_caches: Vec<ascend::memory::DeviceBuffer>,
    /// Per-layer value cache device buffers: same shape
    #[cfg(feature = "ascend")]
    kv_value_caches: Vec<ascend::memory::DeviceBuffer>,
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
        let kv_cache_manager = KVCacheManager::new(config.clone(), config.max_position_embeddings);

        // Paged KV Cache: block_size=16, num_blocks=256 (tunable per model/GPU memory)
        let block_size = 16;
        let num_blocks = 256;
        let kv_config = KVCacheConfig {
            num_layers: config.num_hidden_layers,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            block_size,
            num_blocks,
        };
        let block_manager = Mutex::new(BlockManager::new(num_blocks, block_size));
        let kv_pool = KVCachePool::new(kv_config);

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
            block_manager,
            kv_pool,
            #[cfg(feature = "ascend")]
            kv_key_caches: Vec::new(),
            #[cfg(feature = "ascend")]
            kv_value_caches: Vec::new(),
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
        let kv_cache_manager = KVCacheManager::new(config.clone(), config.max_position_embeddings);

        // Paged KV Cache
        let block_size = 16;
        let num_blocks = 256;
        let kv_config = KVCacheConfig {
            num_layers: config.num_hidden_layers,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            block_size,
            num_blocks,
        };
        let block_manager = Mutex::new(BlockManager::new(num_blocks, block_size));
        let kv_pool = KVCachePool::new(kv_config);

        // Allocate per-layer KV cache device buffers on NPU
        let per_layer_bytes = num_blocks * block_size * config.num_key_value_heads * config.head_dim * 2; // FP16
        let mut kv_key_caches = Vec::with_capacity(config.num_hidden_layers);
        let mut kv_value_caches = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let mut k_buf = ascend::memory::DeviceBuffer::alloc(per_layer_bytes)
                .expect("KV cache: failed to allocate K cache");
            let mut v_buf = ascend::memory::DeviceBuffer::alloc(per_layer_bytes)
                .expect("KV cache: failed to allocate V cache");
            k_buf.memset_zero().expect("KV cache: failed to zero K");
            v_buf.memset_zero().expect("KV cache: failed to zero V");
            kv_key_caches.push(k_buf);
            kv_value_caches.push(v_buf);
        }
        tracing::info!(
            "Allocated NPU KV cache: {} layers × 2 × {:.2} MB = {:.2} MB total",
            config.num_hidden_layers,
            per_layer_bytes as f64 / 1e6,
            (config.num_hidden_layers * 2 * per_layer_bytes) as f64 / 1e6,
        );

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
            block_manager,
            kv_pool,
            kv_key_caches,
            kv_value_caches,
        }
    }

    /// Generate tokens from a prompt using Paged KV Cache.
    ///
    /// Uses typed DeviceTensor/WeightTensor with full RAII device memory management.
    /// 1. Prefill: process all prompt tokens at once, allocate KV blocks
    /// 2. Decode: generate one token at a time, append slots to paged cache
    #[cfg(feature = "ascend")]
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_config: &GenerationConfig,
    ) -> GenerationResult {
        use super::plan::PagedKVContext;

        let ascend_ops = self.ascend_ops.as_ref()
            .expect("Engine::generate requires ascend_ops (use Engine::new_ascend)");
        let weights = &self.weight_tensors_v2;
        let mut pool = crate::model::device_tensor::TensorPool::new(
            self.compiled_plan.plan().num_buffers,
        );
        let mut generated_tokens = Vec::new();
        let block_size = self.kv_pool.config.block_size;

        // Allocate a unique sequence ID for this request
        let seq_id = SequenceId(0); // single-sequence for now

        // 1. Prefill — allocate blocks and build context (hold lock briefly)
        let prefill_ctx = {
            let mut bm = self.block_manager.lock().unwrap();
            bm.allocate_prefix(seq_id, prompt_ids)
                .expect("Failed to allocate KV blocks for prompt");

            let tracker = bm.active_seqs.get(&seq_id).unwrap();
            let block_table_flat = KVCachePool::build_block_table(
                &[tracker.physical_blocks.clone()],
                tracker.physical_blocks.len(),
            );

            // Build slot mapping for all prompt tokens
            let mut slot_mapping = Vec::new();
            for (i, _token) in prompt_ids.iter().enumerate() {
                let block_idx = i / block_size;
                let offset = i % block_size;
                let bid = tracker.physical_blocks[block_idx];
                slot_mapping.push((bid as usize * block_size + offset) as i32);
            }

            tracing::info!("Prefill: {} tokens, {} blocks allocated", prompt_ids.len(), tracker.physical_blocks.len());

            PagedKVContext {
                is_decode: false,
                context_len: prompt_ids.len(),
                block_table: block_table_flat,
                max_blocks_per_seq: tracker.physical_blocks.len(),
                slot_mapping,
                block_size,
                layer_idx: std::cell::Cell::new(0),
            }
        }; // lock released here before expensive NPU work

        let positions: Vec<u32> = (0..prompt_ids.len() as u32).collect();
        let next_token = self.compiled_plan.execute_paged(
            ascend_ops, &mut pool, weights, prompt_ids, &positions, &prefill_ctx,
            &self.kv_key_caches, &self.kv_value_caches,
        );

        if next_token == gen_config.eos_token_id {
            self.block_manager.lock().unwrap().free_sequence(seq_id);
            return GenerationResult {
                token_ids: generated_tokens,
                prompt_tokens: prompt_ids.len(),
                completion_tokens: 0,
            };
        }
        generated_tokens.push(next_token);

        // 2. Decode — all tokens each step (until paged KV cache is physically wired)
        //
        // NOTE: We pass ALL tokens (prompt + generated so far) because the Attention
        // step still uses flash_attention_score_with_mask which needs full Q/K/V.
        // Once reshape_and_cache + paged_attention_decode are wired to actual NPU
        // device memory, we switch to single-token decode.
        let mut all_tokens: Vec<u32> = prompt_ids.to_vec();
        all_tokens.push(next_token);

        for _step in 0..gen_config.max_new_tokens.saturating_sub(1) {
            // Track blocks logically (for future paged attention)
            {
                let mut bm = self.block_manager.lock().unwrap();
                let _ = bm.append_slot(seq_id, Some(*all_tokens.last().unwrap()));
            }

            let context_len = all_tokens.len();
            let positions: Vec<u32> = (0..context_len as u32).collect();

            // Build slot mapping for all tokens
            let decode_ctx = {
                let bm = self.block_manager.lock().unwrap();
                let tracker = bm.active_seqs.get(&seq_id).unwrap();
                let block_table = KVCachePool::build_block_table(
                    &[tracker.physical_blocks.clone()],
                    tracker.physical_blocks.len(),
                );

                let mut slot_mapping = Vec::new();
                for i in 0..context_len {
                    let block_idx = i / block_size;
                    let offset = i % block_size;
                    let bid = tracker.physical_blocks[block_idx];
                    slot_mapping.push((bid as usize * block_size + offset) as i32);
                }

                PagedKVContext {
                    is_decode: false, // still using full FlashAttention
                    context_len,
                    block_table,
                    max_blocks_per_seq: tracker.physical_blocks.len(),
                    slot_mapping,
                    block_size,
                    layer_idx: std::cell::Cell::new(0),
                }
            };

            let next_token_inner = self.compiled_plan.execute_paged(
                ascend_ops, &mut pool, weights, &all_tokens, &positions, &decode_ctx,
                &self.kv_key_caches, &self.kv_value_caches,
            );

            if next_token_inner == gen_config.eos_token_id {
                break;
            }
            generated_tokens.push(next_token_inner);
            all_tokens.push(next_token_inner);

            // Check OOM
            if !self.block_manager.lock().unwrap().can_allocate(1) {
                tracing::warn!("KV cache full, stopping generation");
                break;
            }
        }

        // Free sequence blocks
        self.block_manager.lock().unwrap().free_sequence(seq_id);

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
