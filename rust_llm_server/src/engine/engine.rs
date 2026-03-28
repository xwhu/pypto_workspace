use super::kv_cache::KVCacheManager;
use super::plan::{compile_plan, CompiledPlan};
use crate::model::config::Qwen3Config;
use crate::model::network::Qwen3Model;
use crate::model::parallel::ParallelConfig;
use crate::model::quantize::QuantConfig;
use crate::ops::OpsBundle;

// Paged KV Cache integration
use kv_cache::block_manager::{BlockManager, SequenceId};
use kv_cache::npu_memory::{KVCacheConfig, KVCachePool};
use std::sync::Mutex;

/// Generation configuration for a single request.
#[derive(Debug, Clone)]
#[allow(dead_code)]
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
#[allow(dead_code)]
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
    /// Communication ops for TP/PP (HCCL-based AllReduce, Send, Recv).
    #[cfg(feature = "ascend")]
    comm_ops: Option<crate::ops::ascend_comm::AscendCommOps>,
    /// ACL context captured from the main thread after device init.
    /// Worker threads call `Device::set_current_context(acl_context)` to bind
    /// the device — aclrtSetDevice can only be called once per process per device.
    #[cfg(feature = "ascend")]
    acl_context: ascend::AclContext,
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
    /// Pre-allocated decode buffers (block_table etc.) to avoid per-step allocs.
    #[cfg(feature = "ascend")]
    decode_buffers: Mutex<Option<crate::ops::ascend::DecodeBuffers>>,
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
            comm_ops: None,
            #[cfg(feature = "ascend")]
            acl_context: ascend::AclContext(std::ptr::null_mut()),
            #[cfg(feature = "ascend")]
            weight_tensors_v2: Vec::new(),
            block_manager,
            kv_pool,
            #[cfg(feature = "ascend")]
            kv_key_caches: Vec::new(),
            #[cfg(feature = "ascend")]
            kv_value_caches: Vec::new(),
            #[cfg(feature = "ascend")]
            decode_buffers: Mutex::new(None),
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
            plan.num_steps(),
            plan.num_buffers,
            plan.weight_names.len(),
        );
        plan.dump();

        // Convert weight tensors to non-owning WeightTensor views.
        // The model's device_bufs keep the memory alive for the Engine's lifetime.
        let weight_tensors_v2: Vec<WeightTensor> = plan
            .weight_tensors
            .iter()
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

        // Paged KV Cache — TP-aware: each rank has fewer KV heads
        let block_size = 16;
        let num_blocks = 256;
        let kv_heads_per_rank = config.num_key_value_heads / parallel.tp_size;
        let kv_config = KVCacheConfig {
            num_layers: model.num_layers(), // PP-aware: only this stage's layers
            num_kv_heads: kv_heads_per_rank,
            head_dim: config.head_dim,
            block_size,
            num_blocks,
        };
        let block_manager = Mutex::new(BlockManager::new(num_blocks, block_size));
        let kv_pool = KVCachePool::new(kv_config);

        // Allocate per-layer KV cache device buffers on NPU
        // TP: num_kv_heads / tp_size per rank
        let per_layer_bytes =
            num_blocks * block_size * kv_heads_per_rank * config.head_dim * 2; // FP16
        let num_layers = model.num_layers(); // PP-aware: only this stage's layers
        let mut kv_key_caches = Vec::with_capacity(num_layers);
        let mut kv_value_caches = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
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
            "Allocated NPU KV cache: {} layers × 2 × {:.2} MB = {:.2} MB total (kv_heads_per_rank={})",
            num_layers,
            per_layer_bytes as f64 / 1e6,
            (num_layers * 2 * per_layer_bytes) as f64 / 1e6,
            kv_heads_per_rank,
        );

        // Pre-allocate decode buffers
        let decode_buffers = ascend_ops.init_decode_buffers(num_blocks);
        tracing::info!(
            "Pre-allocated decode buffers: block_table capacity={}",
            num_blocks
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
            comm_ops: None, // Set via set_comm_ops() after distributed init
            // Capture the ACL context from this (main) thread so worker threads
            // can call set_current_context() — the correct CANN multi-thread pattern.
            acl_context: ascend::Device::get_current_context()
                .unwrap_or(ascend::AclContext(std::ptr::null_mut())),
            weight_tensors_v2,
            block_manager,
            kv_pool,
            kv_key_caches,
            kv_value_caches,
            decode_buffers: Mutex::new(Some(decode_buffers)),
        }
    }

    /// Ensure the calling thread has the Ascend device context bound.
    ///
    /// Uses a thread-local flag so the (cheap) `aclrtSetCurrentContext` call is
    /// made exactly once per OS or Tokio worker thread. Safe to call repeatedly.
    #[cfg(feature = "ascend")]
    fn ensure_device_context(&self) {
        use std::cell::Cell;
        thread_local! {
            static CONTEXT_SET: Cell<bool> = const { Cell::new(false) };
        }
        CONTEXT_SET.with(|flag| {
            if !flag.get() {
                ascend::Device::set_current_context(self.acl_context)
                    .expect("ensure_device_context: aclrtSetCurrentContext failed");
                flag.set(true);
            }
        });
    }

    /// Generate tokens from a prompt using Paged KV Cache.
    ///
    /// Uses typed DeviceTensor/WeightTensor with full RAII device memory management.
    /// 1. Prefill: process all prompt tokens at once, allocate KV blocks
    /// 2. Decode: generate one token at a time, append slots to paged cache
    #[cfg(feature = "ascend")]
    pub fn generate(&self, prompt_ids: &[u32], gen_config: &GenerationConfig) -> GenerationResult {
        use super::plan::PagedKVContext;

        // Bind device context to this thread (no-op if already set).
        self.ensure_device_context();

        let ascend_ops = self
            .ascend_ops
            .as_ref()
            .expect("Engine::generate requires ascend_ops (use Engine::new_ascend)");
        let weights = &self.weight_tensors_v2;
        let mut pool =
            crate::model::device_tensor::TensorPool::new(self.compiled_plan.plan().num_buffers);
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

            tracing::info!(
                "Prefill: {} tokens, {} blocks allocated",
                prompt_ids.len(),
                tracker.physical_blocks.len()
            );

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

        // --- TP: broadcast input_ids, positions, and full KV Context to worker ranks ---
        #[cfg(all(feature = "ascend", feature = "hccl"))]
        self.broadcast_paged_inputs(prompt_ids, &positions, &prefill_ctx);

        let next_token = self.compiled_plan.execute_paged(
            ascend_ops,
            self.comm_ops.as_ref(),
            &mut pool,
            weights,
            prompt_ids,
            &positions,
            &prefill_ctx,
            &self.kv_key_caches,
            &self.kv_value_caches,
            self.decode_buffers.lock().unwrap().as_mut(),
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

        // 2. Decode — single-token PagedAttention per step
        //
        // Each step we pass ONLY the newly generated token. The Attention step
        // uses paged_decode_attention (IncreFlashAttentionV4) to read K/V from
        // the physical NPU cache populated by reshape_and_cache during prefill
        // and prior decode steps.
        let mut context_len = prompt_ids.len() + 1; // prompt + first generated token

        for _step in 0..gen_config.max_new_tokens.saturating_sub(1) {
            let latest_token = *generated_tokens.last().unwrap();

            // Append new token to BlockManager and track its slot
            {
                let mut bm = self.block_manager.lock().unwrap();
                let _ = bm.append_slot(seq_id, Some(latest_token));
            }

            // Single-token decode: position = context_len - 1 (0-indexed)
            let positions: Vec<u32> = vec![(context_len - 1) as u32];

            // Build decode context with ONLY the new token's slot
            let decode_ctx = {
                let bm = self.block_manager.lock().unwrap();
                let tracker = bm.active_seqs.get(&seq_id).unwrap();
                let block_table = KVCachePool::build_block_table(
                    &[tracker.physical_blocks.clone()],
                    tracker.physical_blocks.len(),
                );

                // Slot mapping: only the latest token
                let token_pos = context_len - 1;
                let block_idx = token_pos / block_size;
                let offset = token_pos % block_size;
                let bid = tracker.physical_blocks[block_idx];
                let slot = (bid as usize * block_size + offset) as i32;

                PagedKVContext {
                    is_decode: true, // ← Activate paged decode attention!
                    context_len,
                    block_table,
                    max_blocks_per_seq: tracker.physical_blocks.len(),
                    slot_mapping: vec![slot],
                    block_size,
                    layer_idx: std::cell::Cell::new(0),
                }
            };

            // --- TP: broadcast decode token/position and full KV Context to all worker ranks ---
            #[cfg(all(feature = "ascend", feature = "hccl"))]
            {
                tracing::debug!("[Rank 0] Decode: broadcasting inputs");
                self.broadcast_paged_inputs(&[latest_token], &positions, &decode_ctx);
                tracing::debug!("[Rank 0] Decode: broadcast done, executing forward pass");
            }
            #[cfg(not(all(feature = "ascend", feature = "hccl")))]
            let _ = &positions; // To avoid unused warning if no TP

            let next_token_inner = self.compiled_plan.execute_paged(
                ascend_ops,
                self.comm_ops.as_ref(),
                &mut pool,
                weights,
                &[latest_token],
                &positions,
                &decode_ctx,
                &self.kv_key_caches,
                &self.kv_value_caches,
                self.decode_buffers.lock().unwrap().as_mut(),
            );
            tracing::debug!("[Rank 0] Decode: execute_paged done (sampled {})", next_token_inner);

            if next_token_inner == gen_config.eos_token_id {
                break;
            }
            generated_tokens.push(next_token_inner);
            context_len += 1;

            // Check OOM
            if !self.block_manager.lock().unwrap().can_allocate(1) {
                tracing::warn!("KV cache full, stopping generation");
                break;
            }
        }

        // Free sequence blocks
        self.block_manager.lock().unwrap().free_sequence(seq_id);

        // --- TP: send shutdown sentinel to workers (sequence complete) ---
        // Workers loop back to waiting for the next broadcast; if the engine
        // is about to shut down the primary sends [u32::MAX] via main.rs shutdown path.

        GenerationResult {
            prompt_tokens: prompt_ids.len(),
            completion_tokens: generated_tokens.len(),
            token_ids: generated_tokens,
        }
    }

    /// Helper to efficiently broadcast the PagedKVContext and sequence data
    /// to all worker ranks in a single packed contiguous block.
    #[cfg(all(feature = "ascend", feature = "hccl"))]
    fn broadcast_paged_inputs(
        &self,
        prompt_ids: &[u32],
        positions: &[u32],
        ctx: &super::plan::PagedKVContext,
    ) {
        if let Some(ref comm) = self.comm_ops {
            let meta = [
                prompt_ids.len() as u32,
                ctx.context_len as u32,
                ctx.max_blocks_per_seq as u32,
                ctx.block_table.len() as u32,
                ctx.slot_mapping.len() as u32,
            ];
            comm.broadcast_u32_slice(&meta, "paged_meta", 0, true);

            let payload_len = prompt_ids.len()
                + positions.len()
                + ctx.block_table.len()
                + ctx.slot_mapping.len();

            let mut payload = Vec::with_capacity(payload_len);
            payload.extend_from_slice(prompt_ids);
            payload.extend_from_slice(positions);
            for &b in &ctx.block_table {
                payload.push(b as u32);
            }
            for &s in &ctx.slot_mapping {
                payload.push(s as u32);
            }
            comm.broadcast_u32_slice(&payload, "paged_payload", 0, true);
        }
    }


    /// Worker rank main loop — blocks on HCCL broadcast from primary (tp_rank=0).
    ///
    /// Called by non-primary TP ranks (tp_rank != 0). The primary rank sends input
    /// tensors via `HcclBroadcast` before every forward pass. The worker receives
    /// the broadcast, runs the identical forward pass (including AllReduce for TP),
    /// and discards the output — only the primary sends results back via HTTP.
    ///
    /// The loop terminates when the primary broadcasts the shutdown sentinel
    /// `[u32::MAX]` as input_ids (a single-token batch with token ID = u32::MAX).
    ///
    /// # Design (matches vLLM `worker_busy_loop` pattern)
    /// - CPU carries only metadata (loop control); all tensor ops happen on NPU.
    /// - HCCL Broadcast is the blocking synchronization point — no CPU mutex.
    /// - AllReduce inside execute_paged() syncs TP ranks during the forward pass.
    #[cfg(all(feature = "ascend", feature = "hccl"))]
    pub fn run_worker_loop(&self) {
        use super::plan::PagedKVContext;

        // Bind device context to this thread (no-op if already set).
        self.ensure_device_context();

        let ascend_ops = self
            .ascend_ops
            .as_ref()
            .expect("run_worker_loop requires ascend_ops");
        let comm = self
            .comm_ops
            .as_ref()
            .expect("run_worker_loop requires comm_ops (HCCL)");
        let weights = &self.weight_tensors_v2;
        let block_size = self.kv_pool.config.block_size;

        tracing::info!("Worker rank: entering HCCL broadcast loop");

        loop {
            // Wait for broadcast metadata (5 ints)
            let meta_tensor = comm.broadcast_u32_slice(&[0; 5], "paged_meta", 0, false);
            let mut meta = [0u32; 5];
            {
                let bytes = unsafe { std::slice::from_raw_parts_mut(meta.as_mut_ptr() as *mut u8, 20) };
                meta_tensor.buf.copy_to_host(bytes).expect("Worker: copy meta from device failed");
            }
            drop(meta_tensor);

            // Sentinel check for shutdown
            if meta[0] == u32::MAX {
                tracing::info!("Worker rank: received shutdown sentinel, exiting loop");
                break;
            }

            let input_len = meta[0] as usize;
            let context_len = meta[1] as usize;
            let max_blocks_per_seq = meta[2] as usize;
            let block_table_len = meta[3] as usize;
            let slot_mapping_len = meta[4] as usize;

            // Wait for packed payload
            let payload_len = input_len + input_len + block_table_len + slot_mapping_len;
            let dummy_payload = vec![0u32; payload_len];
            let payload_tensor = comm.broadcast_u32_slice(&dummy_payload, "paged_payload", 0, false);

            let mut payload = vec![0u32; payload_len];
            {
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(payload.as_mut_ptr() as *mut u8, payload_len * 4)
                };
                payload_tensor.buf.copy_to_host(bytes).expect("Worker: copy payload from device failed");
            }
            drop(payload_tensor);

            // Unpack payload into typed buffers
            let mut offset = 0;
            let input_ids_host = payload[offset..offset + input_len].to_vec();
            offset += input_len;

            let positions_host = payload[offset..offset + input_len].to_vec();
            offset += input_len;

            let block_table: Vec<i32> = payload[offset..offset + block_table_len]
                .iter()
                .map(|&x| x as i32)
                .collect();
            offset += block_table_len;

            let slot_mapping: Vec<i32> = payload[offset..offset + slot_mapping_len]
                .iter()
                .map(|&x| x as i32)
                .collect();

            // Reconstruct perfectly mirrored KV Context
            let paged_ctx = PagedKVContext {
                is_decode: input_len == 1,
                context_len,
                block_table,
                max_blocks_per_seq,
                slot_mapping,
                block_size,
                layer_idx: std::cell::Cell::new(0),
            };

            let mut pool = crate::model::device_tensor::TensorPool::new(
                self.compiled_plan.plan().num_buffers,
            );

            // Execute — HCCL AllReduce syncs this rank with the primary automatically.
            tracing::debug!("Worker: before execute_paged (is_decode={})", paged_ctx.is_decode);
            let _discarded_token = self.compiled_plan.execute_paged(
                ascend_ops,
                self.comm_ops.as_ref(),
                &mut pool,
                weights,
                &input_ids_host,
                &positions_host,
                &paged_ctx,
                &self.kv_key_caches,
                &self.kv_value_caches,
                self.decode_buffers.lock().unwrap().as_mut(),
            );
            tracing::debug!("Worker: execute_paged done");

            tracing::debug!(
                "Worker rank: step complete ({} tokens, context_len={}, is_decode={})",
                input_len,
                context_len,
                paged_ctx.is_decode
            );
        }

        tracing::info!("Worker rank: HCCL loop exited");
    }

    /// Stub generate for non-ascend backends (tests).
    #[cfg(not(feature = "ascend"))]
    pub fn generate(&self, prompt_ids: &[u32], _gen_config: &GenerationConfig) -> GenerationResult {
        // Stub: return empty result for non-ascend builds
        GenerationResult {
            prompt_tokens: prompt_ids.len(),
            completion_tokens: 0,
            token_ids: Vec::new(),
        }
    }

    /// Set the communication ops for distributed execution.
    ///
    /// Called after HCCL initialization to enable AllReduce, Send, Recv
    /// in the execution plan. Must be called before `generate()` when
    /// using TP or PP.
    #[cfg(all(feature = "ascend", feature = "hccl"))]
    pub fn set_comm_ops(&mut self, comm_ops: crate::ops::ascend_comm::AscendCommOps) {
        self.comm_ops = Some(comm_ops);  // field is always Option<AscendCommOps> (stub when no hccl)
    }

    /// Get the raw AscendCL stream handle from the compute ops.
    ///
    /// Used to share the compute stream with HCCL comm ops — all collectives
    /// should run on the same stream as compute for automatic serialization.
    #[cfg(feature = "ascend")]
    pub fn compute_stream(&self) -> Option<ascendcl_sys::AclrtStream> {
        self.ascend_ops.as_ref().map(|ops| ops.stream().raw())
    }

    /// Get model configuration.
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// Get model info string.
    pub fn model_info(&self) -> &str {
        &self.model_info
    }
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
