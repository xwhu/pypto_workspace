//! Paged KV cache manager — bridges logical blocks and NPU memory.
//!
//! Combines `kv_cache::SimpleKvManager` (CPU logical block management)
//! with `KvCachePool` (NPU physical device memory) so that:
//!
//! 1. `SimpleKvManager` tracks which blocks belong to which sequence
//! 2. `KvCachePool` owns the actual NPU memory for K/V cache data
//! 3. `build_block_table()` returns integer arrays for attention kernels
//! 4. `block_k_ptr(block_id, layer)` / `block_v_ptr(block_id, layer)` return NPU pointers

use crate::model::config::Qwen3Config;
use crate::model::tensor::DType;

#[cfg(feature = "ascend")]
use crate::model::device_tensor::KvCachePool;
use kv_cache::kv_manager::SimpleKvManager;
use kv_cache::types::{KvCacheConfig, SeqId};

/// Initial number of blocks to allocate when creating the cache pool.
const INITIAL_BLOCKS: usize = 64;

/// Paged KV cache manager for the inference engine.
///
/// On non-ascend builds, only the logical `SimpleKvManager` is active
/// (useful for tests). On ascend builds, `KvCachePool` manages real NPU memory.
pub struct PagedKvCacheManager {
    /// CPU-side logical block management.
    pub logical: SimpleKvManager,
    /// KV cache config (block_size, num_layers, etc.).
    pub config: KvCacheConfig,
    /// NPU-side physical memory pool.
    #[cfg(feature = "ascend")]
    pub pool: KvCachePool,
}

impl PagedKvCacheManager {
    /// Create a new paged KV cache manager from model config.
    ///
    /// On ascend builds, allocates initial NPU memory for `INITIAL_BLOCKS` blocks.
    pub fn new(model_config: &Qwen3Config) -> Self {
        let kv_config = KvCacheConfig {
            block_size: 16,  // tokens per block
            num_layers: model_config.num_hidden_layers,
            num_kv_heads: model_config.num_key_value_heads,
            head_dim: model_config.head_dim,
            dtype_size: 2,   // fp16
        };

        let logical = SimpleKvManager::new(kv_config, INITIAL_BLOCKS as u32);

        #[cfg(feature = "ascend")]
        let pool = {
            let mut p = KvCachePool::new(
                kv_config.num_layers,
                kv_config.num_kv_heads,
                kv_config.head_dim,
                kv_config.block_size,
                DType::Float16,
            );
            // Pre-allocate initial blocks
            p.grow(INITIAL_BLOCKS)
                .expect("Failed to allocate initial KV cache blocks on NPU");
            p
        };

        Self {
            logical,
            config: kv_config,
            #[cfg(feature = "ascend")]
            pool,
        }
    }

    /// Allocate blocks for a new sequence (prefill).
    ///
    /// Returns `true` on success.
    pub fn allocate_for_seq(&mut self, seq_id: SeqId, num_prompt_tokens: usize) -> bool {
        // Check if we need to grow the pool
        let blocks_needed = self.config.blocks_for_tokens(num_prompt_tokens) as u32;
        if !self.logical.can_allocate(blocks_needed) {
            // Try to grow the pool
            let grow_amount = (blocks_needed as usize).max(INITIAL_BLOCKS);
            #[cfg(feature = "ascend")]
            {
                if let Err(e) = self.pool.grow(grow_amount) {
                    eprintln!("[PagedKvCacheManager] Failed to grow pool: {:?}", e);
                    return false;
                }
            }
            // Rebuild logical manager with more blocks
            // Note: SimpleKvManager's BlockPool cannot grow dynamically,
            // so we need a workaround. For now, we over-allocate initially.
            // TODO: Make BlockPool growable or use RadixKvManager instead.
            if !self.logical.can_allocate(blocks_needed) {
                return false;
            }
        }
        self.logical.allocate_for_seq(seq_id, num_prompt_tokens)
    }

    /// Append one decode token to a sequence.
    pub fn append_token(&mut self, seq_id: SeqId) -> bool {
        self.logical.append_token(seq_id).is_some()
    }

    /// Build block table for batch of sequences (for attention kernel).
    pub fn build_block_table(&self, seq_ids: &[SeqId]) -> Vec<Vec<u32>> {
        self.logical.build_block_table(seq_ids)
    }

    /// Release all blocks for a finished sequence.
    pub fn release_seq(&mut self, seq_id: SeqId) -> u32 {
        self.logical.release_seq(seq_id)
    }

    /// Number of tokens cached for a sequence.
    pub fn seq_token_count(&self, seq_id: SeqId) -> usize {
        self.logical.seq_token_count(seq_id)
    }

    /// Block size (tokens per block).
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Get NPU pointer for block K cache at given layer.
    #[cfg(feature = "ascend")]
    pub fn block_k_ptr(&self, block_id: u32, layer: usize) -> *mut std::os::raw::c_void {
        self.pool.block_k_ptr(block_id as usize, layer)
    }

    /// Get NPU pointer for block V cache at given layer.
    #[cfg(feature = "ascend")]
    pub fn block_v_ptr(&self, block_id: u32, layer: usize) -> *mut std::os::raw::c_void {
        self.pool.block_v_ptr(block_id as usize, layer)
    }
}

impl std::fmt::Debug for PagedKvCacheManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PagedKvCacheManager(block_size={}, free_blocks={}/{})",
            self.config.block_size,
            self.logical.free_blocks(),
            self.logical.total_blocks(),
        )
    }
}

// Keep the old types as aliases for compatibility during transition
pub use kv_cache::types::SeqId;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_manager_creation() {
        let config = Qwen3Config::qwen3_0_6b();
        let mgr = PagedKvCacheManager::new(&config);
        assert_eq!(mgr.block_size(), 16);
        assert_eq!(mgr.logical.total_blocks(), INITIAL_BLOCKS as u32);
    }

    #[test]
    fn test_paged_manager_allocate_and_decode() {
        let config = Qwen3Config::qwen3_0_6b();
        let mut mgr = PagedKvCacheManager::new(&config);

        let seq = SeqId(42);
        assert!(mgr.allocate_for_seq(seq, 50));
        assert_eq!(mgr.seq_token_count(seq), 50);

        // Decode step
        assert!(mgr.append_token(seq));
        assert_eq!(mgr.seq_token_count(seq), 51);

        // Build block table
        let table = mgr.build_block_table(&[seq]);
        assert_eq!(table.len(), 1);
        assert!(!table[0].is_empty());

        // Release
        let freed = mgr.release_seq(seq);
        assert!(freed > 0);
    }
}
