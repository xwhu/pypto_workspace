//! Paged KV cache manager — bridges logical blocks and NPU memory.
//!
//! Combines `kv_cache::RadixKvManager` (CPU logical block management with
//! prefix caching) with `KvCachePool` (NPU physical device memory) so that:
//!
//! 1. `RadixKvManager` tracks blocks per sequence with radix-tree prefix sharing
//! 2. `KvCachePool` owns the actual NPU memory for K/V cache data
//! 3. `build_block_table()` returns integer arrays for attention kernels
//! 4. `block_k_ptr(block_id, layer)` / `block_v_ptr(block_id, layer)` return NPU pointers

use crate::model::config::Qwen3Config;
use crate::model::tensor::DType;

#[cfg(feature = "ascend")]
use crate::model::device_tensor::KvCachePool;
use kv_cache::radix_tree::RadixKvManager;
use kv_cache::types::{KvCacheConfig, SeqId};

/// Initial number of blocks to allocate when creating the cache pool.
const INITIAL_BLOCKS: usize = 256;

/// Paged KV cache manager for the inference engine.
///
/// Uses `RadixKvManager` for prefix-aware block management:
/// - Sequences sharing the same token prefix reuse the same physical KV blocks
/// - `match_and_allocate()` returns how many tokens are already cached
/// - `insert_computed_blocks()` registers newly computed blocks for future sharing
/// - LRU eviction frees unreferenced cache blocks when memory is tight
pub struct PagedKvCacheManager {
    /// CPU-side logical block management with radix-tree prefix caching.
    pub logical: RadixKvManager,
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

        let logical = RadixKvManager::new(kv_config.clone(), INITIAL_BLOCKS as u32);

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

    /// Match cached prefix and allocate blocks for a new sequence (prefill).
    ///
    /// Returns `Some(cached_token_count)` on success — the caller should
    /// only compute the forward pass for `tokens[cached_token_count..]`.
    ///
    /// If blocks are insufficient, tries eviction first, then fails.
    pub fn match_and_allocate(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
    ) -> Option<usize> {
        // Try direct allocation first
        if let Some(cached) = self.logical.match_and_allocate(seq_id, tokens) {
            return Some(cached);
        }
        // Allocation failed — try evicting some blocks
        let blocks_needed = self.config.blocks_for_tokens(tokens.len());
        let evicted = self.logical.evict(blocks_needed);
        if evicted > 0 {
            // Retry after eviction
            self.logical.match_and_allocate(seq_id, tokens)
        } else {
            None
        }
    }

    /// After prefill computation, register newly computed blocks in the
    /// radix tree for future prefix sharing.
    ///
    /// - `seq_id`: the sequence that was just prefilled
    /// - `tokens`: the full token sequence
    /// - `cached_tokens`: count returned by `match_and_allocate()`
    pub fn insert_computed_blocks(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        cached_tokens: usize,
    ) {
        self.logical.insert_computed_blocks(seq_id, tokens, cached_tokens);
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
    /// Cached prefix blocks remain in the tree for future sharing.
    pub fn release_seq(&mut self, seq_id: SeqId) -> u32 {
        self.logical.release_seq(seq_id)
    }

    /// Number of tokens cached for a sequence.
    pub fn seq_token_count(&self, seq_id: SeqId) -> usize {
        self.logical.seq_token_count(seq_id)
    }

    /// Check if seq can accept one more token.
    pub fn can_append(&self, seq_id: SeqId) -> bool {
        self.logical.can_append(seq_id)
    }

    /// Probe how many tokens of a prompt are cached (read-only).
    pub fn probe_prefix(&self, tokens: &[u32]) -> usize {
        self.logical.probe_prefix(tokens)
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
            "PagedKvCacheManager(block_size={}, free_blocks={}/{}, tree_nodes={})",
            self.config.block_size,
            self.logical.free_blocks(),
            self.logical.total_blocks(),
            self.logical.tree_node_count(),
        )
    }
}

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
        let tokens: Vec<u32> = (0..50).collect();
        let cached = mgr.match_and_allocate(seq, &tokens).unwrap();
        assert_eq!(cached, 0); // no prefix cache yet
        assert_eq!(mgr.seq_token_count(seq), 50);

        // Register computed blocks for future prefix sharing
        mgr.insert_computed_blocks(seq, &tokens, cached);

        // Decode step
        assert!(mgr.append_token(seq));
        assert_eq!(mgr.seq_token_count(seq), 51);

        // Build block table
        let table = mgr.build_block_table(&[seq]);
        assert_eq!(table.len(), 1);
        assert!(!table[0].is_empty());

        // Release
        mgr.release_seq(seq);
    }

    #[test]
    fn test_prefix_sharing() {
        let config = Qwen3Config::qwen3_0_6b();
        let mut mgr = PagedKvCacheManager::new(&config);

        // Seq 1: tokens [0..32]
        let tokens1: Vec<u32> = (0..32).collect();
        let cached1 = mgr.match_and_allocate(SeqId(1), &tokens1).unwrap();
        assert_eq!(cached1, 0);
        mgr.insert_computed_blocks(SeqId(1), &tokens1, cached1);
        mgr.release_seq(SeqId(1));

        // Seq 2: tokens [0..48] — first 32 should be cached
        let tokens2: Vec<u32> = (0..48).collect();
        let cached2 = mgr.match_and_allocate(SeqId(2), &tokens2).unwrap();
        assert_eq!(cached2, 32); // 32 tokens (2 blocks of 16) cached from Seq 1!
    }
}
