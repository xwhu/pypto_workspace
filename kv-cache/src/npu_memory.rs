//! NPU Physical KV Cache Memory Pool.
//!
//! Manages the actual device memory for the KV Cache on Ascend NPU.
//! Pre-allocates a large contiguous buffer per layer and maps logical
//! block IDs to physical offsets within that buffer.
//!
//! KV Cache Physical Layout (per layer):
//!   key_cache:   [num_blocks, block_size, num_kv_heads, head_dim]  FP16
//!   value_cache: [num_blocks, block_size, num_kv_heads, head_dim]  FP16
//!
//! This matches the layout expected by:
//!   - torch_npu._npu_paged_attention (decode)
//!   - torch_npu.npu_fused_infer_attention_score (prefill/mixed)

use crate::radix_tree::BlockId;

/// Configuration for the physical KV cache memory pool.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of Key-Value heads (after GQA grouping).
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Number of tokens per physical block.
    pub block_size: usize,
    /// Total number of physical blocks in the pool.
    pub num_blocks: usize,
}

impl KVCacheConfig {
    /// Bytes needed for ONE key or value cache across all blocks for ONE layer.
    /// Shape: [num_blocks, block_size, num_kv_heads, head_dim] in FP16
    pub fn layer_cache_bytes(&self) -> usize {
        self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim * 2 // FP16 = 2 bytes
    }

    /// Total bytes for all layers (K + V).
    pub fn total_bytes(&self) -> usize {
        self.num_layers * 2 * self.layer_cache_bytes()
    }

    /// Byte offset of a specific (layer, block_id, is_value) within the global pool.
    /// Used to compute the device pointer for a specific block's K or V data.
    pub fn block_offset(&self, layer: usize, block_id: BlockId, is_value: bool) -> usize {
        let layer_offset = layer * 2 * self.layer_cache_bytes();
        let kv_offset = if is_value { self.layer_cache_bytes() } else { 0 };
        let block_bytes = self.block_size * self.num_kv_heads * self.head_dim * 2;
        let block_offset = (block_id as usize) * block_bytes;
        layer_offset + kv_offset + block_offset
    }

    /// Byte offset for a specific token slot within a block.
    /// slot_offset = token_offset_in_block * num_kv_heads * head_dim * sizeof(FP16)
    pub fn slot_offset_in_block(&self, token_offset: usize) -> usize {
        token_offset * self.num_kv_heads * self.head_dim * 2
    }

    /// Compute the global slot mapping index for reshape_and_cache.
    /// slot = block_id * block_size + token_offset_in_block
    pub fn slot_mapping(block_id: BlockId, token_offset: usize, block_size: usize) -> usize {
        (block_id as usize) * block_size + token_offset
    }
}

/// Represents the pre-allocated physical KV cache pool on NPU device memory.
/// This is a metadata-only struct — actual device allocation happens via CANN APIs
/// in the engine layer (`aclrtMalloc`).
///
/// The pool holds:
///   - For each layer: key_cache [num_blocks, block_size, num_kv_heads, head_dim]
///   - For each layer: value_cache [num_blocks, block_size, num_kv_heads, head_dim]
///
/// All stored contiguously in a single device allocation for cache coherency.
pub struct KVCachePool {
    pub config: KVCacheConfig,
    /// Base device pointer (set after aclrtMalloc).
    /// None if not yet allocated (e.g., in unit test mode).
    pub device_base_ptr: Option<usize>,
}

impl KVCachePool {
    pub fn new(config: KVCacheConfig) -> Self {
        Self {
            config,
            device_base_ptr: None,
        }
    }

    /// Set the device base pointer after CANN allocation.
    pub fn set_device_ptr(&mut self, ptr: usize) {
        self.device_base_ptr = Some(ptr);
    }

    /// Get the device pointer for a specific layer's key cache tensor.
    /// Returns the pointer to the start of [num_blocks, block_size, num_kv_heads, head_dim].
    pub fn key_cache_ptr(&self, layer: usize) -> usize {
        let base = self.device_base_ptr.expect("KV cache not allocated on device");
        base + self.config.block_offset(layer, 0, false)
    }

    /// Get the device pointer for a specific layer's value cache tensor.
    pub fn value_cache_ptr(&self, layer: usize) -> usize {
        let base = self.device_base_ptr.expect("KV cache not allocated on device");
        base + self.config.block_offset(layer, 0, true)
    }

    /// Build a slot_mapping vector for a batch of sequences.
    /// Input: for each token in the batch, (block_id, offset_in_block).
    /// Output: flat i32 array of slots for reshape_and_cache.
    pub fn build_slot_mapping(slots: &[(BlockId, usize)], block_size: usize) -> Vec<i32> {
        slots.iter()
            .map(|&(bid, off)| KVCacheConfig::slot_mapping(bid, off, block_size) as i32)
            .collect()
    }

    /// Build a block_table tensor for a batch of sequences.
    /// Input: for each sequence, its ordered list of block_ids.
    /// Output: 2D i32 array [batch_size, max_blocks_per_seq] (zero-padded).
    pub fn build_block_table(
        seq_block_ids: &[Vec<BlockId>],
        max_blocks_per_seq: usize,
    ) -> Vec<i32> {
        let mut table = vec![0i32; seq_block_ids.len() * max_blocks_per_seq];
        for (seq_idx, blocks) in seq_block_ids.iter().enumerate() {
            for (blk_idx, &bid) in blocks.iter().enumerate() {
                if blk_idx < max_blocks_per_seq {
                    table[seq_idx * max_blocks_per_seq + blk_idx] = bid as i32;
                }
            }
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_config_sizing() {
        let config = KVCacheConfig {
            num_layers: 28,
            num_kv_heads: 4,
            head_dim: 128,
            block_size: 16,
            num_blocks: 256,
        };

        // Per-layer K or V: 256 * 16 * 4 * 128 * 2 = 67_108_864 bytes = 64 MiB
        assert_eq!(config.layer_cache_bytes(), 256 * 16 * 4 * 128 * 2);
        
        // Total: 28 layers * 2 (K+V) * 64 MiB = 3584 MiB
        assert_eq!(config.total_bytes(), 28 * 2 * config.layer_cache_bytes());
    }

    #[test]
    fn test_slot_mapping() {
        assert_eq!(KVCacheConfig::slot_mapping(5, 3, 16), 83); // 5*16 + 3
    }

    #[test]
    fn test_block_table_building() {
        let seq_blocks = vec![
            vec![0, 1, 2],
            vec![3, 4],
        ];
        let table = KVCachePool::build_block_table(&seq_blocks, 4);
        assert_eq!(table, vec![
            0, 1, 2, 0,  // seq 0, padded
            3, 4, 0, 0,  // seq 1, padded
        ]);
    }
}
