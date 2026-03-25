/// Logical identifier for a KV cache block.
///
/// This is NOT a GPU pointer — it is an index into the pre-allocated GPU block
/// pool. The attention kernel receives a `block_table: Vec<Vec<u32>>` that maps
/// each sequence's logical block index to a physical `BlockId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Sentinel value meaning "no block assigned".
    pub const INVALID: BlockId = BlockId(u32::MAX);

    /// Convert to raw u32 for kernel interface.
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

/// Identifier for an inference sequence (request).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(pub u64);

/// Configuration for the KV cache, derived from the model config.
///
/// These parameters determine the memory layout of each block and the number
/// of tokens each block can hold. They must match the layout the attention
/// kernel expects.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Number of tokens stored in each block (typically 16).
    pub block_size: usize,
    /// Number of transformer layers (per this PP rank).
    pub num_layers: usize,
    /// Number of KV attention heads (per this TP rank).
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Bytes per element for KV cache (2 for BF16/FP16, 1 for FP8/INT8).
    pub dtype_size: usize,
}

impl KvCacheConfig {
    /// Compute the GPU memory size (in bytes) of a single KV block.
    ///
    /// Layout: `[2(K+V), num_layers, block_size, num_kv_heads, head_dim]`
    ///
    /// Each block stores both K and V for all layers, for `block_size` tokens.
    pub fn block_bytes(&self) -> usize {
        2 * self.num_layers * self.block_size * self.num_kv_heads * self.head_dim * self.dtype_size
    }

    /// Number of blocks needed to hold `num_tokens` tokens.
    pub fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }
}

/// Locates a specific token within the block-based KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenPosition {
    /// Index of the block within the sequence's block list.
    pub block_idx: usize,
    /// Offset of the token within that block (0..block_size-1).
    pub offset: usize,
}

impl KvCacheConfig {
    /// Map a token index (within a sequence) to its block position.
    pub fn token_position(&self, token_idx: usize) -> TokenPosition {
        TokenPosition {
            block_idx: token_idx / self.block_size,
            offset: token_idx % self.block_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> KvCacheConfig {
        KvCacheConfig {
            block_size: 16,
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            dtype_size: 2, // BF16
        }
    }

    #[test]
    fn test_block_bytes() {
        let cfg = test_config();
        // 2 * 32 * 16 * 8 * 128 * 2 = 2_097_152 = 2 MB per block
        assert_eq!(cfg.block_bytes(), 2_097_152);
    }

    #[test]
    fn test_blocks_for_tokens() {
        let cfg = test_config();
        assert_eq!(cfg.blocks_for_tokens(0), 0);
        assert_eq!(cfg.blocks_for_tokens(1), 1);
        assert_eq!(cfg.blocks_for_tokens(16), 1);
        assert_eq!(cfg.blocks_for_tokens(17), 2);
        assert_eq!(cfg.blocks_for_tokens(32), 2);
        assert_eq!(cfg.blocks_for_tokens(33), 3);
    }

    #[test]
    fn test_token_position() {
        let cfg = test_config();
        assert_eq!(
            cfg.token_position(0),
            TokenPosition {
                block_idx: 0,
                offset: 0
            }
        );
        assert_eq!(
            cfg.token_position(15),
            TokenPosition {
                block_idx: 0,
                offset: 15
            }
        );
        assert_eq!(
            cfg.token_position(16),
            TokenPosition {
                block_idx: 1,
                offset: 0
            }
        );
        assert_eq!(
            cfg.token_position(35),
            TokenPosition {
                block_idx: 2,
                offset: 3
            }
        );
    }

    #[test]
    fn test_block_id_invalid() {
        assert_eq!(BlockId::INVALID.as_u32(), u32::MAX);
    }
}
