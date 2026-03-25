use std::collections::HashMap;

use crate::block_pool::BlockPool;
use crate::types::{BlockId, KvCacheConfig, SeqId};

/// Per-sequence KV cache state.
#[derive(Debug)]
struct SeqKvState {
    /// Ordered list of blocks allocated for this sequence.
    block_ids: Vec<BlockId>,
    /// Total number of tokens currently stored.
    token_count: usize,
}

/// Minimal KV cache manager: each sequence owns its blocks exclusively.
///
/// This is the MVP implementation — **no prefix sharing** between sequences.
/// A `RadixKvManager` (v4 §9) can replace this later to enable prefix caching.
///
/// # Responsibilities
/// - Allocate blocks for new sequences (prefill)
/// - Extend blocks as tokens are appended (decode)
/// - Release blocks when sequences finish
/// - Build `block_table` for attention kernels
pub struct SimpleKvManager {
    config: KvCacheConfig,
    pool: BlockPool,
    sequences: HashMap<SeqId, SeqKvState>,
}

impl SimpleKvManager {
    /// Create a new KV manager with the given configuration and block budget.
    pub fn new(config: KvCacheConfig, num_blocks: u32) -> Self {
        Self {
            config,
            pool: BlockPool::new(num_blocks),
            sequences: HashMap::new(),
        }
    }

    // ── Capacity queries ─────────────────────────────────────────────

    /// Check if `blocks_needed` blocks can be allocated right now.
    pub fn can_allocate(&self, blocks_needed: u32) -> bool {
        self.pool.free_count() >= blocks_needed
    }

    /// Check if `seq_id` can accept one more token without new allocation,
    /// OR if a new block can be allocated for it.
    pub fn can_append(&self, seq_id: SeqId) -> bool {
        match self.sequences.get(&seq_id) {
            None => false,
            Some(state) => {
                let pos = self.config.token_position(state.token_count);
                // If offset > 0, we still have space in the current block
                if pos.offset > 0 || state.token_count == 0 {
                    true
                } else {
                    // Need a new block
                    self.pool.free_count() > 0
                }
            }
        }
    }

    /// Number of free blocks available.
    pub fn free_blocks(&self) -> u32 {
        self.pool.free_count()
    }

    /// Total blocks in the pool.
    pub fn total_blocks(&self) -> u32 {
        self.pool.total_count()
    }

    // ── Prefill: allocate blocks for a new sequence ──────────────────

    /// Allocate blocks for a new sequence with `num_prompt_tokens` tokens.
    ///
    /// Returns `true` on success, `false` if insufficient blocks.
    /// The sequence must not already exist.
    pub fn allocate_for_seq(&mut self, seq_id: SeqId, num_prompt_tokens: usize) -> bool {
        debug_assert!(
            !self.sequences.contains_key(&seq_id),
            "Sequence {:?} already exists",
            seq_id
        );

        let blocks_needed = self.config.blocks_for_tokens(num_prompt_tokens) as u32;
        match self.pool.allocate(blocks_needed) {
            Some(block_ids) => {
                self.sequences.insert(
                    seq_id,
                    SeqKvState {
                        block_ids,
                        token_count: num_prompt_tokens,
                    },
                );
                true
            }
            None => false,
        }
    }

    // ── Decode: append tokens one at a time ──────────────────────────

    /// Append one generated token to a sequence's KV cache.
    ///
    /// If the current last block is full, a new block is allocated.
    /// Returns the `BlockId` where the new token resides, or `None` if
    /// a new block was needed but none are available.
    pub fn append_token(&mut self, seq_id: SeqId) -> Option<BlockId> {
        let state = self.sequences.get_mut(&seq_id)?;
        let pos = self.config.token_position(state.token_count);

        if pos.offset == 0 && state.token_count > 0 {
            // Current block is full — need a new one
            match self.pool.allocate(1) {
                Some(new_blocks) => {
                    state.block_ids.push(new_blocks[0]);
                }
                None => return None,
            }
        }

        state.token_count += 1;

        // Return the block where this token landed
        let final_pos = self.config.token_position(state.token_count - 1);
        Some(state.block_ids[final_pos.block_idx])
    }

    // ── Release: free all blocks of a finished sequence ──────────────

    /// Release all blocks owned by a sequence.
    ///
    /// Returns the number of blocks freed, or 0 if the sequence was not found.
    pub fn release_seq(&mut self, seq_id: SeqId) -> u32 {
        match self.sequences.remove(&seq_id) {
            Some(state) => {
                let count = state.block_ids.len() as u32;
                self.pool.free(&state.block_ids);
                count
            }
            None => 0,
        }
    }

    // ── Query ────────────────────────────────────────────────────────

    /// Get the block IDs for a sequence (for building kernel block_table).
    pub fn get_block_ids(&self, seq_id: SeqId) -> Option<&[BlockId]> {
        self.sequences.get(&seq_id).map(|s| s.block_ids.as_slice())
    }

    /// Number of tokens cached for a sequence.
    pub fn seq_token_count(&self, seq_id: SeqId) -> usize {
        self.sequences
            .get(&seq_id)
            .map(|s| s.token_count)
            .unwrap_or(0)
    }

    /// Build the `block_table` for a batch of sequences.
    ///
    /// The output is a `Vec<Vec<u32>>` where each inner vec is the ordered
    /// list of physical block IDs for one sequence. This is passed directly
    /// to the paged attention kernel.
    pub fn build_block_table(&self, seq_ids: &[SeqId]) -> Vec<Vec<u32>> {
        seq_ids
            .iter()
            .map(|id| {
                self.sequences
                    .get(id)
                    .map(|s| s.block_ids.iter().map(|b| b.as_u32()).collect())
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Number of active sequences being tracked.
    pub fn active_seq_count(&self) -> usize {
        self.sequences.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> KvCacheConfig {
        KvCacheConfig {
            block_size: 4, // small for easier testing
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            dtype_size: 2,
        }
    }

    #[test]
    fn prefill_decode_release_lifecycle() {
        let mut mgr = SimpleKvManager::new(test_config(), 10);
        let seq = SeqId(1);

        // Prefill: 6 tokens → blocks_for_tokens(6) = 2 blocks (block_size=4)
        assert!(mgr.allocate_for_seq(seq, 6));
        assert_eq!(mgr.seq_token_count(seq), 6);
        assert_eq!(mgr.get_block_ids(seq).unwrap().len(), 2);
        assert_eq!(mgr.free_blocks(), 8);

        // Decode: append 2 tokens (fills block 1: offset 2 → 3, then needs block 2)
        // Token 6 → block 1 offset 2 (still room)
        assert!(mgr.append_token(seq).is_some());
        assert_eq!(mgr.seq_token_count(seq), 7);
        assert_eq!(mgr.get_block_ids(seq).unwrap().len(), 2); // still 2 blocks

        // Token 7 → block 1 offset 3 (fills block 1)
        assert!(mgr.append_token(seq).is_some());
        assert_eq!(mgr.seq_token_count(seq), 8);
        assert_eq!(mgr.get_block_ids(seq).unwrap().len(), 2); // still 2 blocks (exactly full)

        // Token 8 → needs block 2 (new block)
        assert!(mgr.append_token(seq).is_some());
        assert_eq!(mgr.seq_token_count(seq), 9);
        assert_eq!(mgr.get_block_ids(seq).unwrap().len(), 3); // now 3 blocks
        assert_eq!(mgr.free_blocks(), 7);

        // Release
        let freed = mgr.release_seq(seq);
        assert_eq!(freed, 3);
        assert_eq!(mgr.free_blocks(), 10);
        assert_eq!(mgr.active_seq_count(), 0);
    }

    #[test]
    fn multiple_sequences() {
        let mut mgr = SimpleKvManager::new(test_config(), 10);

        // Seq 1: 4 tokens = 1 block
        assert!(mgr.allocate_for_seq(SeqId(1), 4));
        // Seq 2: 5 tokens = 2 blocks
        assert!(mgr.allocate_for_seq(SeqId(2), 5));
        // Seq 3: 8 tokens = 2 blocks
        assert!(mgr.allocate_for_seq(SeqId(3), 8));

        assert_eq!(mgr.active_seq_count(), 3);
        assert_eq!(mgr.free_blocks(), 5); // 10 - 1 - 2 - 2 = 5

        let table = mgr.build_block_table(&[SeqId(1), SeqId(2), SeqId(3)]);
        assert_eq!(table.len(), 3);
        assert_eq!(table[0].len(), 1);
        assert_eq!(table[1].len(), 2);
        assert_eq!(table[2].len(), 2);

        // Release seq 2, check space is recovered
        mgr.release_seq(SeqId(2));
        assert_eq!(mgr.free_blocks(), 7);
        assert_eq!(mgr.active_seq_count(), 2);
    }

    #[test]
    fn allocation_fails_when_exhausted() {
        let mut mgr = SimpleKvManager::new(test_config(), 3);

        // 10 tokens → needs 3 blocks → uses all 3
        assert!(mgr.allocate_for_seq(SeqId(1), 10));
        assert_eq!(mgr.free_blocks(), 0);

        // Can't allocate another
        assert!(!mgr.allocate_for_seq(SeqId(2), 1));
        assert_eq!(mgr.active_seq_count(), 1); // seq 2 was NOT created

        // Append also fails (needs new block but none free)
        // First fill current block
        assert!(mgr.append_token(SeqId(1)).is_some()); // token 10 → block 2 offset 2
        assert!(mgr.append_token(SeqId(1)).is_some()); // token 11 → block 2 offset 3 (full)
        assert!(mgr.append_token(SeqId(1)).is_none()); // needs block 3, none free
    }

    #[test]
    fn release_recovers_space() {
        let mut mgr = SimpleKvManager::new(test_config(), 4);

        assert!(mgr.allocate_for_seq(SeqId(1), 8)); // 2 blocks
        assert!(mgr.allocate_for_seq(SeqId(2), 8)); // 2 blocks
        assert_eq!(mgr.free_blocks(), 0);

        // Can't add more
        assert!(!mgr.allocate_for_seq(SeqId(3), 1));

        // Release seq 1 → 2 blocks free
        mgr.release_seq(SeqId(1));
        assert_eq!(mgr.free_blocks(), 2);

        // Now can allocate for seq 3
        assert!(mgr.allocate_for_seq(SeqId(3), 5)); // 2 blocks
        assert_eq!(mgr.free_blocks(), 0);
    }

    #[test]
    fn build_block_table_for_unknown_seq() {
        let mgr = SimpleKvManager::new(test_config(), 10);
        let table = mgr.build_block_table(&[SeqId(999)]);
        assert_eq!(table.len(), 1);
        assert!(table[0].is_empty());
    }

    #[test]
    fn block_table_ids_are_distinct() {
        let mut mgr = SimpleKvManager::new(test_config(), 10);
        assert!(mgr.allocate_for_seq(SeqId(1), 12)); // 3 blocks

        let table = mgr.build_block_table(&[SeqId(1)]);
        let ids = &table[0];
        assert_eq!(ids.len(), 3);

        // All IDs should be distinct
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn can_append_when_block_has_room() {
        let mut mgr = SimpleKvManager::new(test_config(), 2);
        assert!(mgr.allocate_for_seq(SeqId(1), 1)); // 1 block, 3 slots free
        assert_eq!(mgr.free_blocks(), 1);

        // Block has room (offset 1, 2, 3 still free)
        assert!(mgr.can_append(SeqId(1)));
    }

    #[test]
    fn can_append_when_block_full_but_pool_has_space() {
        let mut mgr = SimpleKvManager::new(test_config(), 2);
        assert!(mgr.allocate_for_seq(SeqId(1), 4)); // 1 block, exactly full
        assert_eq!(mgr.free_blocks(), 1);

        // Block is full but pool has 1 free → can append
        assert!(mgr.can_append(SeqId(1)));
    }

    #[test]
    fn cannot_append_when_block_full_and_pool_empty() {
        let mut mgr = SimpleKvManager::new(test_config(), 1);
        assert!(mgr.allocate_for_seq(SeqId(1), 4)); // 1 block, exactly full
        assert_eq!(mgr.free_blocks(), 0);

        // Block is full AND pool is empty → cannot append
        assert!(!mgr.can_append(SeqId(1)));
    }
}
