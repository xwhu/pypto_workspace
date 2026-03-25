use std::collections::HashMap;
use crate::radix_tree::{BlockHash, BlockId, RadixCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(pub u64);

pub struct SequenceBlockTracker {
    pub seq_id: SequenceId,
    pub block_size: usize,
    /// Ordered list of physical blocks allocated to this sequence
    pub physical_blocks: Vec<BlockId>,
    /// Number of tokens currently in the last physical block
    pub last_block_len: usize,
    /// Accumulated cryptographic hashes for completely filled blocks
    pub block_hashes: Vec<BlockHash>,
}

impl SequenceBlockTracker {
    pub fn new(seq_id: SequenceId, block_size: usize) -> Self {
        Self {
            seq_id,
            block_size,
            physical_blocks: Vec::new(),
            last_block_len: 0,
            block_hashes: Vec::new(),
        }
    }

    /// Return the sequence's block table representing its physical memory allocation map
    pub fn get_block_table(&self) -> &[BlockId] {
        &self.physical_blocks
    }

    /// Get the logical number of tokens currently in this sequence cache
    pub fn seq_len(&self) -> usize {
        if self.physical_blocks.is_empty() {
            0
        } else {
            (self.physical_blocks.len() - 1) * self.block_size + self.last_block_len
        }
    }
}

pub struct BlockManager {
    pub cache: RadixCache,
    pub block_size: usize,
    pub active_seqs: HashMap<SequenceId, SequenceBlockTracker>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            cache: RadixCache::new(num_blocks),
            block_size,
            active_seqs: HashMap::new(),
        }
    }

    pub fn can_allocate(&self, num_blocks_needed: usize) -> bool {
        self.cache.num_free_blocks() >= num_blocks_needed
    }

    /// Allocates an initial tracking sequence, optionally attempting a Prefix Cache hit
    /// using the prompt's tokens if Prefix Caching is enabled.
    /// `prompt_tokens` is a continuous slice.
    pub fn allocate_prefix(&mut self, seq_id: SequenceId, prompt_tokens: &[u32]) -> Result<(), String> {
        let mut tracker = SequenceBlockTracker::new(seq_id, self.block_size);
        
        // 1. Chunk prompt tokens into blocks
        let num_blocks = (prompt_tokens.len() + self.block_size - 1) / self.block_size;
        let num_full_blocks = prompt_tokens.len() / self.block_size;
        
        let mut hashes_to_match = Vec::new();
        let mut current_hash = None;

        for i in 0..num_full_blocks {
            let start = i * self.block_size;
            let end = start + self.block_size;
            let hash = BlockHash::compute(current_hash, &prompt_tokens[start..end]);
            hashes_to_match.push(hash);
            current_hash = Some(hash);
        }

        // 2. Perform O(1) hash chain matching
        let matched_blocks = self.cache.match_prefix(&hashes_to_match);
        let num_matched = matched_blocks.len();

        // 3. Populate tracker with matched blocks (already ref_cnt incremented by RadixCache)
        for i in 0..num_matched {
            tracker.physical_blocks.push(matched_blocks[i]);
            tracker.block_hashes.push(hashes_to_match[i]);
        }
        
        // 4. Allocate remaining blocks from free queue
        let blocks_to_alloc = num_blocks - num_matched;
        if !self.can_allocate(blocks_to_alloc) {
            // Rollback references
            self.cache.free_multiple(&matched_blocks);
            return Err("Out of memory for KV blocks".into());
        }

        for i in num_matched..num_blocks {
            let b = self.cache.allocate().unwrap();
            tracker.physical_blocks.push(b);
            
            // If it is a completely full block (during prompt chunking), we can cache it immediately
            if i < num_full_blocks {
                let hash = hashes_to_match[i];
                self.cache.cache_block(b, hash);
                tracker.block_hashes.push(hash);
            }
        }
        
        tracker.last_block_len = prompt_tokens.len() - ((num_blocks - 1) * self.block_size);
        if prompt_tokens.len() == 0 {
            tracker.last_block_len = 0;
        }

        self.active_seqs.insert(seq_id, tracker);
        Ok(())
    }

    /// Appends a newly decoded token to a sequence.
    /// In a real scenario, the caller provides the token ID so we can hash the block once it's full.
    /// For KV block allocation, we return whether a new block allocation occurred and the target Block ID.
    pub fn append_slot(&mut self, seq_id: SequenceId, token_id: Option<u32>) -> Result<(BlockId, usize), String> {
        let tracker = self.active_seqs.get_mut(&seq_id).ok_or("Sequence not found")?;

        if tracker.physical_blocks.is_empty() {
            let b = self.cache.allocate().ok_or("OOM")?;
            tracker.physical_blocks.push(b);
            tracker.last_block_len = 1;
            return Ok((b, 0)); // returned block_id, offset
        }

        let mut offset = tracker.last_block_len;

        if tracker.last_block_len == self.block_size {
            // Previous block is completely full, we should allocate a new one.
            // Note: In an optimal prefix cache, we would have hashed the previous block's tokens.
            // Since we incrementally decoded, the caller must supply the sequence of tokens or we
            // just manage memory here. Let's simplify by assuming the caller will hash it later if needed.
            let b = self.cache.allocate().ok_or("OOM")?;
            tracker.physical_blocks.push(b);
            tracker.last_block_len = 1;
            offset = 0;
        } else {
            tracker.last_block_len += 1;
        }
        
        let block_id = *tracker.physical_blocks.last().unwrap();
        Ok((block_id, offset))
    }

    /// Free a sequence and decrement all its block references
    pub fn free_sequence(&mut self, seq_id: SequenceId) {
        if let Some(tracker) = self.active_seqs.remove(&seq_id) {
            self.cache.free_multiple(&tracker.physical_blocks);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_manager_allocation_and_prefix_caching() {
        let mut manager = BlockManager::new(10, 4); // 10 blocks, block_size 4
        
        // Seq 1 gets 9 tokens (2 full blocks, 1 partial)
        let prompt1: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        manager.allocate_prefix(SequenceId(1), &prompt1).unwrap();
        
        {
            let t1 = manager.active_seqs.get(&SequenceId(1)).unwrap();
            assert_eq!(t1.physical_blocks.len(), 3);
            assert_eq!(t1.last_block_len, 1);
            assert_eq!(t1.seq_len(), 9);
            assert_eq!(t1.block_hashes.len(), 2); // only two complete blocks were hashed
        }
        
        // Exact completely overlapping prefix request
        let prompt2: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8]; // exactly 2 blocks
        manager.allocate_prefix(SequenceId(2), &prompt2).unwrap();
        
        {
            let t1 = manager.active_seqs.get(&SequenceId(1)).unwrap();
            let t2 = manager.active_seqs.get(&SequenceId(2)).unwrap();
            assert_eq!(t2.physical_blocks.len(), 2);
            assert_eq!(t2.last_block_len, 4); // The second block is full
            
            // Assert they share physical memory
            assert_eq!(t1.physical_blocks[0], t2.physical_blocks[0]);
            assert_eq!(t1.physical_blocks[1], t2.physical_blocks[1]);
            assert_eq!(manager.cache.get_block_ref(t1.physical_blocks[0]), 2);
        }
        
        // Append slot to seq 1 (now 10 tokens)
        let (bid, offset) = manager.append_slot(SequenceId(1), Some(10)).unwrap();
        
        {
            let t1 = manager.active_seqs.get(&SequenceId(1)).unwrap();
            assert_eq!(bid, t1.physical_blocks[2]);
            assert_eq!(offset, 1);
            assert_eq!(t1.seq_len(), 10);
        }
        
        // Free sequences
        manager.free_sequence(SequenceId(1));
        manager.free_sequence(SequenceId(2));
    }
}
