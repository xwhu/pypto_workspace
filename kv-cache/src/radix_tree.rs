use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub type BlockId = u32;

/// A stable hash representing the sequence of tokens from the beginning to the end of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub u64);

impl BlockHash {
    /// Compute the hash of a new block extending `parent_hash` with `tokens`.
    pub fn compute(parent_hash: Option<BlockHash>, tokens: &[u32]) -> Self {
        let mut hasher = DefaultHasher::new();
        if let Some(h) = parent_hash {
            h.0.hash(&mut hasher);
        } else {
            0u64.hash(&mut hasher);
        }
        tokens.hash(&mut hasher);
        BlockHash(hasher.finish())
    }
}

/// Represents a single fixed-size physical KV Cache block.
#[derive(Debug)]
pub struct KVCacheBlock {
    pub block_id: BlockId,
    pub ref_cnt: u32,
    pub block_hash: Option<BlockHash>,
    
    // Doubly linked list pointers for the Free Queue (LRU eviction).
    // Using `u32::MAX` to denote `None`.
    pub prev_free: u32,
    pub next_free: u32,
}

impl KVCacheBlock {
    pub fn new(block_id: BlockId) -> Self {
        Self {
            block_id,
            ref_cnt: 0,
            block_hash: None,
            prev_free: u32::MAX,
            next_free: u32::MAX,
        }
    }
}

/// A Doubly Linked List for Free Blocks.
/// O(1) Push Back (LRU tail), Pop Front (LRU head/eviction), and Remove Middle (Cache Hit/Reuse).
struct FreeQueue {
    head: u32,
    tail: u32,
    len: usize,
}

impl FreeQueue {
    fn new() -> Self {
        Self { head: u32::MAX, tail: u32::MAX, len: 0 }
    }

    fn push_back(&mut self, block: u32, blocks: &mut [KVCacheBlock]) {
        blocks[block as usize].prev_free = self.tail;
        blocks[block as usize].next_free = u32::MAX;

        if self.tail != u32::MAX {
            blocks[self.tail as usize].next_free = block;
        } else {
            self.head = block;
        }
        self.tail = block;
        self.len += 1;
    }

    fn pop_front(&mut self, blocks: &mut [KVCacheBlock]) -> Option<u32> {
        if self.head == u32::MAX {
            return None;
        }
        let b = self.head;
        self.head = blocks[b as usize].next_free;
        if self.head != u32::MAX {
            blocks[self.head as usize].prev_free = u32::MAX;
        } else {
            self.tail = u32::MAX;
        }
        blocks[b as usize].next_free = u32::MAX;
        self.len -= 1;
        Some(b)
    }

    fn remove(&mut self, block: u32, blocks: &mut [KVCacheBlock]) {
        let prev = blocks[block as usize].prev_free;
        let next = blocks[block as usize].next_free;

        if prev != u32::MAX {
            blocks[prev as usize].next_free = next;
        } else {
            self.head = next;
        }

        if next != u32::MAX {
            blocks[next as usize].prev_free = prev;
        } else {
            self.tail = prev;
        }

        blocks[block as usize].prev_free = u32::MAX;
        blocks[block as usize].next_free = u32::MAX;
        self.len -= 1;
    }
}

/// An O(1) Prefix Cache that simulates a Radix Tree using a Hash Chain.
/// This matches the architecture of vLLM v1.
pub struct RadixCache {
    /// Global block pool holding the physical states of blocks
    blocks: Vec<KVCacheBlock>,
    /// Free/Evictable block queue ordered by LRU (Least Recently Used at the head)
    free_queue: FreeQueue,
    /// O(1) lookup table: BlockHash -> BlockId
    cached_blocks: HashMap<BlockHash, BlockId>,
}

impl RadixCache {
    pub fn new(num_blocks: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_queue = FreeQueue::new();
        
        for i in 0..num_blocks {
            blocks.push(KVCacheBlock::new(i as u32));
            free_queue.push_back(i as u32, &mut blocks);
        }

        Self {
            blocks,
            free_queue,
            cached_blocks: HashMap::new(),
        } // All blocks start with ref_cnt = 0 and are in the free queue
    }

    pub fn num_free_blocks(&self) -> usize {
        self.free_queue.len
    }

    /// Recursively match a sequence of hashes representing the request's prefix.
    /// Returns the matched block IDs.
    pub fn match_prefix(&mut self, hashes: &[BlockHash]) -> Vec<BlockId> {
        let mut matched = Vec::new();

        for hash in hashes {
            if let Some(&block_id) = self.cached_blocks.get(hash) {
                if self.blocks[block_id as usize].ref_cnt == 0 {
                    // It was in the free queue, but now it's hit! Remove it so it isn't evicted.
                    self.free_queue.remove(block_id, &mut self.blocks);
                }
                self.blocks[block_id as usize].ref_cnt += 1;
                matched.push(block_id);
            } else {
                break;
            }
        }
        
        // Note: For Prefix Cache, if a block is matched, it fundamentally means its prefix was matched.
        matched
    }

    /// Allocate a single new block. If no free blocks, evict the least recently used block.
    pub fn allocate(&mut self) -> Option<BlockId> {
        // Pop the LRU block from the head of the free queue
        if let Some(block_id) = self.free_queue.pop_front(&mut self.blocks) {
            let b = &mut self.blocks[block_id as usize];
            
            // If the block was caching a prefix, we MUST evict it from the hash map!
            if let Some(hash) = b.block_hash.take() {
                self.cached_blocks.remove(&hash);
            }
            
            b.ref_cnt = 1;
            return Some(block_id);
        }
        None // OOM
    }

    /// Allocate multiple blocks. Returns None if there's insufficient capacity.
    pub fn allocate_n(&mut self, count: usize) -> Option<Vec<BlockId>> {
        if self.num_free_blocks() < count {
            return None; // Ensure atomic allocation (avoids partial OOM crashes)
        }
        let mut allocated = Vec::with_capacity(count);
        for _ in 0..count {
            allocated.push(self.allocate().unwrap());
        }
        Some(allocated)
    }

    /// Free a block. If its ref_cnt hits 0, it becomes historically cached and evictable.
    pub fn free(&mut self, block_id: BlockId) {
        let b = &mut self.blocks[block_id as usize];
        assert!(b.ref_cnt > 0, "Double free on block {}", block_id);
        b.ref_cnt -= 1;
        
        if b.ref_cnt == 0 {
            // Re-enters free queue at the TAIL (Most Recently Used).
            // It might still contain useful cached data and can be a cache hit later.
            self.free_queue.push_back(block_id, &mut self.blocks);
        }
    }

    pub fn free_multiple(&mut self, blocks: &[BlockId]) {
        // To preserve eviction order optimally, we free them in sequence
        for &b in blocks {
            self.free(b);
        }
    }

    /// Cache a full block so it can be retrieved by `match_prefix` later.
    /// This happens once a block finishes computing.
    pub fn cache_block(&mut self, block_id: BlockId, hash: BlockHash) {
        let b = &mut self.blocks[block_id as usize];
        assert!(b.ref_cnt > 0, "Cannot cache a freed block");
        if b.block_hash.is_some() {
            // Already cached, likely from a previous identical prefix computation
            return;
        }
        b.block_hash = Some(hash);
        self.cached_blocks.insert(hash, block_id);
    }

    /// Add an external reference to an existing block (e.g., when branching for beam search or PagedAttention).
    pub fn add_ref(&mut self, block_id: BlockId) {
        if self.blocks[block_id as usize].ref_cnt == 0 {
            self.free_queue.remove(block_id, &mut self.blocks);
        }
        self.blocks[block_id as usize].ref_cnt += 1;
    }

    pub fn get_block_ref(&self, block_id: BlockId) -> u32 {
        self.blocks[block_id as usize].ref_cnt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_cache_allocation_and_eviction() {
        let mut cache = RadixCache::new(3); // 3 blocks total
        
        let b1 = cache.allocate().unwrap();
        let b2 = cache.allocate().unwrap();
        let b3 = cache.allocate().unwrap();
        assert_eq!(cache.allocate(), None); // OOM
        
        // Cache prefixes
        let h1 = BlockHash::compute(None, &[1, 2, 3]);
        let h2 = BlockHash::compute(Some(h1), &[4, 5, 6]);
        
        cache.cache_block(b1, h1);
        cache.cache_block(b2, h2);
        
        // Free blocks (moves to free queue tail)
        cache.free(b1);
        cache.free(b2);
        
        // Match prefix
        let hits = cache.match_prefix(&[h1, h2]);
        assert_eq!(hits, vec![b1, b2]);
        assert_eq!(cache.get_block_ref(b1), 1);
        assert_eq!(cache.get_block_ref(b2), 1);
        
        // Free again
        cache.free(b1);
        cache.free(b2);
        
        // Allocate should evict b1 because b1 is at the head of the free queue (oldest freed)
        cache.free(b3); // Now b3 is freed (moved to tail)
        
        let new_b = cache.allocate().unwrap();
        assert_eq!(new_b, b1); // b1 is the oldest freed block and thus LRU.
    }
}
