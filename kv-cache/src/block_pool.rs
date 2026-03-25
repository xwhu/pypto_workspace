use crate::types::BlockId;

/// Manages a pool of fixed-size KV cache blocks.
///
/// This is the **logical layer** — it tracks which block IDs are free or
/// allocated, but does NOT manage actual GPU memory. The GPU memory pool
/// (a single large pre-allocated buffer) is managed separately; this pool
/// only hands out indices into that buffer.
///
/// Corresponds to v4 §7's BlockPool but without the device-memory backing.
pub struct BlockPool {
    /// Stack of free block IDs. Pop from the end for O(1) allocation.
    free_list: Vec<BlockId>,
    /// Total number of blocks in the pool.
    total: u32,
}

impl BlockPool {
    /// Create a new pool with `num_blocks` blocks, all initially free.
    ///
    /// Block IDs are `0..num_blocks`.
    pub fn new(num_blocks: u32) -> Self {
        let free_list = (0..num_blocks).rev().map(BlockId).collect();
        Self {
            free_list,
            total: num_blocks,
        }
    }

    /// Allocate `count` blocks. Returns `None` if insufficient free blocks.
    ///
    /// On success, all returned blocks are removed from the free list.
    /// On failure (not enough blocks), *no* blocks are allocated (atomic).
    pub fn allocate(&mut self, count: u32) -> Option<Vec<BlockId>> {
        if count == 0 {
            return Some(Vec::new());
        }
        if (self.free_list.len() as u32) < count {
            return None;
        }
        let start = self.free_list.len() - count as usize;
        let allocated = self.free_list.split_off(start);
        Some(allocated)
    }

    /// Free blocks, returning them to the pool.
    ///
    /// # Panics
    ///
    /// Debug-mode panics if any block ID is out of range.
    pub fn free(&mut self, blocks: &[BlockId]) {
        for &block in blocks {
            debug_assert!(
                block.0 < self.total,
                "BlockPool::free: block ID {} out of range (total={})",
                block.0,
                self.total
            );
            self.free_list.push(block);
        }
    }

    /// Number of currently free blocks.
    pub fn free_count(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// Total number of blocks in the pool (free + allocated).
    pub fn total_count(&self) -> u32 {
        self.total
    }

    /// Number of currently allocated (in-use) blocks.
    pub fn used_count(&self) -> u32 {
        self.total - self.free_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_alloc_free() {
        let mut pool = BlockPool::new(10);
        assert_eq!(pool.free_count(), 10);
        assert_eq!(pool.total_count(), 10);
        assert_eq!(pool.used_count(), 0);

        let blocks = pool.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(pool.free_count(), 7);
        assert_eq!(pool.used_count(), 3);

        pool.free(&blocks);
        assert_eq!(pool.free_count(), 10);
        assert_eq!(pool.used_count(), 0);
    }

    #[test]
    fn allocate_zero() {
        let mut pool = BlockPool::new(5);
        let blocks = pool.allocate(0).unwrap();
        assert!(blocks.is_empty());
        assert_eq!(pool.free_count(), 5);
    }

    #[test]
    fn allocate_all() {
        let mut pool = BlockPool::new(4);
        let blocks = pool.allocate(4).unwrap();
        assert_eq!(blocks.len(), 4);
        assert_eq!(pool.free_count(), 0);

        // Verify all block IDs are distinct and in range
        let mut ids: Vec<u32> = blocks.iter().map(|b| b.0).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn allocate_exhausted() {
        let mut pool = BlockPool::new(3);
        let _b = pool.allocate(2).unwrap();
        assert_eq!(pool.free_count(), 1);

        // Requesting 2 more should fail — only 1 free
        assert!(pool.allocate(2).is_none());
        // Pool state unchanged after failed allocation
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn free_then_reuse() {
        let mut pool = BlockPool::new(2);
        let b1 = pool.allocate(2).unwrap();
        assert!(pool.allocate(1).is_none());

        pool.free(&b1);
        let b2 = pool.allocate(2).unwrap();
        assert_eq!(b2.len(), 2);
    }

    #[test]
    fn incremental_alloc() {
        let mut pool = BlockPool::new(10);
        let a = pool.allocate(3).unwrap();
        let b = pool.allocate(4).unwrap();
        assert_eq!(pool.free_count(), 3);

        pool.free(&a);
        assert_eq!(pool.free_count(), 6);

        pool.free(&b);
        assert_eq!(pool.free_count(), 10);
    }

    #[test]
    fn empty_pool() {
        let mut pool = BlockPool::new(0);
        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.total_count(), 0);
        assert!(pool.allocate(1).is_none());
        assert!(pool.allocate(0).unwrap().is_empty());
    }
}
