use crate::block_hash::{self, BlockHash};
use crate::block_pool::BlockPool;
use crate::types::{BlockId, KvCacheConfig, SeqId};
use std::collections::HashMap;

// ── RadixNode ────────────────────────────────────────────────────────

/// A node in the radix tree. Each node corresponds to one KV cache block
/// holding `block_size` tokens of key/value data.
///
/// Children are stored in a sorted `Vec` keyed by first token, which is
/// cache-friendly and fast for the typical 1–4 children per node.
struct RadixNode {
    /// Token content of this block.
    tokens: Vec<u32>,
    /// Physical block ID on the device.
    block_id: BlockId,
    /// SHA-256 hash of the full prefix up to and including this block.
    /// Used as the universal key for cross-instance and tiered storage lookup.
    hash: BlockHash,
    /// Children keyed on first token of the next block — sorted by key.
    children: Vec<(u32, Box<RadixNode>)>,
    /// Number of active sequences referencing this node.
    ref_count: u32,
    /// Monotonic timestamp of last access (for LRU eviction).
    last_used: u64,
}

impl RadixNode {
    fn find_child(&self, first_token: u32) -> Option<&RadixNode> {
        self.children
            .iter()
            .find(|(tok, _)| *tok == first_token)
            .map(|(_, node)| node.as_ref())
    }

    fn find_child_mut(&mut self, first_token: u32) -> Option<&mut RadixNode> {
        self.children
            .iter_mut()
            .find(|(tok, _)| *tok == first_token)
            .map(|(_, node)| node.as_mut())
    }

    fn insert_child(&mut self, first_token: u32, node: Box<RadixNode>) {
        let pos = self.children.partition_point(|(t, _)| *t < first_token);
        self.children.insert(pos, (first_token, node));
    }

    fn remove_child(&mut self, first_token: u32) -> Option<Box<RadixNode>> {
        if let Some(pos) = self.children.iter().position(|(t, _)| *t == first_token) {
            Some(self.children.remove(pos).1)
        } else {
            None
        }
    }

    /// Is this node a leaf (no children)?
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Can this node be evicted? (leaf with no active references)
    fn is_evictable(&self) -> bool {
        self.is_leaf() && self.ref_count == 0
    }
}

// ── SeqState ─────────────────────────────────────────────────────────

/// Tracks per-sequence KV cache state within the radix tree.
#[derive(Debug)]
struct SeqState {
    /// Block IDs currently held by this sequence (in order).
    block_ids: Vec<BlockId>,
    /// How many leading blocks came from prefix cache (find_prefix).
    /// These blocks had their ref_count incremented and must be
    /// decremented on release. The rest are owned exclusively.
    num_cached_blocks: usize,
    /// Total number of tokens stored (including partial last block).
    token_count: usize,
}

// ── RadixKvManager ───────────────────────────────────────────────────

/// KV cache manager with radix-tree prefix caching.
///
/// Multiple sequences sharing the same token prefix will share the same
/// physical KV cache blocks, avoiding redundant computation. This is the
/// core mechanism behind SGLang's RadixAttention and vLLM's Automatic
/// Prefix Caching (APC).
///
/// # Lifecycle
///
/// 1. **Prefill**: `match_and_allocate(seq_id, tokens)` — finds cached prefix,
///    allocates new blocks for the rest, returns how many tokens are cached.
/// 2. **Decode**: `append_token(seq_id)` — extends by one token, may allocate
///    a new block if the last one is full.
/// 3. **Insert**: `insert_computed_blocks(seq_id, tokens)` — after prefill
///    computation, register the newly computed blocks in the tree for future
///    sharing.
/// 4. **Release**: `release_seq(seq_id)` — decrements ref_counts on all
///    nodes; blocks become eviction candidates when ref_count reaches 0.
/// 5. **Evict**: `evict(target)` — frees `target` blocks by removing
///    unreferenced leaf nodes (LRU order).
pub struct RadixKvManager {
    /// Config (block_size etc.)
    config: KvCacheConfig,
    /// The tree root (virtual node, no block).
    root: RadixNode,
    /// Block pool for GPU memory.
    pool: BlockPool,
    /// Per-sequence tracking.
    sequences: HashMap<SeqId, SeqState>,
    /// Monotonic clock for LRU timestamps.
    clock: u64,
}

impl RadixKvManager {
    /// Create a new manager with the given config and block budget.
    pub fn new(config: KvCacheConfig, num_blocks: u32) -> Self {
        Self {
            config,
            root: RadixNode {
                tokens: Vec::new(),
                block_id: BlockId::INVALID,
                hash: [0u8; 32],
                children: Vec::new(),
                ref_count: 0,
                last_used: 0,
            },
            pool: BlockPool::new(num_blocks),
            sequences: HashMap::new(),
            clock: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    // ── Capacity queries ─────────────────────────────────────────

    /// Number of free blocks available for allocation.
    pub fn free_blocks(&self) -> u32 {
        self.pool.free_count()
    }

    /// Total blocks in the pool.
    pub fn total_blocks(&self) -> u32 {
        self.pool.total_count()
    }

    /// Check if `blocks_needed` blocks can be allocated.
    pub fn can_allocate(&self, blocks_needed: u32) -> bool {
        self.pool.free_count() >= blocks_needed
    }

    /// Number of active sequences.
    pub fn active_seq_count(&self) -> usize {
        self.sequences.len()
    }

    // ── Prefix matching ──────────────────────────────────────────

    /// Walk the tree and find the longest matching prefix.
    ///
    /// Returns `(num_cached_tokens, cached_block_ids)`.
    /// Increments `ref_count` on all matched nodes.
    fn find_prefix(&mut self, tokens: &[u32]) -> (usize, Vec<BlockId>) {
        let now = self.tick();
        let block_size = self.config.block_size;
        let mut node = &mut self.root;
        let mut cached = 0;
        let mut block_ids = Vec::new();

        for chunk in tokens.chunks(block_size) {
            if chunk.len() < block_size {
                break; // partial block — can't match
            }
            let first = chunk[0];
            let matched = node
                .find_child_mut(first)
                .filter(|child| child.tokens == chunk);
            match matched {
                Some(child) => {
                    child.ref_count += 1;
                    child.last_used = now;
                    block_ids.push(child.block_id);
                    cached += block_size;
                    node = child;
                }
                None => break,
            }
        }
        (cached, block_ids)
    }

    /// Decrement ref_count on specific nodes by walking the tree to find them.
    fn release_blocks(&mut self, block_ids: &[BlockId]) {
        for &bid in block_ids {
            Self::release_recursive(&mut self.root, bid);
        }
    }

    /// Recursively search for a node with given block_id and decrement ref_count.
    fn release_recursive(node: &mut RadixNode, target: BlockId) -> bool {
        for (_key, child) in node.children.iter_mut() {
            if child.block_id == target {
                debug_assert!(child.ref_count > 0, "ref_count underflow");
                child.ref_count -= 1;
                return true;
            }
            if Self::release_recursive(child, target) {
                return true;
            }
        }
        false
    }

    // ── Prefill: match + allocate ────────────────────────────────

    /// Match cached prefix and allocate blocks for the remaining tokens.
    ///
    /// Returns `Some(cached_token_count)` on success — the caller should
    /// only compute the forward pass for `tokens[cached_token_count..]`.
    ///
    /// Returns `None` if there are not enough free blocks.
    pub fn match_and_allocate(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
    ) -> Option<usize> {
        debug_assert!(
            !self.sequences.contains_key(&seq_id),
            "Sequence {:?} already exists",
            seq_id
        );

        let (cached_tokens, cached_blocks) = self.find_prefix(tokens);
        let remaining_tokens = tokens.len() - cached_tokens;
        let new_blocks_needed = self.config.blocks_for_tokens(remaining_tokens) as u32;

        match self.pool.allocate(new_blocks_needed) {
            Some(new_blocks) => {
                let num_cached_blocks = cached_blocks.len();
                let mut all_blocks = cached_blocks;
                all_blocks.extend_from_slice(&new_blocks);
                self.sequences.insert(
                    seq_id,
                    SeqState {
                        block_ids: all_blocks,
                        num_cached_blocks,
                        token_count: tokens.len(),
                    },
                );
                Some(cached_tokens)
            }
            None => {
                // Undo ref_count increments from find_prefix
                self.release_blocks(&cached_blocks);
                None
            }
        }
    }

    // ── Post-prefill: register computed blocks in tree ────────────

    /// After computing the forward pass for `tokens`, register the
    /// NEW blocks (those not already in the tree) so future sequences
    /// with the same prefix can reuse them.
    ///
    /// `tokens` is the full token sequence; `cached_tokens` is the count
    /// returned by `match_and_allocate`.
    pub fn insert_computed_blocks(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        cached_tokens: usize,
    ) {
        let now = self.tick();
        let block_size = self.config.block_size;

        let state = match self.sequences.get(&seq_id) {
            Some(s) => s,
            None => return,
        };

        let skip_blocks = cached_tokens / block_size;
        let mut node = &mut self.root;

        // Navigate to the end of the cached prefix
        for chunk in tokens[..cached_tokens].chunks(block_size) {
            if chunk.len() < block_size {
                break;
            }
            let first = chunk[0];
            node = node.find_child_mut(first).unwrap();
        }

        // Insert new nodes for the newly computed blocks
        for (i, chunk) in tokens[cached_tokens..].chunks(block_size).enumerate() {
            if chunk.len() < block_size {
                break; // partial last block — don't insert
            }
            let block_idx = skip_blocks + i;
            let block_id = state.block_ids[block_idx];
            let first = chunk[0];

            // Compute the hash for this block (all tokens up to end of this block)
            let prefix_end = cached_tokens + (i + 1) * block_size;
            let hash = block_hash::compute_block_hash(&tokens[..prefix_end]);

            if node.find_child(first).is_some() {
                // Already exists (race: another seq inserted it)
                node = node.find_child_mut(first).unwrap();
                node.last_used = now;
            } else {
                let new_node = Box::new(RadixNode {
                    tokens: chunk.to_vec(),
                    block_id,
                    hash,
                    children: Vec::new(),
                    ref_count: 1, // the inserting seq is using this block
                    last_used: now,
                });
                node.insert_child(first, new_node);
                node = node.find_child_mut(first).unwrap();
            }
        }
    }

    // ── Decode: append one token ─────────────────────────────────

    /// Check if seq can accept one more token (has room or pool is not empty).
    pub fn can_append(&self, seq_id: SeqId) -> bool {
        match self.sequences.get(&seq_id) {
            None => false,
            Some(state) => {
                let pos = self.config.token_position(state.token_count);
                if pos.offset > 0 || state.token_count == 0 {
                    true
                } else {
                    self.pool.free_count() > 0
                }
            }
        }
    }

    /// Append one generated token. May allocate a new block.
    /// Returns the BlockId where this token lands, or None if no blocks free.
    pub fn append_token(&mut self, seq_id: SeqId) -> Option<BlockId> {
        let state = self.sequences.get_mut(&seq_id)?;
        let pos = self.config.token_position(state.token_count);

        if pos.offset == 0 && state.token_count > 0 {
            // Need a new block
            match self.pool.allocate(1) {
                Some(new_blocks) => {
                    state.block_ids.push(new_blocks[0]);
                }
                None => return None,
            }
        }

        state.token_count += 1;
        let final_pos = self.config.token_position(state.token_count - 1);
        Some(state.block_ids[final_pos.block_idx])
    }

    // ── Release ──────────────────────────────────────────────────

    /// Release a sequence: decrement ref_counts on cached prefix blocks,
    /// free non-cached (decode-only) blocks back to the pool.
    ///
    /// Cached blocks (those in the tree) are NOT freed — they remain
    /// available for future prefix sharing. They will be freed by `evict()`
    /// when their ref_count reaches 0 and they are the oldest leaves.
    pub fn release_seq(&mut self, seq_id: SeqId) -> u32 {
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => return 0,
        };

        // Decrement ref_count on cached prefix blocks (the ones from find_prefix)
        for &block_id in &state.block_ids[..state.num_cached_blocks] {
            Self::release_recursive(&mut self.root, block_id);
        }

        // Release blocks that are in the tree (from insert_computed_blocks):
        // decrement their ref_count too
        for &block_id in &state.block_ids[state.num_cached_blocks..] {
            if Self::node_exists_in_tree(&self.root, block_id) {
                Self::release_recursive(&mut self.root, block_id);
            } else {
                // Block is not in the tree (decode-generated, partial) — free it
                self.pool.free(&[block_id]);
            }
        }

        // Return count of blocks freed directly (non-tree blocks)
        let non_tree = state.block_ids[state.num_cached_blocks..]
            .iter()
            .filter(|bid| !Self::node_exists_in_tree(&self.root, **bid))
            .count();
        non_tree as u32
    }

    /// Check if a block_id exists as a node in the tree.
    fn node_exists_in_tree(node: &RadixNode, target: BlockId) -> bool {
        for (_key, child) in &node.children {
            if child.block_id == target {
                return true;
            }
            if Self::node_exists_in_tree(child, target) {
                return true;
            }
        }
        false
    }

    // ── Eviction ─────────────────────────────────────────────────

    /// Evict unreferenced leaf nodes (LRU order) to free blocks.
    ///
    /// Returns the number of blocks actually freed.
    pub fn evict(&mut self, target: usize) -> usize {
        let mut freed = 0;
        while freed < target {
            match Self::find_oldest_evictable_leaf(&self.root) {
                Some((parent_key, leaf_key)) => {
                    let block_id =
                        Self::remove_leaf(&mut self.root, parent_key, leaf_key);
                    if let Some(bid) = block_id {
                        self.pool.free(&[bid]);
                        freed += 1;
                    }
                }
                None => break, // no more evictable blocks
            }
        }
        freed
    }

    /// Find the evictable leaf with the oldest `last_used` timestamp.
    /// Returns `(parent_first_token_path, leaf_first_token)` for removal.
    /// Uses (parent_key, leaf_key) to locate the leaf for removal.
    fn find_oldest_evictable_leaf(root: &RadixNode) -> Option<(Vec<u32>, u32)> {
        let mut best: Option<(Vec<u32>, u32, u64)> = None; // (path, leaf_key, last_used)
        Self::scan_leaves(root, &mut Vec::new(), &mut best);
        best.map(|(path, key, _)| (path, key))
    }

    fn scan_leaves(
        node: &RadixNode,
        path: &mut Vec<u32>,
        best: &mut Option<(Vec<u32>, u32, u64)>,
    ) {
        for (key, child) in &node.children {
            if child.is_evictable() {
                match best {
                    Some((_, _, best_ts)) if child.last_used >= *best_ts => {}
                    _ => *best = Some((path.clone(), *key, child.last_used)),
                }
            } else {
                path.push(*key);
                Self::scan_leaves(child, path, best);
                path.pop();
            }
        }
    }

    /// Remove a leaf node identified by `(parent_path, leaf_key)`.
    fn remove_leaf(
        root: &mut RadixNode,
        parent_path: Vec<u32>,
        leaf_key: u32,
    ) -> Option<BlockId> {
        let mut node = root;
        for &key in &parent_path {
            node = node.find_child_mut(key)?;
        }
        let removed = node.remove_child(leaf_key)?;
        Some(removed.block_id)
    }

    // ── Query ────────────────────────────────────────────────────

    /// Get the block IDs for a sequence.
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

    /// Build the `block_table` for a batch of sequences (for the attention kernel).
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

    /// Probe how many tokens of a prompt are cached WITHOUT committing.
    /// Does NOT modify ref_counts.
    pub fn probe_prefix(&self, tokens: &[u32]) -> usize {
        let block_size = self.config.block_size;
        let mut node = &self.root;
        let mut cached = 0;

        for chunk in tokens.chunks(block_size) {
            if chunk.len() < block_size {
                break;
            }
            let first = chunk[0];
            match node.find_child(first).filter(|child| child.tokens == chunk) {
                Some(child) => {
                    cached += block_size;
                    node = child;
                }
                None => break,
            }
        }
        cached
    }

    /// Count total nodes in the tree (for debugging/metrics).
    pub fn tree_node_count(&self) -> usize {
        Self::count_nodes(&self.root)
    }

    fn count_nodes(node: &RadixNode) -> usize {
        let mut count = node.children.len();
        for (_, child) in &node.children {
            count += Self::count_nodes(child);
        }
        count
    }

    // ── Hash-based APIs (for KvConnector / cross-instance) ───────

    /// Get the hashes of all tree nodes currently cached for a token sequence.
    ///
    /// Walks the tree matching the prefix, collecting (BlockId, BlockHash) pairs.
    /// Does NOT modify ref_counts.
    pub fn get_prefix_hashes(&self, tokens: &[u32]) -> Vec<(BlockId, BlockHash)> {
        let block_size = self.config.block_size;
        let mut node = &self.root;
        let mut result = Vec::new();

        for chunk in tokens.chunks(block_size) {
            if chunk.len() < block_size {
                break;
            }
            let first = chunk[0];
            match node.find_child(first).filter(|child| child.tokens == chunk) {
                Some(child) => {
                    result.push((child.block_id, child.hash));
                    node = child;
                }
                None => break,
            }
        }
        result
    }

    /// Check if any tree node has the given hash.
    pub fn contains_hash(&self, target: &BlockHash) -> bool {
        Self::search_hash(&self.root, target)
    }

    fn search_hash(node: &RadixNode, target: &BlockHash) -> bool {
        for (_, child) in &node.children {
            if &child.hash == target {
                return true;
            }
            if Self::search_hash(child, target) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg4() -> KvCacheConfig {
        KvCacheConfig {
            block_size: 4, // small for easy testing
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            dtype_size: 2,
        }
    }

    // Helper: create tokens [start, start+1, ..., start+len-1]
    fn tokens(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    #[test]
    fn basic_prefill_and_release() {
        let mut mgr = RadixKvManager::new(cfg4(), 10);
        let seq = SeqId(1);

        // Prefill 8 tokens → 2 full blocks, 0 cached
        let cached = mgr.match_and_allocate(seq, &tokens(0, 8)).unwrap();
        assert_eq!(cached, 0);
        assert_eq!(mgr.seq_token_count(seq), 8);
        assert_eq!(mgr.get_block_ids(seq).unwrap().len(), 2);
        assert_eq!(mgr.free_blocks(), 8);

        // Insert computed blocks into tree
        mgr.insert_computed_blocks(seq, &tokens(0, 8), 0);
        assert_eq!(mgr.tree_node_count(), 2);

        // Release
        mgr.release_seq(seq);
        // Blocks stay in tree (still 8 free — but tree blocks are not freed!)
        assert_eq!(mgr.tree_node_count(), 2);
        assert_eq!(mgr.active_seq_count(), 0);
    }

    #[test]
    fn prefix_sharing_between_sequences() {
        let mut mgr = RadixKvManager::new(cfg4(), 20);

        // Seq 1: tokens [0..8]
        let cached1 = mgr
            .match_and_allocate(SeqId(1), &tokens(0, 8))
            .unwrap();
        assert_eq!(cached1, 0); // nothing cached yet
        mgr.insert_computed_blocks(SeqId(1), &tokens(0, 8), 0);
        assert_eq!(mgr.free_blocks(), 18); // 2 blocks used

        // Seq 2: tokens [0..12] — first 8 should match Seq 1's prefix
        let cached2 = mgr
            .match_and_allocate(SeqId(2), &tokens(0, 12))
            .unwrap();
        assert_eq!(cached2, 8); // 8 tokens cached from Seq 1!
        assert_eq!(mgr.free_blocks(), 17); // only 1 new block for tokens [8..12]

        // Seq 2 shares Seq 1's 2 blocks + 1 new block = 3 total
        assert_eq!(mgr.get_block_ids(SeqId(2)).unwrap().len(), 3);

        // The first 2 block IDs should be the SAME as Seq 1's
        let ids1 = mgr.get_block_ids(SeqId(1)).unwrap();
        let ids2 = mgr.get_block_ids(SeqId(2)).unwrap();
        assert_eq!(ids1[0], ids2[0]);
        assert_eq!(ids1[1], ids2[1]);
    }

    #[test]
    fn probe_prefix_is_read_only() {
        let mut mgr = RadixKvManager::new(cfg4(), 10);

        // Insert some blocks via seq 1
        mgr.match_and_allocate(SeqId(1), &tokens(0, 8)).unwrap();
        mgr.insert_computed_blocks(SeqId(1), &tokens(0, 8), 0);
        mgr.release_seq(SeqId(1));

        // Probe — should find 8 cached tokens but NOT modify ref_counts
        let cached = mgr.probe_prefix(&tokens(0, 12));
        assert_eq!(cached, 8);

        // Tree nodes should still have ref_count == 0 (evictable)
        assert_eq!(mgr.evict(2), 2);
    }

    #[test]
    fn eviction_frees_unreferenced_leaves() {
        let mut mgr = RadixKvManager::new(cfg4(), 10);

        // Insert and release — leaves become evictable
        mgr.match_and_allocate(SeqId(1), &tokens(0, 8)).unwrap();
        mgr.insert_computed_blocks(SeqId(1), &tokens(0, 8), 0);
        mgr.release_seq(SeqId(1));

        assert_eq!(mgr.free_blocks(), 8);
        assert_eq!(mgr.tree_node_count(), 2);

        // Evict 1 block (leaf = the second block)
        let evicted = mgr.evict(1);
        assert_eq!(evicted, 1);
        assert_eq!(mgr.free_blocks(), 9);
        assert_eq!(mgr.tree_node_count(), 1);

        // Evict 1 more (now the first block is a leaf)
        let evicted = mgr.evict(1);
        assert_eq!(evicted, 1);
        assert_eq!(mgr.free_blocks(), 10);
        assert_eq!(mgr.tree_node_count(), 0);
    }

    #[test]
    fn eviction_skips_referenced_nodes() {
        let mut mgr = RadixKvManager::new(cfg4(), 10);

        // Seq 1 holds a reference to blocks
        mgr.match_and_allocate(SeqId(1), &tokens(0, 8)).unwrap();
        mgr.insert_computed_blocks(SeqId(1), &tokens(0, 8), 0);

        // Try to evict — should fail because seq 1 still references them
        let evicted = mgr.evict(2);
        assert_eq!(evicted, 0);

        // Release seq 1 → now evictable
        mgr.release_seq(SeqId(1));
        let evicted = mgr.evict(2);
        assert_eq!(evicted, 2);
    }

    #[test]
    fn eviction_lru_order() {
        let mut mgr = RadixKvManager::new(cfg4(), 10);

        // Insert two separate prefixes
        // Prefix A: [0..4] (1 block)
        mgr.match_and_allocate(SeqId(1), &tokens(0, 4)).unwrap();
        mgr.insert_computed_blocks(SeqId(1), &tokens(0, 4), 0);
        mgr.release_seq(SeqId(1)); // last_used = early timestamp

        // Prefix B: [100..104] (1 block)
        mgr.match_and_allocate(SeqId(2), &tokens(100, 4)).unwrap();
        mgr.insert_computed_blocks(SeqId(2), &tokens(100, 4), 0);
        mgr.release_seq(SeqId(2)); // last_used = later timestamp

        // Evict 1 → should evict prefix A (older)
        mgr.evict(1);

        // Prefix A should be gone, B should remain
        assert_eq!(mgr.probe_prefix(&tokens(0, 4)), 0); // A evicted
        assert_eq!(mgr.probe_prefix(&tokens(100, 4)), 4); // B still here
    }

    #[test]
    fn decode_tokens_across_block_boundary() {
        let mut mgr = RadixKvManager::new(cfg4(), 10);
        let seq = SeqId(1);

        // Prefill 3 tokens → 1 block (offset = 3, 1 slot free)
        mgr.match_and_allocate(seq, &tokens(0, 3)).unwrap();
        assert_eq!(mgr.free_blocks(), 9);

        // Append 1 token → fills block (offset 3 → now at 4 tokens, block full)
        assert!(mgr.append_token(seq).is_some());
        assert_eq!(mgr.seq_token_count(seq), 4);
        assert_eq!(mgr.get_block_ids(seq).unwrap().len(), 1);
        assert_eq!(mgr.free_blocks(), 9);

        // Append 1 more → needs new block
        assert!(mgr.append_token(seq).is_some());
        assert_eq!(mgr.seq_token_count(seq), 5);
        assert_eq!(mgr.get_block_ids(seq).unwrap().len(), 2);
        assert_eq!(mgr.free_blocks(), 8);
    }

    #[test]
    fn allocation_fails_with_insufficient_blocks() {
        let mut mgr = RadixKvManager::new(cfg4(), 2);

        // 12 tokens → needs 3 blocks, only 2 available
        let result = mgr.match_and_allocate(SeqId(1), &tokens(0, 12));
        assert!(result.is_none());
        assert_eq!(mgr.active_seq_count(), 0);
        assert_eq!(mgr.free_blocks(), 2); // nothing leaked
    }

    #[test]
    fn partial_prefix_match() {
        let mut mgr = RadixKvManager::new(cfg4(), 10);

        // Seq 1: tokens [0..8] (2 blocks)
        mgr.match_and_allocate(SeqId(1), &tokens(0, 8)).unwrap();
        mgr.insert_computed_blocks(SeqId(1), &tokens(0, 8), 0);
        mgr.release_seq(SeqId(1));

        // Seq 2: same first block [0..4], different second block [4,5,6,99]
        let mut divergent = tokens(0, 4);
        divergent.extend_from_slice(&[4, 5, 6, 99]); // differs at position 7
        let cached = mgr
            .match_and_allocate(SeqId(2), &divergent)
            .unwrap();
        assert_eq!(cached, 4); // only first block matches
        assert_eq!(mgr.get_block_ids(SeqId(2)).unwrap().len(), 2);
    }

    #[test]
    fn block_table_output() {
        let mut mgr = RadixKvManager::new(cfg4(), 10);
        mgr.match_and_allocate(SeqId(1), &tokens(0, 8)).unwrap();
        mgr.match_and_allocate(SeqId(2), &tokens(100, 4)).unwrap();

        let table = mgr.build_block_table(&[SeqId(1), SeqId(2)]);
        assert_eq!(table.len(), 2);
        assert_eq!(table[0].len(), 2); // 8 tokens / block_size 4 = 2
        assert_eq!(table[1].len(), 1); // 4 tokens = 1
    }

    #[test]
    fn three_way_prefix_sharing() {
        let mut mgr = RadixKvManager::new(cfg4(), 20);

        // System prompt: tokens [0..8]
        let system = tokens(0, 8);

        // Seq 1: system + user1
        let mut prompt1 = system.clone();
        prompt1.extend(tokens(100, 4));
        let c1 = mgr.match_and_allocate(SeqId(1), &prompt1).unwrap();
        assert_eq!(c1, 0);
        mgr.insert_computed_blocks(SeqId(1), &prompt1, 0);

        // Seq 2: system + user2
        let mut prompt2 = system.clone();
        prompt2.extend(tokens(200, 4));
        let c2 = mgr.match_and_allocate(SeqId(2), &prompt2).unwrap();
        assert_eq!(c2, 8); // system prompt cached!
        mgr.insert_computed_blocks(SeqId(2), &prompt2, 8);

        // Seq 3: system + user1 (same as seq 1)
        let c3 = mgr.match_and_allocate(SeqId(3), &prompt1).unwrap();
        assert_eq!(c3, 12); // full 12-token prefix cached from seq 1!

        // All three share the system prompt blocks
        let ids1 = mgr.get_block_ids(SeqId(1)).unwrap();
        let ids2 = mgr.get_block_ids(SeqId(2)).unwrap();
        let ids3 = mgr.get_block_ids(SeqId(3)).unwrap();
        assert_eq!(ids1[0], ids2[0]); // system block 0 shared
        assert_eq!(ids1[1], ids2[1]); // system block 1 shared
        assert_eq!(ids1[0], ids3[0]);
        assert_eq!(ids1[1], ids3[1]);
        assert_eq!(ids1[2], ids3[2]); // user1 block also shared with seq 3
    }

    #[test]
    fn hash_integration() {
        let mut mgr = RadixKvManager::new(cfg4(), 20);
        let prompt = tokens(0, 12); // 3 blocks

        // Insert blocks
        mgr.match_and_allocate(SeqId(1), &prompt).unwrap();
        mgr.insert_computed_blocks(SeqId(1), &prompt, 0);

        // Get hashes from tree
        let prefix_hashes = mgr.get_prefix_hashes(&prompt);
        assert_eq!(prefix_hashes.len(), 3);

        // Verify hashes match independently computed ones
        let expected = block_hash::compute_all_block_hashes(&prompt, 4);
        assert_eq!(prefix_hashes[0].1, expected[0]);
        assert_eq!(prefix_hashes[1].1, expected[1]);
        assert_eq!(prefix_hashes[2].1, expected[2]);

        // contains_hash should find them
        assert!(mgr.contains_hash(&expected[0]));
        assert!(mgr.contains_hash(&expected[1]));
        assert!(mgr.contains_hash(&expected[2]));

        // A hash for different tokens should NOT be found
        let bogus = block_hash::compute_block_hash(&[99, 99, 99, 99]);
        assert!(!mgr.contains_hash(&bogus));
    }
}
