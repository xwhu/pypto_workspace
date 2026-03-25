//! Block hash computation for KV cache block identification.
//!
//! Each KV block is uniquely identified by the SHA-256 hash of **all tokens**
//! in the prefix up to and including that block's tokens. This hash serves as
//! the universal address across all tiers and instances, enabling:
//!
//! - **Cross-instance cache sharing**: Instance B can reuse blocks computed by
//!   Instance A by looking up the same hash in a shared store.
//! - **Deduplication**: Identical prefixes produce identical hashes regardless
//!   of which sequence or instance computed them.
//! - **Tiered lookup**: The hash is the key in CPU DRAM, SSD, and remote
//!   storage tiers (v5 §3 KvConnector interface).
//!
//! # Hash computation
//!
//! For a prompt `[t₀, t₁, ..., tₙ]` with `block_size = 4`:
//!
//! - Block 0 hash = SHA-256(`[t₀, t₁, t₂, t₃]`)
//! - Block 1 hash = SHA-256(`[t₀, t₁, t₂, t₃, t₄, t₅, t₆, t₇]`)
//! - Block k hash = SHA-256(`[t₀, ..., t_{(k+1)×block_size - 1}]`)
//!
//! The hash includes the FULL prefix, not just the block's own tokens.
//! This ensures that block identity depends on context, not just content.

use sha2::{Digest, Sha256};

/// SHA-256 hash identifying a KV cache block.
///
/// Two blocks with the same hash contain identical KV data and are
/// interchangeable. This is the universal key for block lookup across
/// all storage tiers and instances.
pub type BlockHash = [u8; 32];

/// Compute the hash for a single block, given the full token prefix
/// up to and including that block.
///
/// `prefix_tokens` must contain all tokens from the beginning of the
/// sequence up to and including the last token in this block.
pub fn compute_block_hash(prefix_tokens: &[u32]) -> BlockHash {
    let mut hasher = Sha256::new();
    // Hash the raw bytes of the token array.
    // We use little-endian encoding for cross-platform consistency.
    for &token in prefix_tokens {
        hasher.update(token.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Compute hashes for all complete blocks in a token sequence.
///
/// Returns a `Vec<BlockHash>` where `result[i]` is the hash of block `i`.
/// Only full blocks are included — a trailing partial block is ignored.
///
/// # Example
///
/// ```rust
/// use kv_cache::block_hash::compute_all_block_hashes;
///
/// let tokens: Vec<u32> = (0..20).collect();
/// let hashes = compute_all_block_hashes(&tokens, 4);
/// assert_eq!(hashes.len(), 5); // 20 / 4 = 5 full blocks
///
/// // Each hash covers all tokens up to that block
/// let tokens_partial: Vec<u32> = (0..22).collect();
/// let hashes_partial = compute_all_block_hashes(&tokens_partial, 4);
/// assert_eq!(hashes_partial.len(), 5); // 22 / 4 = 5 full + 2 leftover
///
/// // First 5 hashes are identical regardless of trailing tokens
/// assert_eq!(hashes, hashes_partial);
/// ```
pub fn compute_all_block_hashes(tokens: &[u32], block_size: usize) -> Vec<BlockHash> {
    let num_full_blocks = tokens.len() / block_size;
    let mut hashes = Vec::with_capacity(num_full_blocks);

    for i in 0..num_full_blocks {
        let prefix_end = (i + 1) * block_size;
        hashes.push(compute_block_hash(&tokens[..prefix_end]));
    }

    hashes
}

/// Incrementally compute block hashes, reusing a hasher state for efficiency.
///
/// This is faster than `compute_all_block_hashes` for long sequences because
/// it builds on the previous block's hash state rather than rehashing the
/// entire prefix for every block.
///
/// Note: produces the same hashes as `compute_all_block_hashes`.
pub fn compute_all_block_hashes_incremental(
    tokens: &[u32],
    block_size: usize,
) -> Vec<BlockHash> {
    let num_full_blocks = tokens.len() / block_size;
    let mut hashes = Vec::with_capacity(num_full_blocks);

    // We can't truly incrementally produce the same SHA-256 result as
    // hashing the full prefix (SHA-256 is not prefix-incremental in that
    // way). However, we can avoid re-iterating by building the full
    // prefix hash each time.
    //
    // Optimization for the future: use a Merkle-tree-like structure or
    // a rolling hash for O(1) per-block amortized cost.
    let mut hasher = Sha256::new();
    for (i, &token) in tokens.iter().enumerate() {
        hasher.update(token.to_le_bytes());
        if (i + 1) % block_size == 0 && (i + 1) / block_size <= num_full_blocks {
            // Clone the hasher state — this is much cheaper than rehashing
            // from scratch because Sha256::clone copies ~100 bytes of state
            // vs re-processing all prior tokens.
            let hash: BlockHash = hasher.clone().finalize().into();
            hashes.push(hash);
        }
    }

    hashes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    #[test]
    fn hash_deterministic() {
        let t = tokens(0, 16);
        let h1 = compute_block_hash(&t);
        let h2 = compute_block_hash(&t);
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_tokens_different_hash() {
        let h1 = compute_block_hash(&tokens(0, 4));
        let h2 = compute_block_hash(&tokens(1, 4));
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_is_prefix_dependent() {
        // Block hash for tokens [0,1,2,3] alone vs as part of longer prefix
        let short = compute_block_hash(&[0, 1, 2, 3]);
        let long = compute_block_hash(&[99, 99, 99, 99, 0, 1, 2, 3]);
        // These SHOULD be different — block identity depends on full prefix
        assert_ne!(short, long);
    }

    #[test]
    fn compute_all_hashes() {
        let t = tokens(0, 8);
        let hashes = compute_all_block_hashes(&t, 4);
        assert_eq!(hashes.len(), 2);

        // Verify manually
        assert_eq!(hashes[0], compute_block_hash(&t[..4]));
        assert_eq!(hashes[1], compute_block_hash(&t[..8]));
    }

    #[test]
    fn compute_all_hashes_with_partial_trailing() {
        let t = tokens(0, 10); // 2 full blocks + 2 leftover
        let hashes = compute_all_block_hashes(&t, 4);
        assert_eq!(hashes.len(), 2); // only full blocks
    }

    #[test]
    fn compute_all_hashes_empty() {
        let hashes = compute_all_block_hashes(&[], 4);
        assert!(hashes.is_empty());
    }

    #[test]
    fn incremental_matches_non_incremental() {
        let t = tokens(0, 100);
        let h1 = compute_all_block_hashes(&t, 16);
        let h2 = compute_all_block_hashes_incremental(&t, 16);
        assert_eq!(h1, h2);
    }

    #[test]
    fn incremental_matches_various_sizes() {
        for block_size in [1, 2, 4, 8, 16] {
            let t = tokens(0, 64);
            let h1 = compute_all_block_hashes(&t, block_size);
            let h2 = compute_all_block_hashes_incremental(&t, block_size);
            assert_eq!(h1, h2, "mismatch for block_size={block_size}");
        }
    }

    #[test]
    fn shared_prefix_produces_shared_hashes() {
        // Simulate system prompt sharing
        let system = tokens(0, 8); // 2 blocks

        let mut prompt1 = system.clone();
        prompt1.extend(tokens(100, 4)); // +1 block

        let mut prompt2 = system.clone();
        prompt2.extend(tokens(200, 4)); // +1 different block

        let h1 = compute_all_block_hashes(&prompt1, 4);
        let h2 = compute_all_block_hashes(&prompt2, 4);

        // System prompt blocks have identical hashes
        assert_eq!(h1[0], h2[0]); // block 0
        assert_eq!(h1[1], h2[1]); // block 1

        // User content blocks differ
        assert_ne!(h1[2], h2[2]); // block 2
    }

    #[test]
    fn hash_is_32_bytes() {
        let h = compute_block_hash(&[1, 2, 3, 4]);
        assert_eq!(h.len(), 32);
    }
}
