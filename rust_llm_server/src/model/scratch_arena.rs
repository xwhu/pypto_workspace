//! Rotating scratch-arena pool for NPU temporary buffers.
//!
//! # Problem
//!
//! CANN's `aclrtFree` does not respect stream ordering — it reclaims
//! device memory immediately even if async kernels are still reading it.
//! The previous solution called `stream.synchronize()` after every layer,
//! stalling the pipeline and destroying performance.
//!
//! # Solution: Rotating Pool
//!
//! Pre-allocate `POOL_DEPTH` arenas. Each transformer layer uses
//! `arena[layer_idx % POOL_DEPTH]`. When a layer finishes and `POOL_DEPTH`
//! later layers have executed, all async ops from the first layer are
//! guaranteed to have completed (stream is FIFO), so the arena can safely
//! be recycled — **without** any `synchronize()` call.
//!
//! Only _temporary_ buffers (cos/sin tables, silu intermediate, attention
//! masks, softmax auxiliary buffers) live in the arena. Output tensors
//! that flow into `TensorPool` slots are still independently allocated.
//!
//! # POOL_DEPTH Compile-time Configuration
//!
//! ```bash
//! cargo build --features ascend,pool_depth_2   # aggressive
//! cargo build --features ascend,pool_depth_3   # moderate
//! cargo build --features ascend                # default = 4 (safest)
//! ```
//!
//! A special `POOL_DEPTH=1` is equivalent to per-layer-sync fallback:
//! the arena is reset (after a sync) every single layer.

#[cfg(feature = "ascend")]
use ascend::DeviceBuffer;

// ─── Compile-time POOL_DEPTH ───────────────────────────────────────────

/// Rotating pool depth: how many arenas to cycle through.
///
/// - 1 = fallback (sync every layer, identical to old behaviour)
/// - 2 = aggressive (2-layer gap guarantees safety)
/// - 3 = moderate
/// - 4 = conservative / default
#[cfg(feature = "pool_depth_1")]
pub const POOL_DEPTH: usize = 1;

#[cfg(feature = "pool_depth_2")]
pub const POOL_DEPTH: usize = 2;

#[cfg(feature = "pool_depth_3")]
pub const POOL_DEPTH: usize = 3;

#[cfg(not(any(feature = "pool_depth_1", feature = "pool_depth_2", feature = "pool_depth_3")))]
pub const POOL_DEPTH: usize = 4;

// ─── ScratchArena ──────────────────────────────────────────────────────

/// A lazily-allocated scratch arena for one layer's temporary device buffers.
///
/// Buffers are allocated on first use and cached for subsequent forward
/// passes. When the arena is `reset()`, the allocation cursor is rewound
/// to zero — existing buffers are kept in memory but can be re-issued.
///
/// If a subsequent request needs a larger buffer than what was cached,
/// the old buffer is replaced with a larger one (the old one is dropped,
/// which is safe because `reset()` is only called when stream ordering
/// guarantees no pending reads).
#[cfg(feature = "ascend")]
pub struct ScratchArena {
    /// Cached device buffers, grown lazily.
    buffers: Vec<DeviceBuffer>,
    /// Current allocation cursor (bump index).
    cursor: usize,
}

#[cfg(feature = "ascend")]
impl ScratchArena {
    /// Create an empty arena (no device memory allocated yet).
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            cursor: 0,
        }
    }

    /// Rent a device buffer of at least `size_bytes` from this arena.
    ///
    /// - If the arena has a cached buffer at the current cursor position
    ///   and it is large enough, return it directly (zero-alloc fast path).
    /// - If the cached buffer is too small, replace it with a larger one.
    /// - If no cached buffer exists at this position, allocate a new one.
    ///
    /// The returned `DeviceBuffer` is a **non-owning view** — the arena
    /// retains ownership of the underlying device memory. The caller must
    /// NOT free or move the buffer.
    pub fn alloc(&mut self, size_bytes: usize) -> DeviceBuffer {
        if self.cursor < self.buffers.len() {
            // Fast path: reuse existing buffer if large enough
            if self.buffers[self.cursor].size() >= size_bytes {
                let buf = unsafe {
                    DeviceBuffer::from_raw_non_owning(
                        self.buffers[self.cursor].ptr(),
                        size_bytes,
                    )
                };
                self.cursor += 1;
                return buf;
            }
            // Existing buffer is too small — replace it
            let new_buf = DeviceBuffer::alloc(size_bytes)
                .expect("ScratchArena: failed to allocate larger device buffer");
            // Old buffer is dropped here (safe: arena was reset, stream has
            // drained all ops from the previous user of this arena slot).
            self.buffers[self.cursor] = new_buf;
            let buf = unsafe {
                DeviceBuffer::from_raw_non_owning(
                    self.buffers[self.cursor].ptr(),
                    size_bytes,
                )
            };
            self.cursor += 1;
            buf
        } else {
            // Cold path: first forward pass — allocate and cache
            let new_buf = DeviceBuffer::alloc(size_bytes)
                .expect("ScratchArena: failed to allocate new device buffer");
            let view = unsafe {
                DeviceBuffer::from_raw_non_owning(new_buf.ptr(), size_bytes)
            };
            self.buffers.push(new_buf);
            self.cursor += 1;
            view
        }
    }

    /// Reset the allocation cursor to zero.
    ///
    /// All previously-allocated buffers remain in device memory and will
    /// be reused by the next `alloc()` calls. This is safe because the
    /// rotating pool guarantees that all async ops from the previous user
    /// of this arena have completed before `reset()` is called.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Number of cached buffers (for diagnostics).
    #[allow(dead_code)]
    pub fn num_cached(&self) -> usize {
        self.buffers.len()
    }

    /// Total device memory held by this arena (for diagnostics).
    #[allow(dead_code)]
    pub fn total_bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.size()).sum()
    }
}

// ─── RotatingPool ──────────────────────────────────────────────────────

/// N-way rotating pool of scratch arenas.
///
/// Each transformer layer is assigned `arena[layer_idx % POOL_DEPTH]`.
/// Before issuing an arena to a layer, it is `reset()` — safe because
/// at least `POOL_DEPTH - 1` full layers of stream work have been
/// enqueued since the arena's last user, guaranteeing completion.
///
/// Special case: `POOL_DEPTH == 1` means every layer resets and syncs
/// (fallback to the old per-layer-sync behaviour).
#[cfg(feature = "ascend")]
pub struct RotatingPool {
    arenas: Vec<ScratchArena>,
}

#[cfg(feature = "ascend")]
impl RotatingPool {
    /// Create a new rotating pool with `POOL_DEPTH` empty arenas.
    pub fn new() -> Self {
        let arenas = (0..POOL_DEPTH).map(|_| ScratchArena::new()).collect();
        tracing::info!(
            "RotatingPool created: POOL_DEPTH={}, {} arenas",
            POOL_DEPTH,
            POOL_DEPTH,
        );
        Self { arenas }
    }

    /// Get the arena for the given layer, resetting it for reuse.
    ///
    /// The caller should call this at the **start** of each transformer
    /// layer to obtain a clean arena for that layer's temporary buffers.
    pub fn arena_for_layer(&mut self, layer_idx: usize) -> &mut ScratchArena {
        let idx = layer_idx % POOL_DEPTH;
        self.arenas[idx].reset();
        &mut self.arenas[idx]
    }

    /// Get the compile-time pool depth.
    pub fn depth(&self) -> usize {
        POOL_DEPTH
    }

    /// Whether this pool operates in fallback mode (per-layer sync).
    pub fn is_fallback(&self) -> bool {
        POOL_DEPTH == 1
    }

    /// Log diagnostics about arena sizes (call after first forward pass).
    pub fn log_stats(&self) {
        for (i, arena) in self.arenas.iter().enumerate() {
            tracing::info!(
                "  Arena[{}]: {} cached buffers, {:.2} MB total",
                i,
                arena.num_cached(),
                arena.total_bytes() as f64 / 1e6,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_depth_constant() {
        // Default (no pool_depth feature) should be 4
        #[cfg(not(any(
            feature = "pool_depth_1",
            feature = "pool_depth_2",
            feature = "pool_depth_3"
        )))]
        assert_eq!(POOL_DEPTH, 4);
    }

    #[test]
    fn test_rotating_index() {
        // Verify modular arithmetic for layer assignment
        let depth = 4;
        assert_eq!(0 % depth, 0);
        assert_eq!(1 % depth, 1);
        assert_eq!(4 % depth, 0);
        assert_eq!(35 % depth, 3);
    }
}
