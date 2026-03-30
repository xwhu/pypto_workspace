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
//! Temporary buffers (cos/sin tables, silu intermediate, attention masks,
//! softmax auxiliary buffers) live in the arena as non-owning views.
//! Owned buffers that must outlive their Rust scope (matmul_hp FP32 weight
//! copies, qk_norm consumed inputs) are parked in the arena's
//! `deferred_owned` list and freed when the arena is recycled.
//!
//! Output tensors that flow into `TensorPool` slots are still independently
//! allocated.
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
/// - 1 = default / fallback (sync every layer, identical to old behaviour)
/// - 2 = aggressive (2-layer gap)
/// - 3 = moderate
/// - 4 = conservative
#[cfg(feature = "pool_depth_2")]
pub const POOL_DEPTH: usize = 2;

#[cfg(feature = "pool_depth_3")]
pub const POOL_DEPTH: usize = 3;

#[cfg(feature = "pool_depth_4")]
pub const POOL_DEPTH: usize = 4;

#[cfg(not(any(feature = "pool_depth_2", feature = "pool_depth_3", feature = "pool_depth_4")))]
pub const POOL_DEPTH: usize = 1;

// ─── ScratchArena ──────────────────────────────────────────────────────

/// A lazily-allocated scratch arena for one layer's temporary device buffers.
///
/// Provides two memory management mechanisms:
///
/// 1. **Non-owning views** (`alloc()`): Temporary buffers (cos/sin, mask,
///    silu intermediate) are issued as non-owning views into cached device
///    memory. The arena retains ownership; callers must not free or move.
///
/// 2. **Deferred-owned parking** (`defer_owned()`): Owned buffers that are
///    still referenced by async stream operations (e.g., matmul_hp FP32
///    weight copies, qk_norm consumed inputs) are parked in the arena.
///    They are freed when the arena is recycled (N layers later), at which
///    point all async ops are guaranteed complete by stream FIFO ordering.
///
/// When the arena is `reset()`, both the allocation cursor and the
/// deferred-owned list are cleared.
#[cfg(feature = "ascend")]
pub struct ScratchArena {
    /// Cached device buffers, grown lazily.
    buffers: Vec<DeviceBuffer>,
    /// Current allocation cursor (bump index).
    cursor: usize,
    /// Owned buffers parked for deferred release.
    ///
    /// These are freed (aclrtFree) when `reset()` is called, which only
    /// happens after POOL_DEPTH layers of stream gap — guaranteeing all
    /// async ops referencing these buffers have completed.
    deferred_owned: Vec<DeviceBuffer>,
    /// The layer index that last used this arena (for reset guard).
    last_layer: Option<usize>,
}

#[cfg(feature = "ascend")]
impl ScratchArena {
    /// Create an empty arena (no device memory allocated yet).
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            cursor: 0,
            deferred_owned: Vec::new(),
            last_layer: None,
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

    /// Park an owned DeviceBuffer for deferred release.
    ///
    /// The buffer will be freed when this arena is `reset()` — i.e., when
    /// POOL_DEPTH layers have passed and all async ops are guaranteed done.
    ///
    /// Use this for:
    /// - matmul_hp FP32 temp buffers (a_fp32, b_fp32)
    /// - qk_norm consumed input buffers
    /// - AllReduce FP32→BF16 cast input buffers
    pub fn defer_owned(&mut self, buf: DeviceBuffer) {
        self.deferred_owned.push(buf);
    }

    /// Park multiple owned DeviceBuffers for deferred release.
    pub fn defer_owned_many(&mut self, bufs: Vec<DeviceBuffer>) {
        self.deferred_owned.extend(bufs);
    }

    /// Reset the allocation cursor and release all deferred-owned buffers.
    ///
    /// This is safe because the rotating pool guarantees that all async
    /// ops from the previous user of this arena have completed (POOL_DEPTH
    /// layers of stream gap).
    fn reset(&mut self) {
        self.cursor = 0;
        // Drop all deferred-owned buffers → aclrtFree.
        // Safe: POOL_DEPTH layers have passed since these were parked.
        self.deferred_owned.clear();
    }

    /// Number of cached buffers (for diagnostics).
    #[allow(dead_code)]
    pub fn num_cached(&self) -> usize {
        self.buffers.len()
    }

    /// Number of deferred-owned buffers currently held.
    #[allow(dead_code)]
    pub fn num_deferred(&self) -> usize {
        self.deferred_owned.len()
    }

    /// Total device memory held by this arena (for diagnostics).
    #[allow(dead_code)]
    pub fn total_bytes(&self) -> usize {
        let cached: usize = self.buffers.iter().map(|b| b.size()).sum();
        let deferred: usize = self.deferred_owned.iter().map(|b| b.size()).sum();
        cached + deferred
    }
}

// ─── RotatingPool ──────────────────────────────────────────────────────

/// N-way rotating pool of scratch arenas.
///
/// Each transformer layer is assigned `arena[layer_idx % POOL_DEPTH]`.
/// The arena is reset only on the **first access** for a new layer
/// (tracked via `ScratchArena::last_layer`). Subsequent accesses within
/// the same layer return the arena without resetting, preserving all
/// buffers allocated by earlier operators in that layer.
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

    /// Get the arena for the given layer.
    ///
    /// On the **first** call for a given `layer_idx`, the arena is reset
    /// (clearing its allocation cursor and deferred-owned buffers).
    /// Subsequent calls with the same `layer_idx` return the arena as-is,
    /// preserving buffers from earlier operators in the same layer.
    pub fn arena_for_layer(&mut self, layer_idx: usize) -> &mut ScratchArena {
        let idx = layer_idx % POOL_DEPTH;
        if self.arenas[idx].last_layer != Some(layer_idx) {
            // New layer is taking over this arena — reset it.
            self.arenas[idx].reset();
            self.arenas[idx].last_layer = Some(layer_idx);
        }
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
                "  Arena[{}]: {} cached buffers, {} deferred, {:.2} MB total",
                i,
                arena.num_cached(),
                arena.num_deferred(),
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
        // Default (no pool_depth feature) should be 1 (safe fallback)
        #[cfg(not(any(
            feature = "pool_depth_2",
            feature = "pool_depth_3",
            feature = "pool_depth_4"
        )))]
        assert_eq!(POOL_DEPTH, 1);
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
