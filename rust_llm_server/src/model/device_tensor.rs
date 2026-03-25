//! Strong-typed device tensor types for RAII-safe device memory management.
//!
//! Three types form a hierarchy:
//! - `TensorMeta`: shape + dtype metadata (compile-time, no device resources)
//! - `WeightTensor`: immutable model weights on device (long-lived, RAII)
//! - `DeviceTensor`: mutable computation buffers (per-step, RAII)
//!
//! By encoding device memory ownership in the type system, Rust's Drop
//! trait automatically manages all device allocations. No `std::mem::forget`
//! or manual `aclrtFree` needed.

use std::fmt;
use super::tensor::DType;

#[cfg(feature = "ascend")]
use ascend::DeviceBuffer;

// ─── TensorMeta ────────────────────────────────────────────────────────

/// Pure metadata: shape, dtype, and debug name.
///
/// Used during graph construction and shape inference. Holds no device
/// resources and can be freely cloned.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub name: String,
}

impl TensorMeta {
    pub fn new(shape: Vec<usize>, dtype: DType, name: impl Into<String>) -> Self {
        Self { shape, dtype, name: name.into() }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }
}

impl fmt::Display for TensorMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{:?}]", self.name, self.shape)
    }
}

// ─── DeviceTensor ──────────────────────────────────────────────────────

/// A computation buffer on the NPU device.
///
/// Owns device memory via `DeviceBuffer` (RAII). When dropped, device
/// memory is automatically freed — no manual cleanup needed.
///
/// Used for intermediate results during the forward pass. Lifetime is
/// typically a single decode step.
#[cfg(feature = "ascend")]
pub struct DeviceTensor {
    pub meta: TensorMeta,
    pub buf: DeviceBuffer,
}

#[cfg(feature = "ascend")]
impl DeviceTensor {
    /// Create from an existing device buffer.
    pub fn from_buf(shape: Vec<usize>, dtype: DType, name: impl Into<String>, buf: DeviceBuffer) -> Self {
        Self {
            meta: TensorMeta::new(shape, dtype, name),
            buf,
        }
    }

    /// Allocate a new device buffer with the given shape.
    pub fn alloc(shape: Vec<usize>, dtype: DType, name: impl Into<String>) -> Result<Self, ascend::AscendError> {
        let meta = TensorMeta::new(shape, dtype, name);
        let buf = DeviceBuffer::alloc(meta.size_bytes())?;
        Ok(Self { meta, buf })
    }

    /// Raw device pointer (for passing to CANN operators).
    pub fn ptr(&self) -> *mut std::os::raw::c_void {
        self.buf.ptr()
    }

    pub fn shape(&self) -> &[usize] {
        &self.meta.shape
    }

    pub fn dtype(&self) -> DType {
        self.meta.dtype
    }

    pub fn size_bytes(&self) -> usize {
        self.meta.size_bytes()
    }

    pub fn name(&self) -> &str {
        &self.meta.name
    }
}

#[cfg(feature = "ascend")]
impl fmt::Debug for DeviceTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeviceTensor({:?}, {:?}, name={:?})", self.meta.shape, self.meta.dtype, self.meta.name)
    }
}

#[cfg(feature = "ascend")]
impl fmt::Display for DeviceTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{:?}]", self.meta.name, self.meta.shape)
    }
}

// ─── WeightTensor ──────────────────────────────────────────────────────

/// An immutable model weight tensor on the NPU device.
///
/// Owns device memory via `DeviceBuffer` (RAII). Lifetime is the entire
/// model lifetime (loaded once, never freed until process exit).
///
/// Separated from `DeviceTensor` to prevent accidental mutation of weights.
#[cfg(feature = "ascend")]
pub struct WeightTensor {
    pub meta: TensorMeta,
    pub buf: DeviceBuffer,
}

#[cfg(feature = "ascend")]
impl WeightTensor {
    /// Create from an existing device buffer (used during weight upload).
    pub fn from_buf(shape: Vec<usize>, dtype: DType, name: impl Into<String>, buf: DeviceBuffer) -> Self {
        Self {
            meta: TensorMeta::new(shape, dtype, name),
            buf,
        }
    }

    /// Convert from an old-style Tensor by taking ownership of its DeviceBuffer.
    ///
    /// This bridges the old weight loading system with the new typed system.
    /// Panics if the Tensor has no device_buf.
    pub fn from_tensor(mut tensor: super::tensor::Tensor) -> Self {
        let buf = tensor.device_buf.take()
            .expect("WeightTensor::from_tensor: tensor has no device_buf");
        Self {
            meta: TensorMeta::new(tensor.shape.clone(), tensor.dtype, &tensor.name),
            buf,
        }
    }

    /// Raw device pointer (for passing to CANN operators).
    pub fn ptr(&self) -> *mut std::os::raw::c_void {
        self.buf.ptr()
    }

    pub fn shape(&self) -> &[usize] {
        &self.meta.shape
    }

    pub fn dtype(&self) -> DType {
        self.meta.dtype
    }

    pub fn size_bytes(&self) -> usize {
        self.meta.size_bytes()
    }

    pub fn name(&self) -> &str {
        &self.meta.name
    }
}

#[cfg(feature = "ascend")]
impl fmt::Debug for WeightTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WeightTensor({:?}, {:?}, name={:?})", self.meta.shape, self.meta.dtype, self.meta.name)
    }
}

#[cfg(feature = "ascend")]
impl fmt::Display for WeightTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{:?}]", self.meta.name, self.meta.shape)
    }
}

// ─── TensorPool ────────────────────────────────────────────────────────

/// Pool of computation buffers for a single forward pass.
///
/// Each slot holds an `Option<DeviceTensor>`. When a new result is put
/// into a slot, the old `DeviceTensor` is dropped automatically, freeing
/// its device memory via RAII. No manual cleanup needed.
#[cfg(feature = "ascend")]
pub struct TensorPool {
    slots: Vec<Option<DeviceTensor>>,
}

#[cfg(feature = "ascend")]
impl TensorPool {
    /// Create a new pool with `n` empty slots.
    pub fn new(num_slots: usize) -> Self {
        Self {
            slots: (0..num_slots).map(|_| None).collect(),
        }
    }

    /// Store a computation result. Old value is automatically dropped (RAII).
    pub fn put(&mut self, idx: usize, tensor: DeviceTensor) {
        self.slots[idx] = Some(tensor);
    }

    /// Borrow a tensor for reading (e.g., as input to an operator).
    pub fn get(&self, idx: usize) -> &DeviceTensor {
        self.slots[idx].as_ref()
            .unwrap_or_else(|| panic!("TensorPool::get: slot {} is empty", idx))
    }

    /// Take ownership of a tensor (e.g., for in-place operators like add, qk_norm).
    /// The slot becomes empty after this call.
    pub fn take(&mut self, idx: usize) -> DeviceTensor {
        self.slots[idx].take()
            .unwrap_or_else(|| panic!("TensorPool::take: slot {} is empty", idx))
    }
}

// Drop is automatic: Vec<Option<DeviceTensor>> → each DeviceTensor::drop → DeviceBuffer::drop → aclrtFree

// ─── KvCachePool ───────────────────────────────────────────────────────

/// Dynamic, segmented KV cache pool on NPU device.
///
/// Bridges logical `BlockId`s (from `kv-cache` crate's `RadixKvManager`) to
/// physical device memory. K and V have separate buffer pools because
/// `aclnnIncreFlashAttentionV4` requires separate `key_cache` and `value_cache`
/// tensor arguments.
///
/// Memory layout per chunk:
/// ```text
/// chunk = DeviceBuffer of (blocks_per_chunk × num_layers × block_size × num_kv_heads × head_dim × dtype_size) bytes
///
/// Within a chunk, data is laid out as:
///   [block_0_layer_0] [block_0_layer_1] ... [block_0_layer_N]
///   [block_1_layer_0] [block_1_layer_1] ... [block_1_layer_N]
///   ...
///
/// Each block-layer region = block_size × num_kv_heads × head_dim elements
/// ```
///
/// This layout allows each layer's cache tensor to be a slice view into the
/// chunk, which is what `aclnnIncreFlashAttentionV4` needs via `aclTensorList`.
#[cfg(feature = "ascend")]
pub struct KvCachePool {
    /// Separate chunk pools for K and V.
    k_chunks: Vec<DeviceBuffer>,
    v_chunks: Vec<DeviceBuffer>,

    /// Number of blocks that fit in each chunk.
    blocks_per_chunk: usize,

    /// Total number of blocks currently allocated.
    total_blocks: usize,

    /// Model config (fixed at creation).
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,  // tokens per block
    pub dtype: DType,
}

#[cfg(feature = "ascend")]
impl KvCachePool {
    /// Create an empty pool. Call `grow()` to allocate chunks.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: DType,
    ) -> Self {
        Self {
            k_chunks: Vec::new(),
            v_chunks: Vec::new(),
            blocks_per_chunk: 0,
            total_blocks: 0,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            dtype,
        }
    }

    /// Bytes per block per layer (K or V, not both).
    fn block_layer_bytes(&self) -> usize {
        self.block_size * self.num_kv_heads * self.head_dim * self.dtype.size_bytes()
    }

    /// Bytes per block across all layers (K or V, not both).
    fn block_bytes(&self) -> usize {
        self.block_layer_bytes() * self.num_layers
    }

    /// Grow the pool by allocating a new chunk with `num_blocks` blocks.
    ///
    /// Returns the range of new block IDs: `[old_total .. old_total + num_blocks)`.
    pub fn grow(&mut self, num_blocks: usize) -> Result<std::ops::Range<usize>, ascend::AscendError> {
        let chunk_bytes = num_blocks * self.block_bytes();
        let k_chunk = DeviceBuffer::alloc(chunk_bytes)?;
        let v_chunk = DeviceBuffer::alloc(chunk_bytes)?;

        let old_total = self.total_blocks;
        self.k_chunks.push(k_chunk);
        self.v_chunks.push(v_chunk);

        if self.blocks_per_chunk == 0 {
            self.blocks_per_chunk = num_blocks;
        }
        // Allow chunks of different sizes by storing the count
        self.total_blocks += num_blocks;

        eprintln!(
            "[KvCachePool] grew by {} blocks ({:.1} MB K + {:.1} MB V), total {} blocks",
            num_blocks,
            chunk_bytes as f64 / 1_048_576.0,
            chunk_bytes as f64 / 1_048_576.0,
            self.total_blocks,
        );

        Ok(old_total..self.total_blocks)
    }

    /// Get the device pointer for block `block_id`, layer `layer`, K side.
    ///
    /// Returns a pointer to the start of `[block_size, num_kv_heads, head_dim]`
    /// region in device memory.
    pub fn block_k_ptr(&self, block_id: usize, layer: usize) -> *mut std::os::raw::c_void {
        self.block_ptr_inner(&self.k_chunks, block_id, layer)
    }

    /// Get the device pointer for block `block_id`, layer `layer`, V side.
    pub fn block_v_ptr(&self, block_id: usize, layer: usize) -> *mut std::os::raw::c_void {
        self.block_ptr_inner(&self.v_chunks, block_id, layer)
    }

    fn block_ptr_inner(
        &self,
        chunks: &[DeviceBuffer],
        block_id: usize,
        layer: usize,
    ) -> *mut std::os::raw::c_void {
        assert!(block_id < self.total_blocks, "block_id {} >= total_blocks {}", block_id, self.total_blocks);
        assert!(layer < self.num_layers, "layer {} >= num_layers {}", layer, self.num_layers);

        // Find which chunk this block lives in
        let mut remaining = block_id;
        let mut chunk_idx = 0;
        let bpc = self.blocks_per_chunk;
        while remaining >= bpc {
            remaining -= bpc;
            chunk_idx += 1;
        }
        let local_block = remaining;

        // Offset within the chunk:
        //   block_offset = local_block * (num_layers * block_layer_bytes)
        //                + layer * block_layer_bytes
        let block_layer_bytes = self.block_layer_bytes();
        let offset = local_block * self.num_layers * block_layer_bytes
                   + layer * block_layer_bytes;

        unsafe {
            (chunks[chunk_idx].ptr() as *mut u8).add(offset) as *mut std::os::raw::c_void
        }
    }

    /// Total number of blocks currently allocated.
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Number of chunks allocated.
    pub fn num_chunks(&self) -> usize {
        self.k_chunks.len()
    }
}

#[cfg(feature = "ascend")]
impl fmt::Debug for KvCachePool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KvCachePool(blocks={}, chunks={}, layers={}, kv_heads={}, head_dim={}, block_size={})",
            self.total_blocks,
            self.k_chunks.len(),
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )
    }
}
