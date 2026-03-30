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

use super::tensor::DType;
use std::fmt;

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

#[allow(dead_code)]
impl TensorMeta {
    pub fn new(shape: Vec<usize>, dtype: DType, name: impl Into<String>) -> Self {
        Self {
            shape,
            dtype,
            name: name.into(),
        }
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
    pub fn from_buf(
        shape: Vec<usize>,
        dtype: DType,
        name: impl Into<String>,
        buf: DeviceBuffer,
    ) -> Self {
        Self {
            meta: TensorMeta::new(shape, dtype, name),
            buf,
        }
    }

    /// Allocate a new device buffer with the given shape.
    pub fn alloc(
        shape: Vec<usize>,
        dtype: DType,
        name: impl Into<String>,
    ) -> Result<Self, ascend::AscendError> {
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

    /// Consume the tensor and return the underlying device buffer.
    /// Used for deferred-drop patterns where the buffer must outlive the tensor.
    pub fn into_buf(self) -> DeviceBuffer {
        self.buf
    }

    /// Create a DeviceTensor backed by a **non-owning** device buffer.
    ///
    /// The buffer's memory is NOT freed when this tensor is dropped.
    /// Used for temporary buffers allocated from a `ScratchArena`, whose
    /// memory lifetime is managed by the arena's `RotatingPool`.
    ///
    /// # Safety
    /// The caller must ensure the underlying device memory remains valid
    /// for the duration of any async stream operations referencing this tensor.
    pub fn from_buf_non_owning(
        shape: Vec<usize>,
        dtype: DType,
        name: impl Into<String>,
        buf: DeviceBuffer,
    ) -> Self {
        // `buf` is already a non-owning view (created by ScratchArena::alloc
        // via DeviceBuffer::from_raw_non_owning), so dropping this tensor
        // will NOT call aclrtFree.
        Self {
            meta: TensorMeta::new(shape, dtype, name),
            buf,
        }
    }

    /// Whether this tensor's buffer is owned (will free on drop).
    ///
    /// Non-owning tensors come from `ScratchArena` and should NOT be
    /// pushed to the deferred-drop list.
    pub fn is_owned(&self) -> bool {
        // DeviceBuffer exposes ownership via the `owned` field.
        // We check by seeing if the buffer size > 0 and ptr is non-null
        // but the buffer won't free on drop. The `owned` field is private
        // in DeviceBuffer, so we expose it through a method.
        self.buf.is_owned()
    }
}

#[cfg(feature = "ascend")]
impl fmt::Debug for DeviceTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeviceTensor({:?}, {:?}, name={:?})",
            self.meta.shape, self.meta.dtype, self.meta.name
        )
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
    pub fn from_buf(
        shape: Vec<usize>,
        dtype: DType,
        name: impl Into<String>,
        buf: DeviceBuffer,
    ) -> Self {
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
        let buf = tensor
            .device_buf
            .take()
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
        write!(
            f,
            "WeightTensor({:?}, {:?}, name={:?})",
            self.meta.shape, self.meta.dtype, self.meta.name
        )
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
/// **Deferred-drop semantics**: When a new result replaces an existing
/// tensor in a slot, the old `DeviceBuffer` is NOT freed immediately.
/// Instead it is moved to an internal deferred list. This is necessary
/// because CANN's `aclrtFree` does not respect stream ordering — it
/// reclaims GPU memory immediately, even if the stream still has pending
/// async operations reading from that memory. The deferred buffers are
/// freed when the pool is dropped (after `sample_argmax` calls
/// `stream.synchronize()`).
#[cfg(feature = "ascend")]
pub struct TensorPool {
    slots: Vec<Option<DeviceTensor>>,
    /// Buffers whose device memory must stay alive until the stream
    /// has been synchronized. Freed on Drop (end of forward pass).
    _deferred: Vec<DeviceBuffer>,
}

#[cfg(feature = "ascend")]
impl TensorPool {
    /// Create a new pool with `n` empty slots.
    pub fn new(num_slots: usize) -> Self {
        Self {
            slots: (0..num_slots).map(|_| None).collect(),
            _deferred: Vec::new(),
        }
    }

    /// Store a computation result.
    ///
    /// If a tensor already exists at this slot, its `DeviceBuffer` is moved
    /// to the deferred-drop list (NOT freed immediately). This prevents
    /// use-after-free on the GPU when the stream has pending reads from
    /// the old buffer.
    ///
    /// Exception: if the old buffer is **non-owning** (arena-backed), it is
    /// simply dropped without being deferred — the arena manages its lifetime.
    pub fn put(&mut self, idx: usize, tensor: DeviceTensor) {
        if let Some(old) = self.slots[idx].take() {
            if old.is_owned() {
                self._deferred.push(old.into_buf());
            }
            // Non-owning (arena) buffers: drop silently — no aclrtFree.
        }
        self.slots[idx] = Some(tensor);
    }
    /// Borrow a tensor for reading (e.g., as input to an operator).
    ///
    /// Also accepts external `DeviceBuffer`s for deferred dropping (e.g.,
    /// temporary FP32 buffers from `matmul_hp`).
    pub fn defer_buffers(&mut self, bufs: Vec<DeviceBuffer>) {
        self._deferred.extend(bufs);
    }

    /// Release deferred buffers after the caller has synchronized the stream.
    ///
    /// This is the escape hatch for long forward passes that need to trim
    /// memory pressure before the whole pool is dropped.
    pub fn release_deferred_after_sync(&mut self) {
        self._deferred.clear();
    }

    /// Borrow a tensor for reading (e.g., as input to an operator).
    pub fn get(&self, idx: usize) -> &DeviceTensor {
        self.slots[idx]
            .as_ref()
            .unwrap_or_else(|| panic!("TensorPool::get: slot {} is empty", idx))
    }

    /// Take ownership of a tensor (e.g., for in-place operators like add, qk_norm).
    /// The slot becomes empty after this call.
    pub fn take(&mut self, idx: usize) -> DeviceTensor {
        self.slots[idx]
            .take()
            .unwrap_or_else(|| panic!("TensorPool::take: slot {} is empty", idx))
    }
}

// Drop order: slots first (DeviceTensor → DeviceBuffer → aclrtFree),
// then _deferred (DeviceBuffer → aclrtFree). By the time Drop runs,
// the forward pass is complete and stream has been synchronized.
