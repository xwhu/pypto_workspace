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
