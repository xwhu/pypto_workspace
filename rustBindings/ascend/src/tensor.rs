//! Safe aclTensor wrapper with RAII.

use std::os::raw::c_void;
use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use aclnn_sys::common::{AclTensor as RawAclTensor, AclDataType, AclFormat};

/// Safe RAII wrapper around an `aclTensor` descriptor.
///
/// The tensor descriptor points to device memory managed by a `DeviceBuffer`.
/// The descriptor is destroyed on drop.
pub struct AclTensor {
    raw: *mut RawAclTensor,
    shape: Vec<i64>,
    dtype: AclDataType,
}

impl AclTensor {
    /// Create a tensor descriptor pointing to device memory.
    ///
    /// # Arguments
    /// - `shape`: tensor dimensions (e.g., `[batch, seq, hidden]`)
    /// - `dtype`: element data type
    /// - `buffer`: the device memory this tensor points to
    ///
    /// # Safety Requirements (enforced at runtime)
    /// - `buffer.size() >= element_count * dtype_size`
    pub fn new(shape: &[i64], dtype: AclDataType, buffer: &DeviceBuffer) -> Result<Self> {
        // Compute strides (row-major / C-contiguous)
        let ndim = shape.len();
        let mut strides = vec![1i64; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let raw = unsafe {
            aclnn_sys::common::aclCreateTensor(
                shape.as_ptr(),
                ndim as u64,
                dtype,
                strides.as_ptr(),
                0, // offset
                AclFormat::Nd,
                shape.as_ptr(),    // storage dims = view dims for contiguous
                ndim as u64,
                buffer.ptr(),
            )
        };

        if raw.is_null() {
            return Err(crate::error::AscendError::InvalidArgument(
                "aclCreateTensor returned null".to_string(),
            ));
        }

        Ok(Self {
            raw,
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Get the raw pointer (for passing to aclnn operator calls).
    pub fn raw(&self) -> *const RawAclTensor {
        self.raw
    }

    /// Get the raw mutable pointer.
    pub fn raw_mut(&mut self) -> *mut RawAclTensor {
        self.raw
    }

    /// Get the shape.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Get the data type.
    pub fn dtype(&self) -> AclDataType {
        self.dtype
    }

    /// Total number of elements.
    pub fn numel(&self) -> i64 {
        self.shape.iter().product()
    }
}

impl Drop for AclTensor {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = aclnn_sys::common::aclDestroyTensor(self.raw);
            }
        }
    }
}

// Safety: AclTensor descriptors are thread-safe (they're just metadata + pointer).
unsafe impl Send for AclTensor {}
unsafe impl Sync for AclTensor {}
