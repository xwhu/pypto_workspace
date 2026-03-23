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
        Self::from_ptr(shape, dtype, buffer.ptr())
    }

    /// Create a tensor descriptor from a raw device pointer.
    ///
    /// Use this when device memory is managed externally (e.g., weight tensors
    /// whose memory is owned by a `DeviceBuffer` in the model).
    ///
    /// # Safety
    /// The caller must ensure the pointer remains valid for the lifetime of this AclTensor.
    pub fn from_ptr(shape: &[i64], dtype: AclDataType, device_ptr: *mut c_void) -> Result<Self> {
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
                device_ptr,
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

    /// Create a **transposed** 2D tensor view from a [N, K] storage layout.
    ///
    /// The resulting AclTensor appears as [K, N] to CANN operators.
    /// This is needed for PyTorch-convention weights stored as [out_features, in_features]
    /// when used in matmul(input, weight) which expects B=[K, N].
    ///
    /// # Arguments
    /// - `storage_shape`: physical layout [N, K] (how data is stored)
    /// - `dtype`: element data type
    /// - `device_ptr`: pointer to device memory
    ///
    /// # Panics
    /// If `storage_shape` is not 2D.
    pub fn from_ptr_transposed_2d(
        storage_shape: &[i64],
        dtype: AclDataType,
        device_ptr: *mut c_void,
    ) -> Result<Self> {
        assert_eq!(storage_shape.len(), 2, "from_ptr_transposed_2d requires 2D shape");
        let n = storage_shape[0]; // out_features
        let k = storage_shape[1]; // in_features

        // Transposed view: logical shape [K, N], but physical strides from [N, K] row-major
        let view_shape = [k, n];
        let strides = [1i64, k]; // stride[0]=1 (step within row), stride[1]=K (step between cols)

        let raw = unsafe {
            aclnn_sys::common::aclCreateTensor(
                view_shape.as_ptr(),
                2,
                dtype,
                strides.as_ptr(),
                0,
                AclFormat::Nd,
                storage_shape.as_ptr(), // storage dims = original [N, K]
                2,
                device_ptr,
            )
        };

        if raw.is_null() {
            return Err(crate::error::AscendError::InvalidArgument(
                "aclCreateTensor (transposed) returned null".to_string(),
            ));
        }

        Ok(Self {
            raw,
            shape: view_shape.to_vec(),
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
