//! Common aclnn types and the two-stage execution pattern.

use std::os::raw::c_void;

// Re-export base types from ascendcl-sys
pub use ascendcl_sys::types::{AclDataType, AclFormat};
pub use ascendcl_sys::AclrtStream;

// ─── Opaque aclnn Types ────────────────────────────────────────────────

/// Opaque tensor descriptor for aclnn APIs. NOT the same as aclTensorDesc.
/// Created via `aclCreateTensor`, destroyed via `aclDestroyTensor`.
#[repr(C)]
pub struct AclTensor {
    _private: [u8; 0],
}

/// Opaque scalar descriptor.
#[repr(C)]
pub struct AclScalar {
    _private: [u8; 0],
}

/// Opaque integer array descriptor.
#[repr(C)]
pub struct AclIntArray {
    _private: [u8; 0],
}

/// Opaque boolean array descriptor.
#[repr(C)]
pub struct AclBoolArray {
    _private: [u8; 0],
}

/// Opaque float array descriptor.
#[repr(C)]
pub struct AclFloatArray {
    _private: [u8; 0],
}

/// Opaque tensor list descriptor.
#[repr(C)]
pub struct AclTensorList {
    _private: [u8; 0],
}

/// Opaque operator executor handle (from GetWorkspaceSize, consumed by Execute).
#[repr(C)]
pub struct AclOpExecutor {
    _private: [u8; 0],
}

// ─── Status Code ───────────────────────────────────────────────────────

/// aclnn operation status code. 0 = success.
pub type AclnnStatus = i32;

pub const ACLNN_SUCCESS: AclnnStatus = 0;

// ─── Tensor Descriptor API ─────────────────────────────────────────────

extern "C" {
    /// Create an aclTensor descriptor from raw device memory.
    ///
    /// # Arguments
    /// - `view_dims`: shape array (e.g., [batch, seq, hidden])
    /// - `view_dims_num`: number of dimensions
    /// - `data_type`: element data type
    /// - `strides`: stride array (must match ndim)
    /// - `offset`: byte offset into tensor data
    /// - `format`: memory format (use ND for most LLM ops)
    /// - `storage_dims`: storage shape (usually same as view_dims)
    /// - `storage_dims_num`: storage ndim
    /// - `tensor_data`: pointer to device memory
    pub fn aclCreateTensor(
        view_dims: *const i64,
        view_dims_num: u64,
        data_type: AclDataType,
        strides: *const i64,
        offset: i64,
        format: AclFormat,
        storage_dims: *const i64,
        storage_dims_num: u64,
        tensor_data: *mut c_void,
    ) -> *mut AclTensor;

    /// Destroy an aclTensor descriptor.
    pub fn aclDestroyTensor(tensor: *const AclTensor) -> AclnnStatus;

    /// Create an aclScalar from a float value.
    pub fn aclCreateScalar(
        value: *const c_void,
        data_type: AclDataType,
    ) -> *mut AclScalar;

    /// Destroy an aclScalar.
    pub fn aclDestroyScalar(scalar: *const AclScalar) -> AclnnStatus;

    /// Create an aclIntArray.
    pub fn aclCreateIntArray(
        values: *const i64,
        num: u64,
    ) -> *mut AclIntArray;

    /// Destroy an aclIntArray.
    pub fn aclDestroyIntArray(array: *const AclIntArray) -> AclnnStatus;

    /// Create an aclTensorList from an array of aclTensor pointers.
    pub fn aclCreateTensorList(
        tensors: *const *const AclTensor,
        num: u64,
    ) -> *mut AclTensorList;

    /// Destroy an aclTensorList.
    pub fn aclDestroyTensorList(list: *const AclTensorList) -> AclnnStatus;
}
