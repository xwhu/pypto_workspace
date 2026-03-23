//! AscendCL type definitions.
//!
//! These correspond to the C enums and typedefs in:
//! - `acl/acl_base.h`
//! - `acl/acl_rt.h`

use std::os::raw::c_void;

// ─── Opaque Handle Types ───────────────────────────────────────────────

/// Opaque stream handle. Corresponds to `aclrtStream` (typedef `void*`).
pub type AclrtStream = *mut c_void;

/// Opaque context handle. Corresponds to `aclrtContext` (typedef `void*`).
pub type AclrtContext = *mut c_void;

/// Opaque event handle. Corresponds to `aclrtEvent` (typedef `void*`).
pub type AclrtEvent = *mut c_void;

// ─── Error Codes ───────────────────────────────────────────────────────

/// AscendCL error/status code. 0 = success.
pub type AclError = i32;

// Common error codes
pub const ACL_SUCCESS: AclError = 0;
pub const ACL_ERROR_INVALID_PARAM: AclError = 100000;
pub const ACL_ERROR_UNINITIALIZE: AclError = 100001;
pub const ACL_ERROR_REPEAT_INITIALIZE: AclError = 100002;
pub const ACL_ERROR_INVALID_FILE: AclError = 100003;
pub const ACL_ERROR_WRITE_FILE: AclError = 100004;
pub const ACL_ERROR_INVALID_FILE_SIZE: AclError = 100005;
pub const ACL_ERROR_PARSE_FILE: AclError = 100006;
pub const ACL_ERROR_FILE_MISSING_ATTR: AclError = 100007;
pub const ACL_ERROR_FILE_ATTR_INVALID: AclError = 100008;
pub const ACL_ERROR_INVALID_DUMP_CONFIG: AclError = 100009;
pub const ACL_ERROR_INVALID_PROFILING_CONFIG: AclError = 100010;
pub const ACL_ERROR_INVALID_MODEL_ID: AclError = 100011;
pub const ACL_ERROR_DESERIALIZE_MODEL: AclError = 100012;
pub const ACL_ERROR_PARSE_MODEL: AclError = 100013;
pub const ACL_ERROR_READ_MODEL_FAILURE: AclError = 100014;
pub const ACL_ERROR_MODEL_SIZE_INVALID: AclError = 100015;
pub const ACL_ERROR_MODEL_MISSING_ATTR: AclError = 100016;
pub const ACL_ERROR_BAD_ALLOC: AclError = 200000;
pub const ACL_ERROR_RT_FAILURE: AclError = 200001;
pub const ACL_ERROR_STORAGE_OVER_LIMIT: AclError = 200002;
pub const ACL_ERROR_INTERNAL_ERROR: AclError = 500000;
pub const ACL_ERROR_FAILURE: AclError = 500001;
pub const ACL_ERROR_NOT_SUPPORTED: AclError = 500002;

// ─── Data Types ────────────────────────────────────────────────────────

/// Tensor data type. Corresponds to `aclDataType` enum.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AclDataType {
    Float = 0,
    Float16 = 1,
    Int8 = 2,
    Int32 = 3,
    Uint8 = 4,
    Int16 = 6,
    Uint16 = 7,
    Uint32 = 8,
    Int64 = 9,
    Uint64 = 10,
    Double = 11,
    Bool = 12,
    String = 13,
    Complex64 = 16,
    Complex128 = 17,
    BFloat16 = 27,
    Int4 = 29,
}

// ─── Memory Copy Direction ─────────────────────────────────────────────

/// Memory copy kind. Corresponds to `aclrtMemcpyKind`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AclrtMemcpyKind {
    /// Host → Host
    HostToHost = 0,
    /// Host → Device
    HostToDevice = 1,
    /// Device → Host
    DeviceToHost = 2,
    /// Device → Device
    DeviceToDevice = 3,
}

// ─── Memory Allocation Policy ──────────────────────────────────────────

/// Device memory allocation policy. Corresponds to `aclrtMemMallocPolicy`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AclrtMemMallocPolicy {
    /// Allocate max available size, may waste memory.
    Huge = 0,
    /// Allocate on demand.
    Normal = 1,
    /// Allocate in high-bandwidth memory.
    HighBandWidth = 2,
}

// ─── Tensor Format ─────────────────────────────────────────────────────

/// Tensor storage format. Corresponds to `aclFormat`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AclFormat {
    /// N-dimensional (row-major, no special layout).
    Nd = 2,
    /// NCHW format.
    Nchw = 0,
    /// NHWC format.
    Nhwc = 1,
    /// Ascend-specific fractal formats (rarely needed for LLM).
    NdRnn = 25,
    FractalNz = 29,
}

// ─── Run Mode ──────────────────────────────────────────────────────────

/// Execution run mode.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AclrtRunMode {
    /// Running on host CPU with NPU device.
    AclDevice = 0,
    /// Running on host.
    AclHost = 1,
}
