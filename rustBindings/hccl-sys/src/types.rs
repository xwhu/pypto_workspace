//! HCCL type definitions.
//!
//! These correspond to the C types in `hccl/hccl_types.h` from the CANN SDK.

use std::os::raw::c_void;

// ─── Opaque Handle Types ───────────────────────────────────────────────

/// Opaque communicator handle. Corresponds to `HcclComm` (typedef `void*`).
pub type HcclComm = *mut c_void;

// ─── Root Info ─────────────────────────────────────────────────────────

/// Unique ID for collective communication initialization.
/// All ranks must share the same `HcclRootInfo` to form a communicator group.
///
/// Corresponds to `HcclRootInfo` — a 4120-byte opaque struct in HCCL.
/// Rank 0 generates this via `HcclGetRootInfo()`, then broadcasts to all ranks.
#[repr(C)]
#[derive(Clone)]
pub struct HcclRootInfo {
    pub internal: [u8; HCCL_ROOT_INFO_BYTES],
}

/// Size of `HcclRootInfo` in bytes.
pub const HCCL_ROOT_INFO_BYTES: usize = 4120;

impl Default for HcclRootInfo {
    fn default() -> Self {
        Self {
            internal: [0u8; HCCL_ROOT_INFO_BYTES],
        }
    }
}

impl std::fmt::Debug for HcclRootInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HcclRootInfo([{}; {}])", self.internal[0], HCCL_ROOT_INFO_BYTES)
    }
}

// ─── Error Codes ───────────────────────────────────────────────────────

/// HCCL return code. Corresponds to `HcclResult` enum.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcclResult {
    /// success
    Success = 0,               
    /// parameter error
    ErrorPara = 1,                
    /// empty pointer
    ErrorPtr = 2,                 
    /// memory error
    ErrorMemory = 3,              
    /// internal error
    ErrorInternal = 4,            
    /// not support feature
    ErrorNotSupport = 5,         
    /// not found specific resource
    ErrorNotFound = 6,           
    /// resource unavailable
    ErrorUnavail = 7,             
    /// call system interface error
    ErrorSyscall = 8,             
    /// timeout
    ErrorTimeout = 9,             
    /// open file fail
    ErrorOpenFileFailure = 10,  
    /// tcp connect fail
    ErrorTcpConnect = 11,        
    /// roce connect fail
    ErrorRoceConnect = 12,       
    /// tcp transfer fail
    ErrorTcpTransfer = 13,       
    /// roce transfer fail
    ErrorRoceTransfer = 14,      
    /// call runtime api fail
    ErrorRuntime = 15,            
    /// call driver api fail
    ErrorDrv = 16,                
    /// call profiling api fail
    ErrorProfiling = 17,          
    /// call cce api fail
    ErrorCce = 18,                
    /// call network api fail
    ErrorNetwork = 19,            
    /// try again
    ErrorAgain = 20,              
    /// error cqe
    ErrorRemote = 21,             
    /// error communicator suspending
    ErrorSuspending = 22,         
    /// retry constraint
    ErrorOpretryFail = 23,       
    /// out of memory
    ErrorOom = 24,                
    /// The error information is in the status.
    ErrorInStatus = 1041,        
    /// reserved
    ErrorReserved = 1042,
}

impl HcclResult {
    /// Returns true if the call succeeded.
    pub fn is_ok(self) -> bool {
        self == HcclResult::Success
    }
}

// ─── Data Types ────────────────────────────────────────────────────────

/// HCCL data type for collective operations.
/// Corresponds to `HcclDataType` enum in `hccl/hccl_types.h`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcclDataType {
    Int8 = 0,
    Int16 = 1,
    Int32 = 2,
    Float16 = 3,
    Float32 = 4,
    Int64 = 5,
    Uint64 = 6,
    Uint8 = 7,
    Uint16 = 8,
    Uint32 = 9,
    Float64 = 10,
    BFloat16 = 11,
    Int128 = 12,
    Hif8 = 14,
    Fp8E4M3 = 15,
    Fp8E5M2 = 16,
    Fp8E8M0 = 17,
    MxFp8 = 18,
    Reserved = 255,
}

// ─── Reduction Operations ──────────────────────────────────────────────

/// Reduction operation for collective calls (e.g., AllReduce).
/// Corresponds to `HcclReduceOp` enum.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcclReduceOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
}
