//! Element-wise operators — Add, Mul, InplaceAdd.
//!
//! C API: `aclnn/aclnn_add.h`, `aclnn/aclnn_mul.h`

use std::os::raw::c_void;
use super::common::*;

// ─── Add (out = self + other * alpha) ──────────────────────────────────

extern "C" {
    /// Stage 1: Add workspace.
    /// Computes: out = self + other * alpha
    pub fn aclnnAddGetWorkspaceSize(
        self_: *const AclTensor,
        other: *const AclTensor,
        alpha: *const AclScalar,
        out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute Add.
    pub fn aclnnAdd(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}

// ─── InplaceAdd (self += other * alpha) ────────────────────────────────

extern "C" {
    /// Stage 1: In-place Add workspace.
    pub fn aclnnInplaceAddGetWorkspaceSize(
        self_: *const AclTensor,
        other: *const AclTensor,
        alpha: *const AclScalar,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute in-place Add.
    pub fn aclnnInplaceAdd(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}

// ─── Mul (out = self * other) ──────────────────────────────────────────

extern "C" {
    /// Stage 1: Mul workspace.
    pub fn aclnnMulGetWorkspaceSize(
        self_: *const AclTensor,
        other: *const AclTensor,
        out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute Mul.
    pub fn aclnnMul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
