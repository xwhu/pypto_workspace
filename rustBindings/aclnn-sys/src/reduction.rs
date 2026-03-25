//! Reduction operators — Softmax, ArgMax.
//!
//! C API: `aclnn/aclnn_softmax.h`, `aclnn/aclnn_argmax.h`

use super::common::*;
use std::os::raw::c_void;

// ─── Softmax ───────────────────────────────────────────────────────────

extern "C" {
    /// Stage 1: Softmax workspace.
    /// Computes: out = softmax(self, dim)
    pub fn aclnnSoftmaxGetWorkspaceSize(
        self_: *const AclTensor,
        dim: i64,
        out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute Softmax.
    pub fn aclnnSoftmax(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}

// ─── ArgMax ────────────────────────────────────────────────────────────

extern "C" {
    /// Stage 1: ArgMax workspace.
    /// Returns the index of the maximum value along `dim`.
    pub fn aclnnArgMaxGetWorkspaceSize(
        self_: *const AclTensor,
        dim: i64,
        keepdim: bool,
        out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute ArgMax.
    pub fn aclnnArgMax(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
