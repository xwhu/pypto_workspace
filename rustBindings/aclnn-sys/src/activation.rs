//! Activation operators — SiLU, SwiGLU.
//!
//! C API: `aclnn/aclnn_silu.h`, `aclnn/aclnn_swiglu.h`

use super::common::*;
use std::os::raw::c_void;

// ─── SiLU (Sigmoid Linear Unit) ────────────────────────────────────────

extern "C" {
    /// Stage 1: SiLU workspace.
    /// Computes: out = x * sigmoid(x)
    pub fn aclnnSiluGetWorkspaceSize(
        self_: *const AclTensor,
        out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute SiLU.
    pub fn aclnnSilu(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}

// ─── SiLU (in-place) ──────────────────────────────────────────────────

extern "C" {
    /// Stage 1: In-place SiLU workspace.
    pub fn aclnnInplaceSiluGetWorkspaceSize(
        self_: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute in-place SiLU.
    pub fn aclnnInplaceSilu(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}

// ─── SwiGLU ────────────────────────────────────────────────────────────

extern "C" {
    /// Stage 1: SwiGLU workspace.
    /// Computes: out = silu(x1) * x2
    /// where x1 = gate_proj output, x2 = up_proj output
    ///
    /// Note: Some CANN versions may name this differently.
    /// Fallback: use SiLU + elementwise Mul.
    pub fn aclnnSwiGluGetWorkspaceSize(
        x: *const AclTensor,
        dim: i64,
        out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute SwiGLU.
    pub fn aclnnSwiGlu(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
