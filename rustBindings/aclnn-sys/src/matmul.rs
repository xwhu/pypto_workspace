//! aclnnMatmul — matrix multiplication operator.
//!
//! C API: `aclnn/aclnn_matmul.h`
//! Computes: out = self @ mat2

use std::os::raw::c_void;
use super::common::*;

extern "C" {
    /// Stage 1: Get workspace size for matmul.
    ///
    /// - `self_`: [M, K] input tensor
    /// - `mat2`: [K, N] weight tensor
    /// - `out`: [M, N] output tensor
    /// - `cube_math_type`: compute precision (0 = default, 1 = high precision)
    pub fn aclnnMatmulGetWorkspaceSize(
        self_: *const AclTensor,
        mat2: *const AclTensor,
        out: *const AclTensor,
        cube_math_type: i8,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute matmul.
    pub fn aclnnMatmul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}

// ─── Batch MatMul ──────────────────────────────────────────────────────

extern "C" {
    /// Stage 1: Batch matmul workspace.
    /// out = batch_matmul(self, mat2)
    pub fn aclnnBatchMatMulGetWorkspaceSize(
        self_: *const AclTensor,
        mat2: *const AclTensor,
        out: *const AclTensor,
        cube_math_type: i8,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute batch matmul.
    pub fn aclnnBatchMatMul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
