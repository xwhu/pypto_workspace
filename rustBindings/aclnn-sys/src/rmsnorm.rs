//! aclnnRmsNorm — Root Mean Square Layer Normalization.
//!
//! C API: `aclnn/aclnn_rms_norm.h`
//! Computes: out = x * weight / sqrt(mean(x^2) + eps)

use std::os::raw::c_void;
use super::common::*;

extern "C" {
    /// Stage 1: Get workspace size for RmsNorm.
    ///
    /// - `x`: input tensor [*, hidden_size]
    /// - `gamma`: weight tensor [hidden_size]
    /// - `epsilon`: small constant for numerical stability
    /// - `y`: output tensor [*, hidden_size]
    /// - `rstd`: reciprocal standard deviation output (can be null if not needed)
    pub fn aclnnRmsNormGetWorkspaceSize(
        x: *const AclTensor,
        gamma: *const AclTensor,
        epsilon: f64,
        y: *const AclTensor,
        rstd: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute RmsNorm.
    pub fn aclnnRmsNorm(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
