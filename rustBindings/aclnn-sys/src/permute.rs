//! Permute operator — aclnnPermute.

use super::common::*;

extern "C" {
    /// aclnnPermute: transpose/permute tensor dimensions.
    /// - `self_`: input tensor
    /// - `dims`: permutation array (e.g., [0, 2, 1, 3] for BNSD→BSND)
    /// - `out`: output tensor (must be pre-allocated with permuted shape)
    pub fn aclnnPermuteGetWorkspaceSize(
        self_: *const AclTensor,
        dims: *const AclIntArray,
        out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    pub fn aclnnPermute(
        workspace: *mut std::os::raw::c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
