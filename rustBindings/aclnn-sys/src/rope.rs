//! aclnnRotaryPosEmb — Rotary Position Embedding (RoPE).
//!
//! C API: `aclnn/aclnn_rotary_pos_emb.h`
//! Applies rotary position embedding to query and key tensors.

use std::os::raw::c_void;
use super::common::*;

extern "C" {
    /// Stage 1: Get workspace size for RotaryPosEmb.
    ///
    /// - `query`: [batch, seq, num_heads, head_dim]
    /// - `key`: [batch, seq, num_kv_heads, head_dim]
    /// - `cos`: [seq, head_dim] cosine table
    /// - `sin`: [seq, head_dim] sine table
    /// - `query_output`: output for rotated query
    /// - `key_output`: output for rotated key
    pub fn aclnnRotaryPosEmbGetWorkspaceSize(
        query: *const AclTensor,
        key: *const AclTensor,
        cos: *const AclTensor,
        sin: *const AclTensor,
        query_output: *const AclTensor,
        key_output: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute RotaryPosEmb.
    pub fn aclnnRotaryPosEmb(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
