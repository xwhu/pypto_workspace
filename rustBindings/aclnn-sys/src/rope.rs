//! aclnnApplyRotaryPosEmb — Rotary Position Embedding (RoPE).
//!
//! C API: `aclnn/aclnn_apply_rotary_pos_emb.h`
//! Applies rotary position embedding to query and key tensors **in-place**.

use std::os::raw::c_void;
use super::common::*;

extern "C" {
    /// Stage 1: Get workspace size for ApplyRotaryPosEmb.
    ///
    /// **In-place**: modifies `queryRef` and `keyRef` directly.
    ///
    /// - `queryRef`: [batch, seq, num_heads, head_dim] (modified in-place)
    /// - `keyRef`: [batch, seq, num_kv_heads, head_dim] (modified in-place)
    /// - `cos`: [seq, head_dim] cosine table
    /// - `sin`: [seq, head_dim] sine table
    /// - `layout`: tensor layout (0 = BSH, 1 = BNSD, etc.)
    pub fn aclnnApplyRotaryPosEmbGetWorkspaceSize(
        queryRef: *const AclTensor,
        keyRef: *const AclTensor,
        cos: *const AclTensor,
        sin: *const AclTensor,
        layout: i64,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute ApplyRotaryPosEmb.
    pub fn aclnnApplyRotaryPosEmb(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
