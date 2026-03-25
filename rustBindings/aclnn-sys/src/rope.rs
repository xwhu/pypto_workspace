//! aclnnRotaryPositionEmbedding — Rotary Position Embedding (RoPE).
//!
//! C API: `aclnn/aclnn_rotary_position_embedding.h`
//! Applies rotary position embedding to a single tensor (Q or K).
//!
//! - x: input tensor, dtype supports Float32/Float16/BFloat16, format ND
//! - cos: cosine table, same dtype as x, shape broadcast-compatible with x
//! - sin: sine table, same dtype as x, shape same as cos
//! - mode: 0=half, 1=quarter, 2=interleave, 3=half-interleave
//! - out: output tensor, same shape and dtype as x

use super::common::*;
use std::os::raw::c_void;

extern "C" {
    /// Stage 1: Get workspace size for RotaryPositionEmbedding.
    pub fn aclnnRotaryPositionEmbeddingGetWorkspaceSize(
        x: *const AclTensor,
        cos: *const AclTensor,
        sin: *const AclTensor,
        mode: i64,
        out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute RotaryPositionEmbedding.
    pub fn aclnnRotaryPositionEmbedding(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
