//! aclnnEmbedding — Token embedding lookup.
//!
//! C API: `aclnn/aclnn_embedding.h`
//! Computes: out[i] = weight[indices[i]]

use super::common::*;
use std::os::raw::c_void;

extern "C" {
    /// Stage 1: Get workspace size for Embedding.
    ///
    /// - `weight`: embedding table [vocab_size, embed_dim]
    /// - `indices`: token ID tensor [batch, seq_len]
    /// - `out`: output tensor [batch, seq_len, embed_dim]
    pub fn aclnnEmbeddingGetWorkspaceSize(
        weight: *const AclTensor,
        indices: *const AclTensor,
        out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute Embedding.
    pub fn aclnnEmbedding(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
