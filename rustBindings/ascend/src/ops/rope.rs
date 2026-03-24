//! Safe RotaryPosEmb wrapper.

use std::os::raw::c_void;
use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;

/// Rotary Position Embedding: apply rotary embeddings to query and key.
///
/// # Arguments
/// - `stream`: execution stream
/// - `query`: input query tensor [batch, seq, num_heads, head_dim]
/// - `key`: input key tensor [batch, seq, num_kv_heads, head_dim]
/// - `cos`: cosine table [seq, head_dim] (FP16/FP32)
/// - `sin`: sine table [seq, head_dim] (FP16/FP32)
/// - `query_out`: output rotated query (same shape as query, pre-allocated)
/// - `key_out`: output rotated key (same shape as key, pre-allocated)
pub fn rotary_pos_emb(
    stream: &Stream,
    query: &AclTensor,
    key: &AclTensor,
    cos: &AclTensor,
    sin: &AclTensor,
    query_out: &mut AclTensor,
    key_out: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Stage 1: Get workspace size
    check_aclnn(unsafe {
        aclnn_sys::rope::aclnnRotaryPosEmbGetWorkspaceSize(
            query.raw(),
            key.raw(),
            cos.raw(),
            sin.raw(),
            query_out.raw(),
            key_out.raw(),
            &mut workspace_size,
            &mut executor,
        )
    })?;

    // Allocate workspace
    let workspace = if workspace_size > 0 {
        Some(DeviceBuffer::alloc(workspace_size as usize)?)
    } else {
        None
    };

    let ws_ptr = workspace
        .as_ref()
        .map(|b| b.ptr())
        .unwrap_or(std::ptr::null_mut());

    // Stage 2: Execute
    check_aclnn(unsafe {
        aclnn_sys::rope::aclnnRotaryPosEmb(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    Ok(())
}
