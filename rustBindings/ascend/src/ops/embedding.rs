//! Safe Embedding wrapper.

use std::os::raw::c_void;
use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;

/// Embedding lookup: out[i] = weight[indices[i]].
///
/// # Arguments
/// - `stream`: execution stream
/// - `weight`: embedding table [vocab_size, embed_dim]
/// - `indices`: token ID tensor [batch, seq_len] (Int64 on device)
/// - `out`: output tensor [batch, seq_len, embed_dim] (must be pre-allocated)
pub fn embedding(
    stream: &Stream,
    weight: &AclTensor,
    indices: &AclTensor,
    out: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Stage 1: Get workspace size
    check_aclnn(unsafe {
        aclnn_sys::embedding::aclnnEmbeddingGetWorkspaceSize(
            weight.raw(),
            indices.raw(),
            out.raw(),
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
        aclnn_sys::embedding::aclnnEmbedding(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    Ok(())
}
