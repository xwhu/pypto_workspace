//! Safe RotaryPositionEmbedding wrapper.
//!
//! Applies RoPE to a single tensor (call once for Q, once for K).

use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;

/// Apply rotary position embedding to a single tensor.
///
/// # Arguments
/// - `stream`: execution stream
/// - `x`: input tensor (e.g. Q or K) [batch, seq, num_heads, head_dim]
/// - `cos`: cosine table, broadcast-compatible with x
/// - `sin`: sine table, same shape as cos
/// - `mode`: 0=half (standard HuggingFace RoPE), 2=interleave
/// - `out`: output tensor, same shape as x (pre-allocated)
pub fn rotary_position_embedding(
    stream: &Stream,
    x: &AclTensor,
    cos: &AclTensor,
    sin: &AclTensor,
    mode: i64,
    out: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Stage 1: Get workspace size
    check_aclnn(unsafe {
        aclnn_sys::rope::aclnnRotaryPositionEmbeddingGetWorkspaceSize(
            x.raw(),
            cos.raw(),
            sin.raw(),
            mode,
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
        aclnn_sys::rope::aclnnRotaryPositionEmbedding(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    Ok(())
}
