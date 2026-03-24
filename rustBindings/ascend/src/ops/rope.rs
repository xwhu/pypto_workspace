//! Safe ApplyRotaryPosEmb wrapper.

use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;

/// Apply rotary position embedding to query and key tensors **in-place**.
///
/// # Arguments
/// - `stream`: execution stream
/// - `query`: query tensor [batch, seq, num_heads, head_dim] (modified in-place)
/// - `key`: key tensor [batch, seq, num_kv_heads, head_dim] (modified in-place)
/// - `cos`: cosine table [seq, head_dim]
/// - `sin`: sine table [seq, head_dim]
/// - `layout`: tensor layout (0 = BSH-like / default)
pub fn apply_rotary_pos_emb(
    stream: &Stream,
    query: &AclTensor,
    key: &AclTensor,
    cos: &AclTensor,
    sin: &AclTensor,
    layout: i64,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Stage 1: Get workspace size
    check_aclnn(unsafe {
        aclnn_sys::rope::aclnnApplyRotaryPosEmbGetWorkspaceSize(
            query.raw(),
            key.raw(),
            cos.raw(),
            sin.raw(),
            layout,
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
        aclnn_sys::rope::aclnnApplyRotaryPosEmb(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    Ok(())
}
