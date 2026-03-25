//! Safe matmul wrapper.

use std::os::raw::c_void;
use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;

/// Matrix multiplication: out = a @ b.
///
/// # Arguments
/// - `stream`: execution stream
/// - `a`: input tensor [M, K]
/// - `b`: weight tensor [K, N]
/// - `out`: output tensor [M, N] (must be pre-allocated)
///
/// The operation is enqueued on the stream but not waited for.
/// Call `stream.synchronize()` for the result.
pub fn matmul(
    stream: &Stream,
    a: &AclTensor,
    b: &AclTensor,
    out: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Stage 1: Get workspace size
    check_aclnn(unsafe {
        aclnn_sys::matmul::aclnnMatmulGetWorkspaceSize(
            a.raw(),
            b.raw(),
            out.raw(),
            0, // cube_math_type: 0=default precision
            &mut workspace_size,
            &mut executor,
        )
    })?;

    // Allocate workspace on device
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
        aclnn_sys::matmul::aclnnMatmul(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    Ok(())
}
