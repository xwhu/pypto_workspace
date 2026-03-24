//! Safe Permute wrapper.

use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;

/// Permute (transpose) tensor dimensions.
///
/// # Arguments
/// - `stream`: execution stream
/// - `input`: input tensor
/// - `dims`: permutation indices (e.g., [0, 2, 1, 3] for BNSD→BSND)
/// - `output`: output tensor with permuted shape (must be pre-allocated)
pub fn permute(
    stream: &Stream,
    input: &AclTensor,
    dims: &[i64],
    output: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Create AclIntArray for dims
    let dims_arr = unsafe {
        aclnn_sys::common::aclCreateIntArray(dims.as_ptr(), dims.len() as u64)
    };
    if dims_arr.is_null() {
        return Err(crate::error::AscendError::InvalidArgument(
            "aclCreateIntArray for dims returned null".to_string(),
        ));
    }

    // Stage 1: Get workspace size
    let ret = check_aclnn(unsafe {
        aclnn_sys::permute::aclnnPermuteGetWorkspaceSize(
            input.raw(),
            dims_arr,
            output.raw(),
            &mut workspace_size,
            &mut executor,
        )
    });

    if ret.is_err() {
        unsafe { aclnn_sys::common::aclDestroyIntArray(dims_arr) };
        return ret;
    }

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
    let ret = check_aclnn(unsafe {
        aclnn_sys::permute::aclnnPermute(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    });

    unsafe { aclnn_sys::common::aclDestroyIntArray(dims_arr) };
    ret
}
