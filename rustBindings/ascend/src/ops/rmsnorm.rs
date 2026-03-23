//! Safe RmsNorm wrapper.

use std::os::raw::c_void;
use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;

/// RMS Layer Normalization: out = x * weight / sqrt(mean(x^2) + eps).
///
/// # Arguments
/// - `stream`: execution stream
/// - `x`: input tensor [*, hidden_size]
/// - `gamma`: weight tensor [hidden_size]
/// - `epsilon`: numerical stability constant (typically 1e-6)
/// - `y`: output tensor [*, hidden_size] (must be pre-allocated)
pub fn rmsnorm(
    stream: &Stream,
    x: &AclTensor,
    gamma: &AclTensor,
    epsilon: f64,
    y: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Stage 1: Get workspace size
    // rstd output is null — we don't need it
    check_aclnn(unsafe {
        aclnn_sys::rmsnorm::aclnnRmsNormGetWorkspaceSize(
            x.raw(),
            gamma.raw(),
            epsilon,
            y.raw(),
            std::ptr::null(), // rstd — not needed
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
        aclnn_sys::rmsnorm::aclnnRmsNorm(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    Ok(())
}
