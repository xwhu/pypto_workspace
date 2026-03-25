//! Safe RmsNorm wrapper.

use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;
use std::os::raw::c_void;

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
    // Build rstd shape: same as x but last dim = 1
    let x_shape = x.shape();
    let mut rstd_shape: Vec<i64> = x_shape.to_vec();
    if let Some(last) = rstd_shape.last_mut() {
        *last = 1;
    }
    let rstd_numel: i64 = rstd_shape.iter().product();
    let rstd_bytes = (rstd_numel as usize) * 4; // Float32 for rstd
    let rstd_buf = DeviceBuffer::alloc(rstd_bytes)?;
    let rstd_tensor = crate::tensor::AclTensor::from_ptr(
        &rstd_shape,
        aclnn_sys::common::AclDataType::Float,
        rstd_buf.ptr(),
    )?;

    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Stage 1: Get workspace size
    check_aclnn(unsafe {
        aclnn_sys::rmsnorm::aclnnRmsNormGetWorkspaceSize(
            x.raw(),
            gamma.raw(),
            epsilon,
            y.raw(),
            rstd_tensor.raw(), // rstd output (required, not optional)
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
        aclnn_sys::rmsnorm::aclnnRmsNorm(ws_ptr, workspace_size, executor, stream.raw())
    })?;

    Ok(())
}
