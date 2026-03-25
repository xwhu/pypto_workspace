//! Safe element-wise operation wrappers (Add, InplaceAdd, Mul).

use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::{AclDataType, AclOpExecutor, AclScalar};
use std::os::raw::c_void;

/// In-place addition: self += other * alpha.
///
/// # Arguments
/// - `stream`: execution stream
/// - `self_`: tensor to add to (modified in-place)
/// - `other`: tensor to add
/// - `alpha`: scale factor for other (typically 1.0)
pub fn inplace_add(
    stream: &Stream,
    self_: &AclTensor,
    other: &AclTensor,
    alpha: f32,
) -> Result<()> {
    // Create alpha scalar
    let alpha_scalar = unsafe {
        let val = alpha;
        aclnn_sys::common::aclCreateScalar(&val as *const f32 as *const c_void, AclDataType::Float)
    };
    if alpha_scalar.is_null() {
        return Err(crate::error::AscendError::InvalidArgument(
            "aclCreateScalar returned null".to_string(),
        ));
    }

    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Stage 1: Get workspace size
    let result = check_aclnn(unsafe {
        aclnn_sys::elementwise::aclnnInplaceAddGetWorkspaceSize(
            self_.raw(),
            other.raw(),
            alpha_scalar as *const AclScalar,
            &mut workspace_size,
            &mut executor,
        )
    });

    // Cleanup scalar on error
    if result.is_err() {
        unsafe {
            aclnn_sys::common::aclDestroyScalar(alpha_scalar as *const AclScalar);
        }
        return result;
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
    let result = check_aclnn(unsafe {
        aclnn_sys::elementwise::aclnnInplaceAdd(ws_ptr, workspace_size, executor, stream.raw())
    });

    // Cleanup scalar
    unsafe {
        aclnn_sys::common::aclDestroyScalar(alpha_scalar as *const AclScalar);
    }

    result
}

/// Element-wise multiplication: out = a * b.
///
/// # Arguments
/// - `stream`: execution stream
/// - `a`: first input tensor
/// - `b`: second input tensor
/// - `out`: output tensor (must be pre-allocated, same shape)
pub fn mul(stream: &Stream, a: &AclTensor, b: &AclTensor, out: &mut AclTensor) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    check_aclnn(unsafe {
        aclnn_sys::elementwise::aclnnMulGetWorkspaceSize(
            a.raw(),
            b.raw(),
            out.raw(),
            &mut workspace_size,
            &mut executor,
        )
    })?;

    let workspace = if workspace_size > 0 {
        Some(DeviceBuffer::alloc(workspace_size as usize)?)
    } else {
        None
    };

    let ws_ptr = workspace
        .as_ref()
        .map(|b| b.ptr())
        .unwrap_or(std::ptr::null_mut());

    check_aclnn(unsafe {
        aclnn_sys::elementwise::aclnnMul(ws_ptr, workspace_size, executor, stream.raw())
    })
}
