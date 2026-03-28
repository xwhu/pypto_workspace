//! Safe matmul wrapper.

use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;
use std::os::raw::c_void;

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
pub fn matmul(stream: &Stream, a: &AclTensor, b: &AclTensor, out: &mut AclTensor) -> Result<()> {
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
        aclnn_sys::matmul::aclnnMatmul(ws_ptr, workspace_size, executor, stream.raw())
    })?;

    Ok(())
}

/// High-precision matrix multiplication via FP32 upcast.
///
/// For BF16/FP16 inputs:
///   1. Cast `a` and `b` to FP32
///   2. Compute matmul in FP32 (full-precision accumulation)
///   3. Cast result back to the original dtype
///
/// If inputs are already FP32, behaves identically to [`matmul`].
///
/// This is used for row-shard matmuls (O-Proj, down_proj) where
/// the dot-product split across TP ranks causes BF16 accumulation errors.
pub fn matmul_fp32(
    stream: &Stream,
    a: &AclTensor,
    b: &AclTensor,
    out: &mut AclTensor,
) -> Result<()> {
    use aclnn_sys::common::AclDataType;

    let src_dtype = a.dtype();

    // If already FP32, just delegate to normal matmul
    if src_dtype == AclDataType::Float {
        return matmul(stream, a, b, out);
    }

    let a_numel = a.numel() as usize;
    let b_numel = b.numel() as usize;
    let out_numel = out.numel() as usize;

    // 1. Allocate FP32 buffers
    let a_fp32_buf = DeviceBuffer::alloc(a_numel * 4)?;
    let b_fp32_buf = DeviceBuffer::alloc(b_numel * 4)?;
    let out_fp32_buf = DeviceBuffer::alloc(out_numel * 4)?;

    // 2. Create FP32 tensor descriptors
    let mut a_fp32 = AclTensor::new(a.shape(), AclDataType::Float, &a_fp32_buf)?;
    let mut b_fp32 = AclTensor::new(b.shape(), AclDataType::Float, &b_fp32_buf)?;
    let mut out_fp32 = AclTensor::new(out.shape(), AclDataType::Float, &out_fp32_buf)?;

    // 3. Cast inputs: BF16/FP16 → FP32
    super::elementwise::cast(stream, a, AclDataType::Float, &mut a_fp32)?;
    super::elementwise::cast(stream, b, AclDataType::Float, &mut b_fp32)?;

    // 4. FP32 matmul
    matmul(stream, &a_fp32, &b_fp32, &mut out_fp32)?;

    // 5. Cast output: FP32 → original dtype
    super::elementwise::cast(stream, &out_fp32, src_dtype, out)?;

    // DeviceBuffers freed on drop (RAII)
    Ok(())
}

