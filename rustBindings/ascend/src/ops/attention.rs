//! Safe FlashAttentionScore wrapper.
//!
//! Calls aclnnFlashAttentionScore V1 for prefill attention.

use std::os::raw::c_void;
use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;

/// Flash Attention Score (prefill).
///
/// # Arguments
/// - `stream`: execution stream
/// - `query`: Q tensor in layout specified by `input_layout`
/// - `key`: K tensor
/// - `value`: V tensor
/// - `scale`: 1/sqrt(head_dim)
/// - `head_num`: number of Q heads (for GQA, > num_kv_heads)
/// - `input_layout`: "BSH" or "BNSD"
/// - `softmax_max`: auxiliary output [B, N, S, 8] Float32
/// - `softmax_sum`: auxiliary output [B, N, S, 8] Float32
/// - `attention_out`: output attention result, same layout as Q
pub fn flash_attention_score(
    stream: &Stream,
    query: &AclTensor,
    key: &AclTensor,
    value: &AclTensor,
    scale: f64,
    head_num: i64,
    input_layout: &str,
    seq_len: i64,
    softmax_max: &AclTensor,
    softmax_sum: &AclTensor,
    attention_out: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Null-terminate the layout string
    let layout_cstr = std::ffi::CString::new(input_layout)
        .map_err(|_| crate::error::AscendError::InvalidArgument(
            "invalid input_layout string".to_string(),
        ))?;

    // Stage 1: Get workspace size
    check_aclnn(unsafe {
        aclnn_sys::attention::aclnnFlashAttentionScoreGetWorkspaceSize(
            query.raw(),
            key.raw(),
            value.raw(),
            std::ptr::null(),       // realShift: none
            std::ptr::null(),       // dropMask: none
            std::ptr::null(),       // paddingMask: none
            std::ptr::null(),       // attenMask: null = auto causal
            std::ptr::null(),       // prefix: none
            scale,
            1.0,                    // keepProb: no dropout
            seq_len,               // preTokens: full causal window
            0,                      // nextTokens: 0 for causal
            head_num,
            layout_cstr.as_ptr() as *mut std::os::raw::c_char,
            0,                      // innerPrecise: default
            2,                      // sparseMode: 2 = defaultMask (causal)
            softmax_max.raw(),
            softmax_sum.raw(),
            std::ptr::null(),       // softmaxOut: not needed
            attention_out.raw(),
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
        aclnn_sys::attention::aclnnFlashAttentionScore(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    Ok(())
}

/// Flash Attention Score with explicit causal mask (prefill).
///
/// Same as `flash_attention_score` but takes an explicit bool attenMask
/// tensor (true = masked/don't attend). This is more robust than relying
/// on sparseMode for causal masking.
pub fn flash_attention_score_with_mask(
    stream: &Stream,
    query: &AclTensor,
    key: &AclTensor,
    value: &AclTensor,
    atten_mask: &AclTensor,
    scale: f64,
    head_num: i64,
    input_layout: &str,
    seq_len: i64,
    softmax_max: &AclTensor,
    softmax_sum: &AclTensor,
    attention_out: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    let layout_cstr = std::ffi::CString::new(input_layout)
        .map_err(|_| crate::error::AscendError::InvalidArgument(
            "invalid input_layout string".to_string(),
        ))?;

    // Stage 1: Get workspace size
    check_aclnn(unsafe {
        aclnn_sys::attention::aclnnFlashAttentionScoreGetWorkspaceSize(
            query.raw(),
            key.raw(),
            value.raw(),
            std::ptr::null(),       // realShift: none
            std::ptr::null(),       // dropMask: none
            std::ptr::null(),       // paddingMask: none
            atten_mask.raw(),       // explicit bool causal mask
            std::ptr::null(),       // prefix: none
            scale,
            1.0,                    // keepProb: no dropout
            seq_len,               // preTokens: full causal window
            0,                      // nextTokens: 0 for causal
            head_num,
            layout_cstr.as_ptr() as *mut std::os::raw::c_char,
            0,                      // innerPrecise: default
            0,                      // sparseMode: 0 = dense (mask provided)
            softmax_max.raw(),
            softmax_sum.raw(),
            std::ptr::null(),       // softmaxOut: not needed
            attention_out.raw(),
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
        aclnn_sys::attention::aclnnFlashAttentionScore(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    Ok(())
}
