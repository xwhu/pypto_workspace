//! Safe wrappers for Paged Attention KV Cache operators.
//!
//! Provides the core decode-phase paged attention function:
//! `paged_attention_decode` — uses IncreFlashAttentionV4 with paged KV cache

use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::{
    AclOpExecutor,
    AclTensorList,
    AclIntArray,
    AclDataType,
};

/// Perform decode-phase Paged Attention using IncreFlashAttentionV4.
///
/// # Arguments
/// - `stream`: execution stream
/// - `query`: Q tensor [batch_size, 1, num_heads * head_dim] FP16
/// - `key_cache`: K cache tensor [num_blocks, block_size, num_kv_heads, head_dim] FP16
/// - `value_cache`: V cache tensor [num_blocks, block_size, num_kv_heads, head_dim] FP16
/// - `actual_seq_lengths`: raw pointer to int array of context lengths
/// - `num_heads`: number of Q heads
/// - `num_kv_heads`: number of KV heads (GQA)
/// - `scale`: 1/sqrt(head_dim)
/// - `block_size`: tokens per block (e.g., 16)
/// - `block_table`: [batch_size, max_blocks_per_seq] INT32
/// - `attention_out`: output tensor [batch_size, 1, num_heads * head_dim] FP16
pub fn paged_attention_decode(
    stream: &Stream,
    query: &AclTensor,
    key_cache: &AclTensor,
    value_cache: &AclTensor,
    actual_seq_lengths: *const AclIntArray,
    num_heads: i64,
    num_kv_heads: i64,
    scale: f64,
    block_size: i64,
    block_table: &AclTensor,
    attention_out: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    let layout_cstr = std::ffi::CString::new("BSH")
        .map_err(|_| crate::error::AscendError::InvalidArgument(
            "invalid layout string".to_string(),
        ))?;

    // Create single-element tensor lists for key and value cache.
    //
    // IMPORTANT: We do NOT call aclDestroyTensorList later, because
    // aclDestroyTensorList also destroys the individual tensors inside it.
    // Our Rust AclTensor wrappers (key_cache, value_cache) already call
    // aclDestroyTensor on Drop. Calling both would cause double-free →
    // CANN heap corruption → 561103 (ACLNN_ERR_INNER_NULLPTR) in
    // subsequent operator calls.
    //
    // The list container itself is a tiny metadata struct — leaking it is
    // harmless compared to corrupting CANN's internal state.
    let key_ptr = key_cache.raw();
    let value_ptr = value_cache.raw();

    let key_list = unsafe {
        aclnn_sys::paged_attention::aclCreateTensorList(
            &key_ptr as *const _ as *const *const aclnn_sys::common::AclTensor,
            1,
        )
    };
    let value_list = unsafe {
        aclnn_sys::paged_attention::aclCreateTensorList(
            &value_ptr as *const _ as *const *const aclnn_sys::common::AclTensor,
            1,
        )
    };

    // Stage 1: Get workspace size
    let status = unsafe {
        aclnn_sys::paged_attention::aclnnIncreFlashAttentionV4GetWorkspaceSize(
            query.raw(),
            key_list,
            value_list,
            std::ptr::null(),       // pseShift: null
            std::ptr::null(),       // attenMask: null for causal
            actual_seq_lengths,
            std::ptr::null(),       // dequantScale1: null (FP16)
            std::ptr::null(),       // quantScale1: null
            std::ptr::null(),       // dequantScale2: null
            std::ptr::null(),       // quantScale2: null
            std::ptr::null(),       // quantOffset2: null
            std::ptr::null(),       // antiquantScale: null
            std::ptr::null(),       // antiquantOffset: null
            block_table.raw(),      // blocktable: block table INT32
            std::ptr::null(),       // kvPaddingSize: null
            num_heads,
            scale,
            layout_cstr.as_ptr() as *const std::os::raw::c_char,
            num_kv_heads,
            block_size,
            0,                      // innerPrecise: default
            attention_out.raw(),
            &mut workspace_size,
            &mut executor,
        )
    };
    check_aclnn(status)?;

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
        aclnn_sys::paged_attention::aclnnIncreFlashAttentionV4(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    // NOTE: We intentionally do NOT call aclDestroyTensorList here.
    // See comment above about double-free prevention.

    // Sync to ensure attention output is ready
    stream.synchronize()?;

    Ok(())
}
