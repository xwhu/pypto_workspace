//! Safe Incremental Flash Attention V4 wrapper (decode-phase paged attention).
//!
//! Calls aclnnIncreFlashAttentionV4 for single-token decode with paged KV cache.

use std::os::raw::c_void;
use crate::error::{check_aclnn, Result};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::{
    AclOpExecutor, AclTensorList as RawAclTensorList,
    AclTensor as RawAclTensor, AclIntArray as RawAclIntArray,
};

// ─── AclTensorList RAII wrapper ────────────────────────────────────────

/// Safe RAII wrapper for `aclTensorList`.
///
/// Used to pass per-layer K/V cache tensors to `aclnnIncreFlashAttentionV4`.
/// The tensor list is destroyed on drop.
pub struct AclTensorListGuard {
    raw: *mut RawAclTensorList,
    _ptrs: Vec<*const RawAclTensor>,
}

impl AclTensorListGuard {
    /// Create a tensor list from a slice of `AclTensor` references.
    ///
    /// The `AclTensor` descriptors must outlive this list (they are not owned).
    pub fn new(tensors: &[&AclTensor]) -> Result<Self> {
        let ptrs: Vec<*const RawAclTensor> = tensors.iter().map(|t| t.raw()).collect();
        let raw = unsafe {
            aclnn_sys::common::aclCreateTensorList(
                ptrs.as_ptr(),
                ptrs.len() as u64,
            )
        };
        if raw.is_null() {
            return Err(crate::error::AscendError::InvalidArgument(
                "aclCreateTensorList returned null".to_string(),
            ));
        }
        Ok(Self { raw, _ptrs: ptrs })
    }

    /// Get the raw pointer for passing to FFI.
    pub fn raw(&self) -> *const RawAclTensorList {
        self.raw
    }
}

impl Drop for AclTensorListGuard {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = aclnn_sys::common::aclDestroyTensorList(self.raw);
            }
        }
    }
}

// ─── AclIntArrayGuard RAII wrapper ─────────────────────────────────────

/// Safe RAII wrapper for `aclIntArray`.
pub struct AclIntArrayGuard {
    raw: *mut RawAclIntArray,
    _values: Vec<i64>,
}

impl AclIntArrayGuard {
    /// Create an int array from a slice.
    pub fn new(values: &[i64]) -> Result<Self> {
        let values_vec = values.to_vec();
        let raw = unsafe {
            aclnn_sys::common::aclCreateIntArray(
                values_vec.as_ptr(),
                values_vec.len() as u64,
            )
        };
        if raw.is_null() {
            return Err(crate::error::AscendError::InvalidArgument(
                "aclCreateIntArray returned null".to_string(),
            ));
        }
        Ok(Self { raw, _values: values_vec })
    }

    /// Get the raw pointer.
    pub fn raw(&self) -> *const RawAclIntArray {
        self.raw
    }
}

impl Drop for AclIntArrayGuard {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = aclnn_sys::common::aclDestroyIntArray(self.raw);
            }
        }
    }
}

// ─── Incremental Flash Attention V4 ────────────────────────────────────

/// Incremental Flash Attention V4 (decode-phase paged attention).
///
/// Computes single-token attention against a paged KV cache.
///
/// # Arguments
/// - `stream`: execution stream
/// - `query`: Q tensor for current decode token, layout per `input_layout`
/// - `key_cache`: per-layer K cache tensors (one AclTensor per layer)
/// - `value_cache`: per-layer V cache tensors (one AclTensor per layer)
/// - `block_table`: [batch, max_blocks] int32 tensor — logical→physical block mapping
/// - `actual_seq_lengths`: per-batch actual sequence lengths (including current token)
/// - `num_heads`: number of Q attention heads
/// - `num_kv_heads`: number of KV heads (for GQA; set equal to num_heads for MHA)
/// - `scale`: 1/sqrt(head_dim)
/// - `block_size`: tokens per cache block (e.g. 16)
/// - `input_layout`: "BSH" or "BNSD"
/// - `attention_out`: output tensor, same shape as query
pub fn incre_flash_attention_v4(
    stream: &Stream,
    query: &AclTensor,
    key_cache: &AclTensorListGuard,
    value_cache: &AclTensorListGuard,
    block_table: &AclTensor,
    actual_seq_lengths: &AclIntArrayGuard,
    num_heads: i64,
    num_kv_heads: i64,
    scale: f64,
    block_size: i64,
    input_layout: &str,
    attention_out: &mut AclTensor,
) -> Result<Option<DeviceBuffer>> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    let layout_cstr = std::ffi::CString::new(input_layout)
        .map_err(|_| crate::error::AscendError::InvalidArgument(
            "invalid input_layout string".to_string(),
        ))?;

    // Stage 1: Get workspace size
    check_aclnn(unsafe {
        aclnn_sys::incre_attention::aclnnIncreFlashAttentionV4GetWorkspaceSize(
            query.raw(),
            key_cache.raw(),
            value_cache.raw(),
            std::ptr::null(),       // pseShift: none
            std::ptr::null(),       // attenMask: null = causal
            actual_seq_lengths.raw(),
            std::ptr::null(),       // dequantScale1: none (fp16)
            std::ptr::null(),       // quantScale1: none
            std::ptr::null(),       // dequantScale2: none
            std::ptr::null(),       // quantScale2: none
            std::ptr::null(),       // quantOffset2: none
            std::ptr::null(),       // antiquantScale: none
            std::ptr::null(),       // antiquantOffset: none
            block_table.raw(),
            std::ptr::null(),       // kvPaddingSize: none
            num_heads,
            scale,
            layout_cstr.as_ptr() as *mut std::os::raw::c_char,
            num_kv_heads,
            block_size,
            0,                      // innerPrecise: default
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
        aclnn_sys::incre_attention::aclnnIncreFlashAttentionV4(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    Ok(workspace)
}
