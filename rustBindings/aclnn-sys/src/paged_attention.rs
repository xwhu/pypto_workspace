//! Paged Attention operators — Incremental FlashAttention for decode.
//!
//! `aclnnIncreFlashAttentionV4`: Decode-phase PagedAttention.
//! Uses paged KV cache with block_table for efficient memory access.
//!
//! Signature derived from CANN 8.5 SDK header:
//!   aclnn_incre_flash_attention_v4.h

use super::common::*;
use std::os::raw::c_void;

extern "C" {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // IncreFlashAttentionV4: Decode-phase PagedAttention.
    //
    // query:       [batch_size, 1, num_heads * head_dim]  FP16 (BSH layout)
    // key:         aclTensorList — list of K cache tensors (one per layer being used)
    // value:       aclTensorList — list of V cache tensors
    // blocktable:  [batch_size, max_blocks_per_seq]  INT32
    // actualSeqLengths: aclIntArray [batch_size] — total context length per seq
    //
    // Returns attention output: [batch_size, 1, num_heads * head_dim]  FP16
    //
    // Exact match to CANN 8.5.0 header: aclnn_incre_flash_attention_v4.h
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Stage 1: Get workspace size for IncreFlashAttentionV4.
    pub fn aclnnIncreFlashAttentionV4GetWorkspaceSize(
        query: *const AclTensor,
        key: *const AclTensorList,              // K cache (tensor list)
        value: *const AclTensorList,            // V cache (tensor list)
        pse_shift: *const AclTensor,            // null for no positional shift
        atten_mask: *const AclTensor,           // null for causal
        actual_seq_lengths: *const AclIntArray, // [batch_size] context lengths
        dequant_scale1: *const AclTensor,       // null (FP16)
        quant_scale1: *const AclTensor,         // null
        dequant_scale2: *const AclTensor,       // null
        quant_scale2: *const AclTensor,         // null
        quant_offset2: *const AclTensor,        // null
        antiquant_scale: *const AclTensor,      // null (FP16, not quantized)
        antiquant_offset: *const AclTensor,     // null
        blocktable: *const AclTensor,           // [batch, max_blocks_per_seq] INT32
        kv_padding_size: *const AclTensor,      // null
        num_heads: i64,
        scale_value: f64,
        input_layout: *const std::os::raw::c_char, // "BSH"
        num_key_value_heads: i64,
        block_size: i64,
        inner_precise: i64,              // 0
        attention_out: *const AclTensor, // output
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute IncreFlashAttentionV4.
    pub fn aclnnIncreFlashAttentionV4(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Helpers: aclTensorList creation/destruction
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Create an aclTensorList from an array of aclTensor pointers.
    pub fn aclCreateTensorList(tensors: *const *const AclTensor, num: u64) -> *mut AclTensorList;

    /// Destroy an aclTensorList.
    pub fn aclDestroyTensorList(list: *const AclTensorList) -> AclnnStatus;
}
