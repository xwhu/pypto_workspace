//! Incremental Flash Attention V4 — decode-phase paged attention.
//!
//! C API: `aclnnop/aclnn_incre_flash_attention_v4.h`
//! Supports paged KV cache via `blocktable` + `blockSize`.

use std::os::raw::c_void;
use super::common::*;

extern "C" {
    /// Stage 1: Compute workspace for incremental flash attention V4.
    ///
    /// - `query`:            [B, 1, N*D] or [B, 1, N, D] (current decode token)
    /// - `key`:              aclTensorList — per-layer K cache tensors
    /// - `value`:            aclTensorList — per-layer V cache tensors
    /// - `pse_shift`:        optional positional shift tensor (null if unused)
    /// - `atten_mask`:       optional attention mask (null for causal default)
    /// - `actual_seq_lengths`: optional per-batch actual sequence lengths
    /// - `dequant_scale1..quantOffset2`: quantization params (null for fp16/bf16)
    /// - `antiquant_scale/offset`: anti-quantization params (null)
    /// - `blocktable`:       [B, max_blocks] int32 — logical→physical block mapping
    /// - `kv_padding_size`:  optional padding (null)
    /// - `num_heads`:        number of Q heads
    /// - `scale_value`:      1/sqrt(head_dim)
    /// - `input_layout`:     "BSH" or "BNSD"
    /// - `num_key_value_heads`: number of KV heads (for GQA)
    /// - `block_size`:       tokens per block (e.g. 16)
    /// - `inner_precise`:    0 = default
    /// - `attention_out`:    output tensor, same layout as query
    pub fn aclnnIncreFlashAttentionV4GetWorkspaceSize(
        query: *const AclTensor,
        key: *const AclTensorList,
        value: *const AclTensorList,
        pse_shift: *const AclTensor,
        atten_mask: *const AclTensor,
        actual_seq_lengths: *const AclIntArray,
        dequant_scale1: *const AclTensor,
        quant_scale1: *const AclTensor,
        dequant_scale2: *const AclTensor,
        quant_scale2: *const AclTensor,
        quant_offset2: *const AclTensor,
        antiquant_scale: *const AclTensor,
        antiquant_offset: *const AclTensor,
        blocktable: *const AclTensor,
        kv_padding_size: *const AclTensor,
        num_heads: i64,
        scale_value: f64,
        input_layout: *const std::os::raw::c_char,
        num_key_value_heads: i64,
        block_size: i64,
        inner_precise: i64,
        attention_out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute incremental flash attention V4.
    pub fn aclnnIncreFlashAttentionV4(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
