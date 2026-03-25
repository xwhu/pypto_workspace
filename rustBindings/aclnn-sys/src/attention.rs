//! Attention operators — GQA and FlashAttention.
//!
//! C API: `aclnn/aclnn_flash_attention_score.h`
//! aclnnFlashAttentionScore — V1 prefill flash attention.

use super::common::*;
use std::os::raw::c_void;

extern "C" {
    /// Stage 1: Flash Attention forward workspace (V1, prefill).
    ///
    /// - `query`: [B, N, S, D] or [B, S, N, D] depending on inputLayout
    /// - `key`: [B, N_kv, S, D] or [B, S, N_kv, D]
    /// - `value`: same layout as key
    /// - Optional params: pass null for unused
    /// - `scaleValue`: 1/sqrt(head_dim)
    /// - `keepProb`: 1.0 (no dropout)
    /// - `preTokens`: seq_len (for causal mask)
    /// - `nextTokens`: 0 (causal: can't attend to future)
    /// - `headNum`: number of Q heads
    /// - `inputLayout`: "BSH" or "BNSD"
    /// - `innerPrecise`: 0
    /// - `sparseMode`: 0 (dense)
    /// - Outputs: softmaxMax, softmaxSum, softmaxOut (aux), attentionOut
    pub fn aclnnFlashAttentionScoreGetWorkspaceSize(
        query: *const AclTensor,
        key: *const AclTensor,
        value: *const AclTensor,
        real_shift: *const AclTensor,   // null
        drop_mask: *const AclTensor,    // null
        padding_mask: *const AclTensor, // null
        atten_mask: *const AclTensor,   // null for auto-causal
        prefix: *const AclIntArray,     // null
        scale_value: f64,
        keep_prob: f64,
        pre_tokens: i64,
        next_tokens: i64,
        head_num: i64,
        input_layout: *const std::os::raw::c_char,
        inner_precise: i64,
        sparse_mode: i64,
        softmax_max: *const AclTensor,
        softmax_sum: *const AclTensor,
        softmax_out: *const AclTensor, // can be null
        attention_out: *const AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute FlashAttentionScore.
    pub fn aclnnFlashAttentionScore(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
