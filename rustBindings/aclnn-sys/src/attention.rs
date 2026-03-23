//! Attention operators — GQA and FlashAttention.
//!
//! C API: `aclnn/aclnn_flash_attention_score.h` and related
//! Supports Grouped-Query Attention (GQA) and Flash Attention variants.

use std::os::raw::c_void;
use super::common::*;

// ─── Flash Attention Score (Prefill) ───────────────────────────────────

extern "C" {
    /// Stage 1: Flash Attention forward workspace.
    ///
    /// Used for prefill / prompt processing.
    ///
    /// - `query`: [batch, num_heads, seq_q, head_dim]
    /// - `key`: [batch, num_kv_heads, seq_kv, head_dim]
    /// - `value`: [batch, num_kv_heads, seq_kv, head_dim]
    /// - `real_shift`: optional, attention bias (null if none)
    /// - `drop_mask`: optional, dropout mask (null if none)
    /// - `padding_mask`: optional (null if none)
    /// - `atten_mask`: optional, causal mask (null for causal-by-default)
    /// - `prefix`: optional prefix lengths (null if none)
    /// - `scale_value`: softmax scale (typically 1/sqrt(head_dim))
    /// - `keep_prob`: dropout keep probability (1.0 = no dropout)
    /// - `pre_tokens`: number of pre-context tokens
    /// - `next_tokens`: number of next-context tokens
    /// - `head_num`: number of query heads
    /// - `input_layout`: "BSH", "BNSD", etc.
    /// - `inner_precise`: precision config (0 = default)
    /// - `sparse_mode`: sparsity mode (0 = dense)
    /// - `softmax_max`: output softmax max (for numerical stability)
    /// - `softmax_sum`: output softmax sum
    /// - `softmax_out`: output softmax values (optional)
    /// - `attention_out`: output attention [batch, num_heads, seq_q, head_dim]
    pub fn aclnnFlashAttentionScoreGetWorkspaceSize(
        query: *const AclTensor,
        key: *const AclTensor,
        value: *const AclTensor,
        real_shift: *const AclTensor,
        drop_mask: *const AclTensor,
        padding_mask: *const AclTensor,
        atten_mask: *const AclTensor,
        prefix: *const AclIntArray,
        scale_value: f64,
        keep_prob: f64,
        pre_tokens: i64,
        next_tokens: i64,
        head_num: i64,
        input_layout: *const std::os::raw::c_char,
        inner_precise: i32,
        sparse_mode: i32,
        softmax_max: *const AclTensor,
        softmax_sum: *const AclTensor,
        softmax_out: *const AclTensor,
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

// ─── Flash Attention V2 Score (Incremental Decode) ─────────────────────

extern "C" {
    /// Stage 1: Incremental (decode) flash attention workspace.
    ///
    /// Used for single-token decode steps with KV cache.
    ///
    /// - `query`: [batch, 1, num_heads, head_dim]
    /// - `key`: [batch, seq_kv, num_kv_heads, head_dim]
    /// - `value`: [batch, seq_kv, num_kv_heads, head_dim]
    /// - `atten_mask`: causal mask (optional)
    /// - `scale`: softmax scale
    /// - `attention_out`: output [batch, 1, num_heads, head_dim]
    pub fn aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
        query: *const AclTensor,
        key: *const AclTensor,
        value: *const AclTensor,
        padding_mask: *const AclTensor,
        atten_mask: *const AclTensor,
        prefix: *const AclIntArray,
        scale_value: f64,
        keep_prob: f64,
        pre_tokens: i64,
        next_tokens: i64,
        head_num: i64,
        input_layout: *const std::os::raw::c_char,
        inner_precise: i32,
        sparse_mode: i32,
        softmax_max: *const AclTensor,
        softmax_sum: *const AclTensor,
        softmax_out: *const AclTensor,
        attention_out: *const AclTensor,
        actual_seq_q_len: *const AclIntArray,
        actual_seq_kv_len: *const AclIntArray,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;

    /// Stage 2: Execute variable-length flash attention.
    pub fn aclnnFlashAttentionVarLenScore(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
