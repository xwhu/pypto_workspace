//! `aclnn-sys` — Raw FFI bindings to CANN aclnn operator APIs.
//!
//! Provides unsafe `extern "C"` declarations for `libopapi.so` functions.
//! Each operator follows the two-stage pattern:
//! 1. `aclnnXxxGetWorkspaceSize(...)` — compute workspace requirements
//! 2. `aclnnXxx(workspace, size, executor, stream)` — execute the operator
//!
//! # Modules
//! - [`common`] — Opaque types (AclTensor, AclScalar, AclOpExecutor) and tensor APIs
//! - [`matmul`] — Matrix multiplication
//! - [`rmsnorm`] — RMS Layer Normalization
//! - [`embedding`] — Token embedding lookup
//! - [`rope`] — Rotary Position Embedding
//! - [`attention`] — Flash Attention (prefill + decode)
//! - [`activation`] — SiLU, SwiGLU
//! - [`elementwise`] — Add, Mul
//! - [`reduction`] — Softmax, ArgMax

pub mod common;
pub mod matmul;
pub mod rmsnorm;
pub mod embedding;
pub mod rope;
pub mod attention;
pub mod activation;
pub mod elementwise;
pub mod reduction;
pub mod permute;

// Re-export common types
pub use common::*;
