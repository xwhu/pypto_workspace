//! Safe wrappers for aclnn operators.
//!
//! Each function encapsulates the two-stage aclnn pattern:
//! 1. GetWorkspaceSize → allocate workspace
//! 2. Execute with workspace + stream
//!
//! All functions are synchronous from the caller's perspective
//! (they enqueue work on the stream but don't wait for completion).
//! Call `stream.synchronize()` to wait.

pub mod matmul;
pub mod rmsnorm;
pub mod embedding;
pub mod activation;
pub mod elementwise;
pub mod reduction;
pub mod rope;
pub mod attention;
pub mod permute;
pub mod paged_attention;

