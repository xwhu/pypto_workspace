//! AscendCL memory management FFI.
//!
//! Corresponds to functions in `acl/acl_rt.h` (memory section).

use std::os::raw::c_void;
use super::types::*;

extern "C" {
    // ─── Device Memory ──────────────────────────────────────────────

    /// Allocate device memory.
    ///
    /// - `dev_ptr`: output pointer to allocated device memory
    /// - `size`: bytes to allocate
    /// - `policy`: allocation policy (use `Normal` for most cases)
    pub fn aclrtMalloc(
        dev_ptr: *mut *mut c_void,
        size: usize,
        policy: AclrtMemMallocPolicy,
    ) -> AclError;

    /// Free device memory.
    pub fn aclrtFree(dev_ptr: *mut c_void) -> AclError;

    /// Allocate host memory (pinned, for efficient DMA).
    pub fn aclrtMallocHost(host_ptr: *mut *mut c_void, size: usize) -> AclError;

    /// Free host memory allocated by `aclrtMallocHost`.
    pub fn aclrtFreeHost(host_ptr: *mut c_void) -> AclError;

    // ─── Memory Copy ────────────────────────────────────────────────

    /// Synchronous memory copy.
    ///
    /// - `dst`: destination pointer
    /// - `dst_max`: max bytes at destination
    /// - `src`: source pointer
    /// - `count`: bytes to copy
    /// - `kind`: copy direction
    pub fn aclrtMemcpy(
        dst: *mut c_void,
        dst_max: usize,
        src: *const c_void,
        count: usize,
        kind: AclrtMemcpyKind,
    ) -> AclError;

    /// Asynchronous memory copy on a stream.
    pub fn aclrtMemcpyAsync(
        dst: *mut c_void,
        dst_max: usize,
        src: *const c_void,
        count: usize,
        kind: AclrtMemcpyKind,
        stream: AclrtStream,
    ) -> AclError;

    // ─── Memory Set ─────────────────────────────────────────────────

    /// Synchronous memset on device memory.
    pub fn aclrtMemset(
        dev_ptr: *mut c_void,
        max_count: usize,
        value: u8,
        count: usize,
    ) -> AclError;

    /// Asynchronous memset on a stream.
    pub fn aclrtMemsetAsync(
        dev_ptr: *mut c_void,
        max_count: usize,
        value: u8,
        count: usize,
        stream: AclrtStream,
    ) -> AclError;

    // ─── Memory Query ───────────────────────────────────────────────

    /// Get free and total device memory in bytes.
    pub fn aclrtGetMemInfo(
        mem_attr: AclrtMemMallocPolicy,
        free: *mut usize,
        total: *mut usize,
    ) -> AclError;
}
