//! AscendCL runtime FFI — device, stream, context management.
//!
//! Corresponds to functions in `acl/acl.h` and `acl/acl_rt.h`.

use std::os::raw::{c_char, c_int};
use super::types::*;

extern "C" {
    // ─── SDK Lifecycle ──────────────────────────────────────────────

    /// Initialize AscendCL. Must be called before any other ACL function.
    /// `config_path` can be null for default config.
    pub fn aclInit(config_path: *const c_char) -> AclError;

    /// Finalize AscendCL. Call once at program exit.
    pub fn aclFinalize() -> AclError;

    // ─── Device Management ──────────────────────────────────────────

    /// Set the current device.
    pub fn aclrtSetDevice(device_id: i32) -> AclError;

    /// Reset a device, freeing all its resources.
    pub fn aclrtResetDevice(device_id: i32) -> AclError;

    /// Get the current device ID.
    pub fn aclrtGetDevice(device_id: *mut i32) -> AclError;

    /// Get the number of devices.
    pub fn aclrtGetDeviceCount(count: *mut u32) -> AclError;

    /// Get the run mode (host or device).
    pub fn aclrtGetRunMode(run_mode: *mut AclrtRunMode) -> AclError;

    // ─── Context Management ─────────────────────────────────────────

    /// Create a context on the given device.
    pub fn aclrtCreateContext(context: *mut AclrtContext, device_id: i32) -> AclError;

    /// Destroy a context.
    pub fn aclrtDestroyContext(context: AclrtContext) -> AclError;

    /// Set the current context.
    pub fn aclrtSetCurrentContext(context: AclrtContext) -> AclError;

    /// Get the current context.
    pub fn aclrtGetCurrentContext(context: *mut AclrtContext) -> AclError;

    // ─── Stream Management ──────────────────────────────────────────

    /// Create a stream on the current device.
    pub fn aclrtCreateStream(stream: *mut AclrtStream) -> AclError;

    /// Destroy a stream.
    pub fn aclrtDestroyStream(stream: AclrtStream) -> AclError;

    /// Block until all tasks in the stream are complete.
    pub fn aclrtSynchronizeStream(stream: AclrtStream) -> AclError;

    // ─── Event Management ───────────────────────────────────────────

    /// Create an event.
    pub fn aclrtCreateEvent(event: *mut AclrtEvent) -> AclError;

    /// Destroy an event.
    pub fn aclrtDestroyEvent(event: AclrtEvent) -> AclError;

    /// Record an event on a stream.
    pub fn aclrtRecordEvent(event: AclrtEvent, stream: AclrtStream) -> AclError;

    /// Block the host until the event is complete.
    pub fn aclrtSynchronizeEvent(event: AclrtEvent) -> AclError;

    /// Query whether an event has completed. Returns ACL_SUCCESS if done.
    pub fn aclrtQueryEvent(event: AclrtEvent, status: *mut c_int) -> AclError;

    /// Elapsed time in milliseconds between two events.
    pub fn aclrtEventElapsedTime(time_ms: *mut f32, start: AclrtEvent, end: AclrtEvent) -> AclError;
}
