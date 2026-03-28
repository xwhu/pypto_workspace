//! Device management with RAII.
//!
//! Handles AscendCL initialization/finalization and device selection.

use crate::error::{check_acl, Result};
use std::sync::atomic::{AtomicBool, Ordering};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// A Send-safe wrapper around an `AclrtContext` handle.
///
/// `AclrtContext = *mut c_void` is not `Send` by default, but CANN context
/// handles are explicitly designed to be passed across threads via
/// `aclrtSetCurrentContext`. This newtype asserts that it is safe to do so.
#[derive(Clone, Copy)]
pub struct AclContext(pub ascendcl_sys::AclrtContext);

// SAFETY: CANN context handles are designed to be shared across threads.
unsafe impl Send for AclContext {}
unsafe impl Sync for AclContext {}

/// RAII guard for AscendCL initialization and device selection.
///
/// Calls `aclInit` on creation and `aclFinalize` on drop.
/// Only one `DeviceGuard` should exist at a time.
///
/// # Example
/// ```no_run
/// let _guard = ascend::Device::init(0).expect("Failed to init device");
/// // ... use device ...
/// // aclFinalize called on drop
/// ```
pub struct Device {
    device_id: i32,
}

impl Device {
    /// Initialize AscendCL and set the current device.
    pub fn init(device_id: i32) -> Result<Self> {
        if INITIALIZED.swap(true, Ordering::SeqCst) {
            // Already initialized — just set device
            check_acl(unsafe { ascendcl_sys::aclrtSetDevice(device_id) })?;
            return Ok(Self { device_id });
        }

        check_acl(unsafe { ascendcl_sys::aclInit(std::ptr::null()) })?;
        check_acl(unsafe { ascendcl_sys::aclrtSetDevice(device_id) })?;

        Ok(Self { device_id })
    }

    /// Initialize from `ASCEND_DEVICE_ID` environment variable.
    ///
    /// Falls back to device 0 if the variable is not set.
    ///
    /// # Example
    /// ```bash
    /// ASCEND_DEVICE_ID=2 cargo test
    /// ```
    pub fn from_env() -> Result<Self> {
        let device_id = std::env::var("ASCEND_DEVICE_ID")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        Self::init(device_id)
    }

    /// Get the device ID.
    pub fn id(&self) -> i32 {
        self.device_id
    }

    /// Get the number of available devices.
    pub fn count() -> Result<u32> {
        let mut count: u32 = 0;
        check_acl(unsafe { ascendcl_sys::aclrtGetDeviceCount(&mut count) })?;
        Ok(count)
    }

    /// Get free and total device memory in bytes.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        check_acl(unsafe {
            ascendcl_sys::aclrtGetMemInfo(
                ascendcl_sys::AclrtMemMallocPolicy::Normal,
                &mut free,
                &mut total,
            )
        })?;
        Ok((free, total))
    }
    /// Capture the current thread's ACL context handle.
    ///
    /// Call this on the main/initializing thread after `Device::init()`.
    /// The returned pointer can be passed to worker threads which should call
    /// `set_current_context()` instead of `aclrtSetDevice()`.
    ///
    /// In CANN, `aclrtSetDevice` can only be called once per process per device.
    /// Worker threads must use `aclrtSetCurrentContext` to share the context.
    pub fn get_current_context() -> Result<AclContext> {
        let mut ctx: ascendcl_sys::AclrtContext = std::ptr::null_mut();
        check_acl(unsafe { ascendcl_sys::aclrtGetCurrentContext(&mut ctx) })?;
        Ok(AclContext(ctx))
    }

    /// Bind an existing ACL context to the current thread.
    ///
    /// Call this at the start of every worker thread before any device operations.
    /// The context must have been captured from the main thread via `get_current_context()`.
    pub fn set_current_context(ctx: AclContext) -> Result<()> {
        check_acl(unsafe { ascendcl_sys::aclrtSetCurrentContext(ctx.0) })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = ascendcl_sys::aclrtResetDevice(self.device_id);
            if INITIALIZED.swap(false, Ordering::SeqCst) {
                let _ = ascendcl_sys::aclFinalize();
            }
        }
    }
}

// Safety: Device pointers can be sent between threads.
unsafe impl Send for Device {}
