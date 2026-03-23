//! Device management with RAII.
//!
//! Handles AscendCL initialization/finalization and device selection.

use crate::error::{check_acl, Result};
use std::sync::atomic::{AtomicBool, Ordering};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

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
