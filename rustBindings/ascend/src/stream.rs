//! Stream management with RAII.

use crate::error::{check_acl, Result};
use ascendcl_sys::AclrtStream;

/// RAII wrapper around `aclrtStream`.
///
/// Creates a stream on construction, destroys on drop.
pub struct Stream {
    raw: AclrtStream,
}

impl Stream {
    /// Create a new stream on the current device.
    pub fn new() -> Result<Self> {
        let mut raw: AclrtStream = std::ptr::null_mut();
        check_acl(unsafe { ascendcl_sys::aclrtCreateStream(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Block the host until all tasks on this stream are complete.
    pub fn synchronize(&self) -> Result<()> {
        check_acl(unsafe { ascendcl_sys::aclrtSynchronizeStream(self.raw) })
    }

    /// Get the raw stream handle (for passing to aclnn operators).
    pub fn raw(&self) -> AclrtStream {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = ascendcl_sys::aclrtDestroyStream(self.raw);
            }
        }
    }
}

// Safety: Streams can be sent between threads (actual synchronization
// is handled by the AscendCL runtime).
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}
