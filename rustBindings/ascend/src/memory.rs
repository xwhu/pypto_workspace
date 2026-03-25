//! Device memory management with RAII.

use crate::error::{check_acl, AscendError, Result};
use ascendcl_sys::{AclrtMemMallocPolicy, AclrtMemcpyKind};
use std::os::raw::c_void;

/// RAII wrapper around device memory allocated via `aclrtMalloc`.
///
/// Frees the memory automatically on drop (if `owned` is true).
pub struct DeviceBuffer {
    ptr: *mut c_void,
    size: usize,
    /// If true, `aclrtFree` is called on drop. If false, this is a
    /// non-owning view and the memory is NOT freed.
    owned: bool,
}

impl DeviceBuffer {
    /// Allocate device memory of the given size in bytes.
    pub fn alloc(size: usize) -> Result<Self> {
        if size == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                size: 0,
                owned: true,
            });
        }

        let mut ptr: *mut c_void = std::ptr::null_mut();
        check_acl(unsafe {
            ascendcl_sys::aclrtMalloc(&mut ptr, size, AclrtMemMallocPolicy::Normal)
        })?;

        Ok(Self {
            ptr,
            size,
            owned: true,
        })
    }

    /// Create a non-owning view of existing device memory.
    ///
    /// SAFETY: The caller must ensure the pointer remains valid for the
    /// lifetime of this DeviceBuffer. The memory will NOT be freed on drop.
    /// Used for weight tensors whose memory is owned by the model's Tensors.
    pub unsafe fn from_raw_non_owning(ptr: *mut c_void, size: usize) -> Self {
        Self {
            ptr,
            size,
            owned: false,
        }
    }

    /// Copy data from host to this device buffer.
    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(AscendError::InvalidArgument(format!(
                "host data ({} bytes) exceeds device buffer ({} bytes)",
                data.len(),
                self.size
            )));
        }
        check_acl(unsafe {
            ascendcl_sys::aclrtMemcpy(
                self.ptr,
                self.size,
                data.as_ptr() as *const c_void,
                data.len(),
                AclrtMemcpyKind::HostToDevice,
            )
        })
    }

    /// Copy data from this device buffer to host.
    pub fn copy_to_host(&self, buf: &mut [u8]) -> Result<()> {
        if buf.len() > self.size {
            return Err(AscendError::InvalidArgument(format!(
                "host buffer ({} bytes) exceeds device buffer ({} bytes)",
                buf.len(),
                self.size
            )));
        }
        check_acl(unsafe {
            ascendcl_sys::aclrtMemcpy(
                buf.as_mut_ptr() as *mut c_void,
                buf.len(),
                self.ptr,
                buf.len(),
                AclrtMemcpyKind::DeviceToHost,
            )
        })
    }

    /// Zero-fill the device buffer.
    pub fn memset_zero(&mut self) -> Result<()> {
        if self.size == 0 {
            return Ok(());
        }
        check_acl(unsafe { ascendcl_sys::aclrtMemset(self.ptr, self.size, 0, self.size) })
    }

    /// Get the raw device pointer.
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Get the buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe {
                let _ = ascendcl_sys::aclrtFree(self.ptr);
            }
        }
    }
}

// Safety: Device pointers can be sent between host threads.
unsafe impl Send for DeviceBuffer {}
// Safety: Device pointers can be shared across threads (read-only access is safe).
unsafe impl Sync for DeviceBuffer {}
