//! Error types for the Ascend crate.

use thiserror::Error;
use ascendcl_sys::AclError;
use aclnn_sys::AclnnStatus;

#[derive(Error, Debug)]
pub enum AscendError {
    #[error("AscendCL error: code {0}")]
    Acl(AclError),

    #[error("aclnn operator error: code {0}")]
    Aclnn(AclnnStatus),

    #[error("Device not initialized")]
    NotInitialized,

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Out of device memory (requested {requested} bytes, {available} bytes free)")]
    OutOfMemory { requested: usize, available: usize },
}

pub type Result<T> = std::result::Result<T, AscendError>;

/// Check an AscendCL return code, converting non-zero to an error.
#[inline]
pub fn check_acl(code: AclError) -> Result<()> {
    if code == ascendcl_sys::ACL_SUCCESS {
        Ok(())
    } else {
        Err(AscendError::Acl(code))
    }
}

/// Check an aclnn return code, converting non-zero to an error.
#[inline]
pub fn check_aclnn(code: AclnnStatus) -> Result<()> {
    if code == aclnn_sys::ACLNN_SUCCESS {
        Ok(())
    } else {
        Err(AscendError::Aclnn(code))
    }
}
