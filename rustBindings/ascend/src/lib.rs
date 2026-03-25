//! `ascend` — Safe Rust wrappers for Ascend NPU.
//!
//! Built on top of `ascendcl-sys` (runtime) and `aclnn-sys` (operators).
//! Provides RAII resource management and idiomatic Rust error handling.
//!
//! # Usage
//! ```no_run
//! use ascend::{Device, Stream, DeviceBuffer, AclTensor};
//! use aclnn_sys::common::AclDataType;
//!
//! // Initialize device
//! let _guard = Device::init(0).unwrap();
//!
//! // Create stream and allocate memory
//! let stream = Stream::new().unwrap();
//! let mut buf_a = DeviceBuffer::alloc(4096 * 4096 * 2).unwrap(); // FP16
//! let mut buf_b = DeviceBuffer::alloc(4096 * 4096 * 2).unwrap();
//! let mut buf_out = DeviceBuffer::alloc(4096 * 4096 * 2).unwrap();
//!
//! // Create tensor descriptors
//! let a = AclTensor::new(&[4096, 4096], AclDataType::Float16, &buf_a).unwrap();
//! let b = AclTensor::new(&[4096, 4096], AclDataType::Float16, &buf_b).unwrap();
//! let mut out = AclTensor::new(&[4096, 4096], AclDataType::Float16, &buf_out).unwrap();
//!
//! // Run matmul
//! ascend::ops::matmul::matmul(&stream, &a, &b, &mut out).unwrap();
//! stream.synchronize().unwrap();
//! ```
//!
//! # Modules
//! - [`device`] — Device management (init/finalize, device selection)
//! - [`stream`] — Async execution streams
//! - [`memory`] — Device memory allocation/deallocation
//! - [`tensor`] — AclTensor descriptors with RAII
//! - [`ops`] — Safe operator wrappers (matmul, rmsnorm, etc.)
//! - [`error`] — Error types and result helpers

pub mod device;
pub mod error;
pub mod memory;
pub mod ops;
pub mod stream;
pub mod tensor;

// Re-export key types
pub use device::Device;
pub use error::{AscendError, Result};
pub use memory::DeviceBuffer;
pub use stream::Stream;
pub use tensor::AclTensor;
