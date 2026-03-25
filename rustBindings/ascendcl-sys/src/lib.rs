//! `ascendcl-sys` — Raw FFI bindings to AscendCL runtime.
//!
//! Provides unsafe `extern "C"` declarations for `libascendcl.so` functions.
//!
//! # Modules
//! - [`types`] — Enums, constants, and opaque handle types
//! - [`runtime`] — Device, context, stream, event management
//! - [`memory`] — Device memory allocation, copy, and query
//!
//! # Safety
//! All functions are `unsafe extern "C"`. Callers must ensure:
//! - AscendCL is initialized (`aclInit`) before calling other functions
//! - Pointers are valid and properly aligned
//! - Device memory is allocated before use
//!
//! # Feature Flags
//! - `stub`: Skip linking `libascendcl.so` (for development without CANN SDK)

pub mod memory;
pub mod runtime;
pub mod types;

// Re-export everything at crate root for convenience
pub use memory::*;
pub use runtime::*;
pub use types::*;
