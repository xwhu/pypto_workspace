pub mod stubs;

#[cfg(feature = "ascend")]
pub mod ascend;

pub use stubs::{ComputeOps, OpsBundle, StubComputeOps};
