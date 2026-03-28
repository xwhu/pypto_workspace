pub mod stubs;

#[cfg(feature = "ascend")]
pub mod ascend;

// Full HCCL-backed comm ops when both features are enabled
#[cfg(all(feature = "ascend", feature = "hccl"))]
pub mod ascend_comm;

// Stub comm ops when ascend is enabled but hccl is not — keeps type signatures uniform
#[cfg(all(feature = "ascend", not(feature = "hccl")))]
pub mod ascend_comm {
    use crate::model::device_tensor::DeviceTensor;
    use crate::model::tensor::DType;

    /// Stub type — never constructed at runtime. Satisfies type signatures in
    /// execute_paged when HCCL support is not compiled in. Methods panic if
    /// somehow called, which cannot happen since comm_ops is always None.
    pub struct AscendCommOps;

    impl AscendCommOps {
        pub fn all_reduce_sum_inplace(&self, _tensor: &DeviceTensor) {
            panic!("HCCL not compiled in — build with --features hccl");
        }
        pub fn send_tensor(&self, _tensor: &DeviceTensor, _dst_rank: usize) {
            panic!("HCCL not compiled in — build with --features hccl");
        }
        pub fn recv_tensor(&self, _shape: &[usize], _dtype: DType, _src_rank: usize) -> DeviceTensor {
            panic!("HCCL not compiled in — build with --features hccl");
        }
    }
}

pub use stubs::{ComputeOps, OpsBundle};
