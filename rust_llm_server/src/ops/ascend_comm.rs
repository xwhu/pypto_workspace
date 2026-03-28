//! Ascend NPU communication backend using HCCL.
//!
//! Provides `AscendCommOps` for collective communication operations
//! (AllReduce, Send, Recv, AllGather) used by tensor parallelism and
//! pipeline parallelism in the compiled execution plan.

use crate::model::device_tensor::DeviceTensor;
use crate::model::tensor::DType;

use ascend::comm::HcclCommunicator;
use ascend::DeviceBuffer;
use hccl_sys::HcclDataType;

/// Convert our `DType` to HCCL's `HcclDataType`.
fn to_hccl_dtype(dtype: DType) -> HcclDataType {
    match dtype {
        DType::Float16 => HcclDataType::Float16,
        DType::BFloat16 => HcclDataType::BFloat16,
        DType::Float32 => HcclDataType::Float32,
        DType::Int32 => HcclDataType::Int32,
        DType::Uint32 => HcclDataType::Uint32,
        DType::Int8 => HcclDataType::Int8,
        // INT4 is not directly supported by HCCL; would need packing
        DType::Int4 => HcclDataType::Int8,
    }
}

/// Convert our `DType` to CANN's `AclDataType` (for aclnnCast).
fn to_acl_dtype(dtype: DType) -> aclnn_sys::common::AclDataType {
    use aclnn_sys::common::AclDataType;
    match dtype {
        DType::Float16 => AclDataType::Float16,
        DType::BFloat16 => AclDataType::BFloat16,
        DType::Float32 => AclDataType::Float,
        DType::Int32 => AclDataType::Int32,
        DType::Uint32 => AclDataType::Uint32,
        DType::Int8 => AclDataType::Int8,
        DType::Int4 => AclDataType::Int4,
    }
}

/// Communication operations backend for Ascend NPUs using HCCL.
///
/// Holds separate communicators for TP and PP groups. Uses the compute
/// stream (non-owning handle) for automatic serialization with compute ops.
pub struct AscendCommOps {
    /// Communicator for tensor parallelism group (AllReduce, AllGather).
    tp_comm: Option<HcclCommunicator>,
    /// Communicator for pipeline parallelism group (Send, Recv).
    pp_comm: Option<HcclCommunicator>,
    /// Raw compute stream handle (non-owning). Shared with AscendComputeOps.
    /// Using the same stream as compute ensures automatic serialization.
    stream_raw: ascendcl_sys::AclrtStream,
}

// SAFETY: AclrtStream is a *mut c_void handle. CANN stream handles are
// designed to be used from multiple threads when the context is shared.
unsafe impl Send for AscendCommOps {}
unsafe impl Sync for AscendCommOps {}

impl AscendCommOps {
    /// Create communication ops with the given communicators.
    ///
    /// `stream_raw` should be the compute stream handle from AscendComputeOps.
    /// Either communicator can be `None` if that parallelism dimension is not used.
    pub fn new(
        tp_comm: Option<HcclCommunicator>,
        pp_comm: Option<HcclCommunicator>,
        stream_raw: ascendcl_sys::AclrtStream,
    ) -> Self {
        Self {
            tp_comm,
            pp_comm,
            stream_raw,
        }
    }

    /// All-reduce sum in-place on a DeviceTensor.
    ///
    /// Used after row-sharded projections (O_proj, down_proj) in TP.
    /// The tensor is modified in-place — send and receive use the same buffer.
    pub fn all_reduce_sum_inplace(&self, tensor: &DeviceTensor) {
        let comm = self
            .tp_comm
            .as_ref()
            .expect("all_reduce_sum requires TP communicator");

        let dtype = to_hccl_dtype(tensor.dtype());
        let count = tensor.meta.numel() as u64;

        comm.all_reduce_sum(self.stream_raw, &tensor.buf, count, dtype)
            .expect("HCCL AllReduce failed");
    }

    /// Cast a device buffer from one dtype to another using aclnnCast.
    ///
    /// This is a helper for the FP32-upscale AllReduce path.
    fn cast_tensor(
        &self,
        src_buf: &DeviceBuffer,
        shape: &[usize],
        src_dtype: DType,
        dst_buf: &DeviceBuffer,
        dst_dtype: DType,
    ) {
        use aclnn_sys::common::AclOpExecutor;
        use aclnn_sys::elementwise::{aclnnCast, aclnnCastGetWorkspaceSize};
        use ascend::tensor::AclTensor;

        let acl_src_dtype = to_acl_dtype(src_dtype);
        let acl_dst_dtype = to_acl_dtype(dst_dtype);

        let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();

        let src_tensor = AclTensor::from_ptr(&shape_i64, acl_src_dtype, src_buf.ptr())
            .expect("Failed to create src AclTensor for cast");
        let dst_tensor = AclTensor::from_ptr(&shape_i64, acl_dst_dtype, dst_buf.ptr())
            .expect("Failed to create dst AclTensor for cast");

        // Stage 1: get workspace size
        let mut ws_size: u64 = 0;
        let mut executor: *mut AclOpExecutor = std::ptr::null_mut();
        let status = unsafe {
            aclnnCastGetWorkspaceSize(
                src_tensor.raw(),
                acl_dst_dtype,
                dst_tensor.raw(),
                &mut ws_size,
                &mut executor,
            )
        };
        assert_eq!(status, 0, "aclnnCastGetWorkspaceSize failed: {}", status);

        // Allocate workspace if needed
        let ws_buf = if ws_size > 0 {
            Some(DeviceBuffer::alloc(ws_size as usize)
                .expect("Failed to allocate cast workspace"))
        } else {
            None
        };
        let ws_ptr = ws_buf.as_ref().map_or(std::ptr::null_mut(), |b| b.ptr());

        // Stage 2: execute cast
        let status = unsafe {
            aclnnCast(ws_ptr, ws_size, executor, self.stream_raw)
        };
        assert_eq!(status, 0, "aclnnCast failed: {}", status);
    }

    /// Point-to-point send a DeviceTensor to a destination PP rank.
    ///
    /// Used at pipeline stage boundaries to send hidden states forward.
    pub fn send_tensor(&self, tensor: &DeviceTensor, dst_rank: usize) {
        let comm = self
            .pp_comm
            .as_ref()
            .expect("send requires PP communicator");

        let dtype = to_hccl_dtype(tensor.dtype());
        let count = tensor.meta.numel() as u64;

        comm.send(self.stream_raw, &tensor.buf, count, dtype, dst_rank as u32)
            .expect("HCCL Send failed");
    }

    /// Point-to-point receive a DeviceTensor from a source PP rank.
    ///
    /// Allocates a new DeviceTensor with the given shape and receives into it.
    /// Used at pipeline stage boundaries to receive hidden states.
    pub fn recv_tensor(
        &self,
        shape: &[usize],
        dtype: DType,
        src_rank: usize,
    ) -> DeviceTensor {
        let comm = self
            .pp_comm
            .as_ref()
            .expect("recv requires PP communicator");

        let hccl_dtype = to_hccl_dtype(dtype);
        let count: u64 = shape.iter().product::<usize>() as u64;

        // Allocate receive buffer
        let size_bytes = count as usize * dtype.size_bytes();
        let buf = DeviceBuffer::alloc(size_bytes).expect("Failed to allocate recv buffer");

        comm.recv(self.stream_raw, &buf, count, hccl_dtype, src_rank as u32)
            .expect("HCCL Recv failed");

        DeviceTensor::from_buf(shape.to_vec(), dtype, "pp_recv", buf)
    }

    /// All-gather: gather tensor shards from all TP ranks.
    ///
    /// Each rank contributes its local tensor. The output tensor has
    /// `n_ranks` times the elements along the gathered dimension.
    pub fn all_gather(
        &self,
        tensor: &DeviceTensor,
        output_shape: &[usize],
        dtype: DType,
    ) -> DeviceTensor {
        let comm = self
            .tp_comm
            .as_ref()
            .expect("all_gather requires TP communicator");

        let hccl_dtype = to_hccl_dtype(dtype);
        let send_count = tensor.meta.numel() as u64;

        // Allocate output buffer
        let out_size: usize = output_shape.iter().product::<usize>() * dtype.size_bytes();
        let out_buf = DeviceBuffer::alloc(out_size).expect("Failed to allocate all_gather output");

        comm.all_gather(self.stream_raw, &tensor.buf, &out_buf, send_count, hccl_dtype)
            .expect("HCCL AllGather failed");

        DeviceTensor::from_buf(output_shape.to_vec(), dtype, "tp_all_gather", out_buf)
    }

    /// Broadcast a device tensor from root rank to all TP ranks.
    ///
    /// Used to distribute `input_ids` and `positions` from the primary rank (0)
    /// to all worker ranks before each forward pass. All ranks must call this
    /// simultaneously — it blocks at the HCCL level until the broadcast completes.
    ///
    /// - On the **root** rank: the tensor's buffer is sent to all other ranks.
    /// - On **non-root** ranks: the buffer must already be allocated; its contents
    ///   will be overwritten with data from the root.
    pub fn broadcast_tensor(&self, tensor: &DeviceTensor, root: usize) {
        let comm = self
            .tp_comm
            .as_ref()
            .expect("broadcast_tensor requires TP communicator");

        let dtype = to_hccl_dtype(tensor.dtype());
        let count = tensor.meta.numel() as u64;

        comm.broadcast(self.stream_raw, &tensor.buf, count, dtype, root as u32)
            .expect("HCCL Broadcast failed");
    }

    /// Upload a `&[u32]` slice to device and broadcast it from root to all TP ranks.
    ///
    /// On the root rank: allocates a device buffer, copies `ids` into it, broadcasts.
    /// On non-root ranks: allocates a device buffer of the same `count` elements,
    ///   receives broadcast data into it.
    ///
    /// Returns the resulting `DeviceTensor` (valid on all ranks after the call).
    pub fn broadcast_u32_slice(
        &self,
        ids: &[u32],
        name: &str,
        root: usize,
        is_root: bool,
    ) -> DeviceTensor {
        use crate::model::tensor::DType;

        let count = ids.len();
        let size_bytes = count * DType::Uint32.size_bytes();

        let mut buf = DeviceBuffer::alloc(size_bytes)
            .expect("broadcast_u32_slice: failed to allocate device buffer");

        if is_root {
            // Upload from CPU to NPU, then broadcast
            // SAFETY: u32 has no invalid bit patterns; reinterpreting as bytes is safe.
            let bytes = unsafe {
                std::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 4)
            };
            buf.copy_from_host(bytes)
                .expect("broadcast_u32_slice: failed to copy to device");
        }
        // All ranks (including root) participate in the broadcast
        let tensor = DeviceTensor::from_buf(vec![count], DType::Uint32, name, buf);
        self.broadcast_tensor(&tensor, root);
        // Synchronize: HcclBroadcast is enqueued asynchronously on the stream.
        // Workers need the data in device memory before copy_to_host, so we must
        // wait for the stream to drain here.
        self.synchronize();
        tensor
    }

    /// Synchronize the communication/compute stream.
    pub fn synchronize(&self) {
        let err = unsafe { ascendcl_sys::aclrtSynchronizeStream(self.stream_raw) };
        assert_eq!(err, 0, "Failed to synchronize comm stream: Acl({})", err);
    }
}
