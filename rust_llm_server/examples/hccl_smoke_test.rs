//! Minimal HCCL smoke test — verifies that HcclBroadcast and HcclAllReduce
//! actually work on this hardware.
//!
//! Run with 2 processes:
//!   ASCEND_DEVICE_ID=0 cargo run --example hccl_smoke_test --features "ascend,hccl" -- --rank 0 --nranks 2
//!   ASCEND_DEVICE_ID=1 cargo run --example hccl_smoke_test --features "ascend,hccl" -- --rank 1 --nranks 2
//!
//! Or use the helper script (launches both):
//!   bash examples/run_hccl_smoke_test.sh

use std::path::Path;
use std::thread;
use std::time::Duration;

fn main() {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let rank: u32 = args.iter()
        .position(|a| a == "--rank")
        .map(|i| args[i + 1].parse().unwrap())
        .expect("Usage: --rank <0|1> --nranks <N>");
    let nranks: u32 = args.iter()
        .position(|a| a == "--nranks")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(2);

    let device_id = std::env::var("ASCEND_DEVICE_ID")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(rank as i32);

    println!("[rank {rank}] HCCL smoke test: nranks={nranks}, device_id={device_id}");

    // ─── Step 1: Init device ───────────────────────────────────────
    println!("[rank {rank}] Initializing device {device_id}...");
    let _device = ascend::Device::init(device_id)
        .expect("Device::init failed");
    println!("[rank {rank}] Device {device_id} initialized OK");

    let stream = ascend::Stream::new().expect("Stream::new failed");
    println!("[rank {rank}] Stream created OK");

    // ─── Step 2: Create HCCL communicator ──────────────────────────
    let root_info_path = Path::new("/tmp/hccl_smoke_test_root.bin");

    let root_info = if rank == 0 {
        // Remove stale file
        let _ = std::fs::remove_file(root_info_path);

        let info = ascend::comm::get_root_info()
            .expect("get_root_info failed");
        ascend::comm::write_root_info_to_file(&info, root_info_path)
            .expect("write_root_info_to_file failed");
        println!("[rank {rank}] Root info written to {}", root_info_path.display());
        info
    } else {
        // Wait for rank 0 to write the file
        println!("[rank {rank}] Waiting for root info file...");
        let start = std::time::Instant::now();
        loop {
            if root_info_path.exists() {
                break;
            }
            if start.elapsed() > Duration::from_secs(30) {
                panic!("[rank {rank}] Timeout waiting for root info file");
            }
            thread::sleep(Duration::from_millis(100));
        }
        let info = ascend::comm::read_root_info_from_file(root_info_path)
            .expect("read_root_info_from_file failed");
        println!("[rank {rank}] Root info read from file");
        info
    };

    println!("[rank {rank}] Calling HcclCommInitRootInfo(nranks={nranks}, rank={rank})...");
    let comm = ascend::comm::HcclCommunicator::init_rank(nranks, &root_info, rank)
        .expect("HcclCommInitRootInfo failed");
    println!("[rank {rank}] ✅ HCCL communicator initialized successfully");

    // Clean up root info file
    if rank == 0 {
        let _ = std::fs::remove_file(root_info_path);
    }

    // ─── Step 3: Test AllReduce (Out-of-place) ─────────────────────
    println!("[rank {rank}] Testing AllReduce (Out-of-place)...");
    {
        // Allocate 1024 bytes (256 elements) to avoid small-buffer issues
        let count = 256;
        let mut data = vec![rank as f32 + 1.0; count];
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, count * 4)
        };
        let mut send_buf = ascend::DeviceBuffer::alloc(count * 4)
            .expect("alloc send_buf failed");
        let recv_buf = ascend::DeviceBuffer::alloc(count * 4)
            .expect("alloc recv_buf failed");

        send_buf.copy_from_host(data_bytes)
            .expect("H2D copy for AllReduce failed");

        println!("[rank {rank}] AllReduce: input[0] = {}", data[0]);

        let result = comm.all_reduce(
            stream.raw(),
            &send_buf,
            &recv_buf,
            count as u64,
            hccl_sys::HcclDataType::Float32,
            hccl_sys::HcclReduceOp::Sum,
        );
        match &result {
            Ok(()) => println!("[rank {rank}] HcclAllReduce call returned OK"),
            Err(e) => println!("[rank {rank}] ❌ HcclAllReduce call FAILED: {:?}", e),
        }

        if result.is_ok() {
            println!("[rank {rank}] Synchronizing stream after AllReduce...");
            let sync_err = unsafe { ascendcl_sys::aclrtSynchronizeStream(stream.raw()) };
            if sync_err == 0 {
                let mut result_data = vec![0f32; count];
                let result_bytes = unsafe {
                    std::slice::from_raw_parts_mut(result_data.as_mut_ptr() as *mut u8, count * 4)
                };
                recv_buf.copy_to_host(result_bytes).expect("D2H copy failed");
                println!("[rank {rank}] ✅ AllReduce result[0] = {} (expected {})", result_data[0], (nranks * (nranks + 1) / 2) as f32);
            } else {
                println!("[rank {rank}] ❌ stream.synchronize after AllReduce FAILED: Acl({})", sync_err);
            }
        }
    }

    // ─── Step 4: Test Broadcast ────────────────────────────────────
    println!("[rank {rank}] Testing Broadcast...");
    {
        let count = 256;
        let mut buf = ascend::DeviceBuffer::alloc(count * 4)
            .expect("alloc for Broadcast failed");

        if rank == 0 {
            let data = vec![42.0f32; count];
            let data_bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, count * 4)
            };
            buf.copy_from_host(data_bytes)
                .expect("H2D copy for Broadcast failed");
            println!("[rank {rank}] Broadcast: sending 42.0 from root");
        } else {
            // init with zeros
            let data = vec![0.0f32; count];
            let data_bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, count * 4)
            };
            buf.copy_from_host(data_bytes).expect("H2D zeroing failed");
            println!("[rank {rank}] Broadcast: waiting to receive from root");
        }

        let result = comm.broadcast(
            stream.raw(),
            &buf,
            count as u64,
            hccl_sys::HcclDataType::Float32,
            0, // root
        );
        match &result {
            Ok(()) => println!("[rank {rank}] HcclBroadcast call returned OK"),
            Err(e) => println!("[rank {rank}] ❌ HcclBroadcast call FAILED: {:?}", e),
        }

        if result.is_ok() {
            println!("[rank {rank}] Synchronizing stream after Broadcast...");
            let sync_err = unsafe { ascendcl_sys::aclrtSynchronizeStream(stream.raw()) };
            if sync_err == 0 {
                let mut result_data = vec![0f32; count];
                let result_bytes = unsafe {
                    std::slice::from_raw_parts_mut(result_data.as_mut_ptr() as *mut u8, count * 4)
                };
                buf.copy_to_host(result_bytes).expect("D2H copy failed");
                println!("[rank {rank}] ✅ Broadcast result[0] = {} (expected 42.0)", result_data[0]);
            } else {
                println!("[rank {rank}] ❌ stream.synchronize after Broadcast FAILED: Acl({})", sync_err);
            }
        }
    }

    // ─── Step 5: Test Broadcast with u32 (our actual use case) ─────
    println!("[rank {rank}] Testing Broadcast with u32...");
    {
        let count = 256;
        let mut buf = ascend::DeviceBuffer::alloc(count * 4)
            .expect("alloc for u32 Broadcast failed");

        if rank == 0 {
            let data = vec![24u32; count]; 
            let data_bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, count * 4)
            };
            buf.copy_from_host(data_bytes)
                .expect("H2D copy for u32 Broadcast failed");
            println!("[rank {rank}] u32 Broadcast: sending 24 from root");
        } else {
            // init with zeros
            let data = vec![0u32; count];
            let data_bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, count * 4)
            };
            buf.copy_from_host(data_bytes).expect("H2D zeroing failed");
            println!("[rank {rank}] u32 Broadcast: waiting to receive from root");
        }

        let result = comm.broadcast(
            stream.raw(),
            &buf,
            count as u64,
            hccl_sys::HcclDataType::Uint32,
            0,
        );
        match &result {
            Ok(()) => println!("[rank {rank}] HcclBroadcast(u32) call returned OK"),
            Err(e) => println!("[rank {rank}] ❌ HcclBroadcast(u32) call FAILED: {:?}", e),
        }

        if result.is_ok() {
            println!("[rank {rank}] Synchronizing stream after u32 Broadcast...");
            let sync_err = unsafe { ascendcl_sys::aclrtSynchronizeStream(stream.raw()) };
            if sync_err == 0 {
                let mut result_data = vec![0u32; count];
                let result_bytes = unsafe {
                    std::slice::from_raw_parts_mut(result_data.as_mut_ptr() as *mut u8, count * 4)
                };
                buf.copy_to_host(result_bytes).expect("D2H copy failed");
                println!("[rank {rank}] ✅ u32 Broadcast result[0] = {} (expected 24)", result_data[0]);
            } else {
                println!("[rank {rank}] ❌ stream.synchronize after u32 Broadcast FAILED: Acl({})", sync_err);
            }
        }
    }

    println!("[rank {rank}] 🏁 HCCL smoke test complete!");
}
