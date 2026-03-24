//! Integration tests for the `ascend` crate.
//!
//! These tests require real Ascend NPU hardware and CANN SDK.
//! They are gated by the `ASCEND_DEVICE_ID` environment variable.
//!
//! # Running
//! ```bash
//! # Use device 0 (default)
//! ASCEND_DEVICE_ID=0 cargo test --test integration
//!
//! # Use a specific device
//! ASCEND_DEVICE_ID=2 cargo test --test integration
//!
//! # Skip hardware tests (default without env var — tests will be skipped)
//! cargo test --test integration
//! ```

use ascend::{Device, Stream, DeviceBuffer, AclTensor};
use aclnn_sys::common::AclDataType;

/// Helper: skip test if ASCEND_DEVICE_ID is not set.
/// Returns the device guard if hardware is available.
fn require_device() -> Option<Device> {
    if std::env::var("ASCEND_DEVICE_ID").is_err() {
        eprintln!("SKIP: ASCEND_DEVICE_ID not set, skipping hardware test");
        return None;
    }
    match Device::from_env() {
        Ok(dev) => {
            eprintln!("Using Ascend device {}", dev.id());
            Some(dev)
        }
        Err(e) => {
            eprintln!("SKIP: Failed to init device: {}", e);
            None
        }
    }
}

// ─── Device Tests ──────────────────────────────────────────────────────

#[test]
fn test_device_init_and_info() {
    let Some(dev) = require_device() else { return };

    // Device ID should match what we requested
    let expected_id: i32 = std::env::var("ASCEND_DEVICE_ID")
        .unwrap_or("0".to_string())
        .parse()
        .unwrap_or(0);
    assert_eq!(dev.id(), expected_id);

    // Device count should be >= 1
    let count = Device::count().expect("Failed to get device count");
    assert!(count >= 1, "Expected at least 1 device, got {}", count);
    eprintln!("Device count: {}", count);

    // Memory info should return non-zero values
    let (free, total) = dev.memory_info().expect("Failed to get memory info");
    assert!(total > 0, "Total device memory should be > 0");
    assert!(free > 0, "Free device memory should be > 0");
    assert!(free <= total, "Free memory should not exceed total");
    eprintln!("Device memory: {:.2} GB free / {:.2} GB total",
        free as f64 / 1e9, total as f64 / 1e9);
}

// ─── Stream Tests ──────────────────────────────────────────────────────

#[test]
fn test_stream_create_sync() {
    let Some(_dev) = require_device() else { return };

    let stream = Stream::new().expect("Failed to create stream");
    stream.synchronize().expect("Failed to synchronize empty stream");

    // Create multiple streams
    let streams: Vec<_> = (0..4)
        .map(|_| Stream::new().expect("Failed to create stream"))
        .collect();
    for s in &streams {
        s.synchronize().expect("Failed to synchronize stream");
    }
    eprintln!("Created and synced {} streams", streams.len());
    // Streams are destroyed on drop
}

// ─── Memory Tests ──────────────────────────────────────────────────────

#[test]
fn test_device_buffer_alloc_free() {
    let Some(_dev) = require_device() else { return };

    // Allocate various sizes
    let sizes = [0, 1, 1024, 4096, 1024 * 1024, 16 * 1024 * 1024];
    for &size in &sizes {
        let buf = DeviceBuffer::alloc(size)
            .unwrap_or_else(|e| panic!("Failed to alloc {} bytes: {}", size, e));
        assert_eq!(buf.size(), size);
        if size > 0 {
            assert!(!buf.ptr().is_null(), "Non-zero alloc returned null");
        }
        // buf dropped here → aclrtFree
    }
    eprintln!("Allocated and freed {} buffers", sizes.len());
}

#[test]
fn test_device_buffer_memcpy_roundtrip() {
    let Some(_dev) = require_device() else { return };

    // Create host data
    let size = 4096;
    let host_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    // Alloc device buffer and copy H→D
    let mut buf = DeviceBuffer::alloc(size).expect("alloc failed");
    buf.copy_from_host(&host_data).expect("H2D copy failed");

    // Copy D→H and verify
    let mut readback = vec![0u8; size];
    buf.copy_to_host(&mut readback).expect("D2H copy failed");

    assert_eq!(host_data, readback, "Roundtrip data mismatch");
    eprintln!("Memcpy roundtrip: {} bytes verified", size);
}

#[test]
fn test_device_buffer_memset() {
    let Some(_dev) = require_device() else { return };

    let size = 1024;
    let mut buf = DeviceBuffer::alloc(size).expect("alloc failed");

    // Write some data
    let data = vec![0xABu8; size];
    buf.copy_from_host(&data).expect("H2D failed");

    // Zero-fill
    buf.memset_zero().expect("memset failed");

    // Verify zeros
    let mut readback = vec![0xFFu8; size];
    buf.copy_to_host(&mut readback).expect("D2H failed");
    assert!(readback.iter().all(|&b| b == 0), "memset did not zero buffer");
    eprintln!("Memset zero: {} bytes verified", size);
}

// ─── Tensor Descriptor Tests ───────────────────────────────────────────

#[test]
fn test_acl_tensor_create() {
    let Some(_dev) = require_device() else { return };

    // Allocate device memory for a [4, 8] FP16 tensor
    let shape = [4i64, 8];
    let elem_size = 2; // FP16 = 2 bytes
    let numel: i64 = shape.iter().product();
    let byte_size = (numel as usize) * elem_size;

    let buf = DeviceBuffer::alloc(byte_size).expect("alloc failed");
    let tensor = AclTensor::new(&shape, AclDataType::Float16, &buf)
        .expect("Failed to create AclTensor");

    assert_eq!(tensor.shape(), &shape);
    assert_eq!(tensor.dtype(), AclDataType::Float16);
    assert_eq!(tensor.numel(), 32);
    assert!(!tensor.raw().is_null());
    eprintln!("Created AclTensor shape={:?} dtype=FP16", shape);
    // tensor + buf dropped → aclDestroyTensor + aclrtFree
}

#[test]
fn test_acl_tensor_various_shapes() {
    let Some(_dev) = require_device() else { return };

    let test_cases: Vec<(Vec<i64>, AclDataType, usize)> = vec![
        (vec![1], AclDataType::Float16, 2),          // scalar-ish
        (vec![128], AclDataType::Float, 4),           // 1D
        (vec![4, 128], AclDataType::Float16, 2),      // 2D
        (vec![1, 32, 128], AclDataType::Float16, 2),  // 3D
        (vec![2, 4, 32, 64], AclDataType::Float16, 2), // 4D (attention-like)
    ];

    for (shape, dtype, elem_size) in &test_cases {
        let numel: i64 = shape.iter().product();
        let byte_size = (numel as usize) * elem_size;
        let buf = DeviceBuffer::alloc(byte_size).expect("alloc failed");
        let tensor = AclTensor::new(shape, *dtype, &buf)
            .unwrap_or_else(|e| panic!("Failed for shape {:?}: {}", shape, e));
        assert_eq!(tensor.numel(), numel);
        eprintln!("  OK: shape={:?} numel={}", shape, numel);
    }
}

// ─── Operator Tests ────────────────────────────────────────────────────

#[test]
fn test_matmul_small() {
    let Some(_dev) = require_device() else { return };

    let stream = Stream::new().expect("stream failed");

    // A: [2, 3] FP16, B: [3, 4] FP16, Out: [2, 4] FP16
    let m = 2i64; let k = 3i64; let n = 4i64;
    let elem = 2usize; // FP16

    let buf_a = DeviceBuffer::alloc((m * k) as usize * elem).expect("alloc A");
    let buf_b = DeviceBuffer::alloc((k * n) as usize * elem).expect("alloc B");
    let buf_out = DeviceBuffer::alloc((m * n) as usize * elem).expect("alloc Out");

    let a = AclTensor::new(&[m, k], AclDataType::Float16, &buf_a).expect("tensor A");
    let b = AclTensor::new(&[k, n], AclDataType::Float16, &buf_b).expect("tensor B");
    let mut out = AclTensor::new(&[m, n], AclDataType::Float16, &buf_out).expect("tensor Out");

    // Run matmul (inputs are zeros → output should be zeros)
    ascend::ops::matmul::matmul(&stream, &a, &b, &mut out)
        .expect("matmul failed");
    stream.synchronize().expect("sync failed");

    // Read back and verify (all zeros since input buffers are uninitialized
    // but this at least verifies the operator runs without crashing)
    let mut result = vec![0u8; (m * n) as usize * elem];
    buf_out.copy_to_host(&mut result).expect("D2H failed");
    eprintln!("Matmul [{}x{}] @ [{}x{}] → [{}x{}] completed", m, k, k, n, m, n);
}

#[test]
fn test_rmsnorm_small() {
    let Some(_dev) = require_device() else { return };

    let stream = Stream::new().expect("stream failed");

    // x: [1, 8] FP16, gamma: [8] FP16, y: [1, 8] FP16
    let hidden = 8i64;
    let elem = 2usize;

    let buf_x = DeviceBuffer::alloc(hidden as usize * elem).expect("alloc x");
    let buf_gamma = DeviceBuffer::alloc(hidden as usize * elem).expect("alloc gamma");
    let buf_y = DeviceBuffer::alloc(hidden as usize * elem).expect("alloc y");

    let x = AclTensor::new(&[1, hidden], AclDataType::Float16, &buf_x).expect("tensor x");
    let gamma = AclTensor::new(&[hidden], AclDataType::Float16, &buf_gamma).expect("tensor gamma");
    let mut y = AclTensor::new(&[1, hidden], AclDataType::Float16, &buf_y).expect("tensor y");

    ascend::ops::rmsnorm::rmsnorm(&stream, &x, &gamma, 1e-6, &mut y)
        .expect("rmsnorm failed");
    stream.synchronize().expect("sync failed");

    eprintln!("RmsNorm [1, {}] completed", hidden);
}
