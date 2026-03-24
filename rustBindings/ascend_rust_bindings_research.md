# Rust Bindings for Ascend Kernel: Research Report

## 1. Executive Summary

There **is** existing work on Rust bindings for Ascend kernels — the `ascend-rs` project by Huawei. However, it is **not publicly available** (private repository, pending open-source decision). No other public Rust crate on `crates.io` provides low-level Ascend NPU bindings.

Given your LLM inference server design (`rust_llm_server_design_v8.md`) which targets GPU/NPU cards with CXL, you need Rust bindings at two distinct layers:
1. **Host-side AscendCL** — device management, memory, streams, kernel launch (analogous to CUDA Runtime API)
2. **Device-side AscendC** — writing compute kernels that run on the NPU itself (analogous to CUDA kernel code)

---

## 2. The Ascend Software Stack

```
┌─────────────────────────────────────────────────────┐
│  Application / Framework (PyTorch, MindSpore, etc.) │
├─────────────────────────────────────────────────────┤
│  CANN SDK (Compute Architecture for Neural Networks)│
│  ┌───────────────────┬──────────────────────────┐   │
│  │ AscendCL (C API)  │ AscendC (C++ API)        │   │
│  │ Host-side:        │ Device-side:             │   │
│  │ - Device mgmt     │ - Kernel programming     │   │
│  │ - Memory mgmt     │ - TPipe, TQue, DMA ops   │   │
│  │ - Stream mgmt     │ - Vector/Cube intrinsics │   │
│  │ - Model loading   │ - Pipeline: CopyIn →     │   │
│  │ - Operator launch │   Compute → CopyOut      │   │
│  │ - Media proc      │                          │   │
│  └───────────────────┴──────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│  bisheng Compiler (LLVM-based, compiles .cpp → .o)  │
├─────────────────────────────────────────────────────┤
│  Ascend NPU Hardware (Da Vinci architecture)        │
│  AI Core: Cube Unit + Vector Unit + Scalar Unit     │
└─────────────────────────────────────────────────────┘
```

### Key APIs

| API Layer | Language | Purpose | Header Files |
|-----------|----------|---------|-------------|
| **AscendCL** | C | Host-side runtime (like CUDA Runtime API) | `acl/acl.h`, `acl/acl_rt.h`, `acl/acl_op.h`, `acl/acl_mdl.h` |
| **AscendC** | C++ | Device-side kernel development (like CUDA kernels) | `kernel_operator.h`, `tpipe.h`, `tque.h` |

---

## 3. Existing Work

### 3.1 `ascend-rs` (Huawei — Private)

**Status**: Private repository at Huawei Boyle Research Center, pending open-source decision.

**Website**: [ascend-rs.org](https://ascend-rs.org) — documentation is public.

**Scope**: Full-stack Rust NPU kernel programming — both host and device side.

**Architecture**:
```
Rust kernel source (.rs)
  │  #![no_core], uses ascend_std
  │  #[ascend_std::aiv_kernel] attribute
  ▼
rustc_codegen_mlir (custom rustc backend)
  │  Translates Rust MIR → MLIR
  ▼
MLIR (Multi-Level IR)
  │  mlir_to_cpp pass
  ▼
C++ with AscendC API calls
  │  (DMA ops, vector intrinsics, pipe barriers)
  ▼
bisheng (CANN C++ compiler)
  │
  ▼
NPU binary (.acl.o) → runs on Ascend NPU
```

**Key features**:
- **502 kernels** compiled through MLIR codegen backend
- **Memory-safe** kernel programming via Rust ownership, lifetimes, RAII
- **Zero overhead** — benchmarks show performance parity with hand-optimized C++ AscendC
- `ascend_compile` crate with C ABI for embedding into C/C++ runtimes
- Covers both AIV (AI Vector) and AIC (AI Cube) kernel entry points

**Why you can't use it**: The repository is private and there's no public release timeline.

### 3.2 Other Rust NPU Crates

| Crate | Description | Relevant? |
|-------|-------------|-----------|
| `npu-rs` | Simulation driver for RISC-based NPUs, CPU-only | ❌ Not Ascend |
| `rknpu2-rs` | Rockchip RKNN Runtime bindings | ❌ Not Ascend |
| `ane-infer` | Apple Neural Engine inference | ❌ Not Ascend |
| `ascend-tools-core` | Web API SDK for Ascend cloud instances | ❌ Not kernel dev |

**Conclusion**: No publicly available Rust bindings exist for Ascend kernel development.

---

## 4. What Your LLM Server Actually Needs

Looking at `rust_llm_server_design_v8.md`, the NPU integration points are:

### 4.1 Host-Side (AscendCL Bindings) — **High Priority**

These are needed for your `ChipNode` (L2) and scheduler/HAL layers:

| Functionality | AscendCL API | Your v8 Component |
|---------------|-------------|-------------------|
| Device init/enumeration | `aclrtSetDevice`, `aclrtGetDeviceCount` | `enumerate_gpus()` in tree builder |
| Memory allocation | `aclrtMalloc`, `aclrtFree` | `GpuAllocator` / `GpuBlockStore` |
| Memory copy H2D/D2H | `aclrtMemcpy` | `BlockStore::fetch()` / `store()` at L2 |
| Stream management | `aclrtCreateStream`, `aclrtSynchronizeStream` | Forward pass scheduler |
| Kernel launch | `aclrtLaunchKernel` | Forward pass, attention compute |
| Model loading | `aclmdlLoadFromFile` | Model loader |
| Profiling | `aclprofInit` | Performance monitoring |

### 4.2 Device-Side (AscendC Kernels) — **Lower Priority Initially**

Custom kernels (attention, flash-attention, etc.) can be written in C++ AscendC and called via the host-side FFI. Writing kernels in Rust requires the `ascend-rs` pipeline which isn't available.

---

## 5. Implementation Plan: AscendCL Rust Bindings

Since there are no public Rust bindings, here's a plan to create them. This follows the standard pattern used by `cuda-rs`, `cudarc`, and `opencl3` crates.

### Phase 1: Raw FFI Bindings (`ascendcl-sys`)

**Approach**: Use `bindgen` to auto-generate Rust FFI bindings from AscendCL C headers.

```
ascendcl-sys/
├── Cargo.toml
├── build.rs          # bindgen + link to libascendcl.so
├── wrapper.h         # #include <acl/acl.h> etc.
└── src/
    └── lib.rs        # pub mod bindings (auto-generated)
```

**Key steps**:
1. Locate CANN SDK headers (typically `/usr/local/Ascend/ascend-toolkit/latest/include/acl/`)
2. Write `wrapper.h` including all needed ACL headers
3. Use `bindgen` in `build.rs` to generate FFI types and function signatures
4. Link against `libascendcl.so` (typically at `/usr/local/Ascend/ascend-toolkit/latest/lib64/`)

**Example `build.rs`**:
```rust
fn main() {
    let cann_path = std::env::var("ASCEND_HOME")
        .unwrap_or_else(|_| "/usr/local/Ascend/ascend-toolkit/latest".to_string());

    println!("cargo:rustc-link-search=native={}/lib64", cann_path);
    println!("cargo:rustc-link-lib=dylib=ascendcl");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", cann_path))
        .allowlist_function("acl.*")
        .allowlist_type("acl.*")
        .allowlist_var("ACL.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).unwrap();
}
```

### Phase 2: Safe Rust Wrapper (`ascendcl`)

**Approach**: Wrap raw FFI with safe, idiomatic Rust types using RAII patterns.

```
ascendcl/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── device.rs     # Device, DeviceGuard (auto-reset on drop)
    ├── context.rs    # Context (RAII, auto-destroy on drop)
    ├── stream.rs     # Stream (async execution, RAII)
    ├── memory.rs     # DeviceBuffer<T>, HostBuffer<T> (typed, RAII)
    ├── model.rs      # Model loading and execution
    ├── operator.rs   # Operator loading and execution
    ├── error.rs      # AclError enum, Result type
    └── profiling.rs  # Profiling helpers
```

**Example safe wrapper**:
```rust
// device.rs
pub struct Device {
    id: i32,
}

impl Device {
    pub fn set_current(id: i32) -> Result<Self, AclError> {
        unsafe {
            let ret = ascendcl_sys::aclrtSetDevice(id);
            check_acl_error(ret)?;
        }
        Ok(Device { id })
    }

    pub fn count() -> Result<u32, AclError> {
        let mut count: u32 = 0;
        unsafe {
            let ret = ascendcl_sys::aclrtGetDeviceCount(&mut count);
            check_acl_error(ret)?;
        }
        Ok(count)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { ascendcl_sys::aclrtResetDevice(self.id); }
    }
}

// memory.rs
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl DeviceBuffer {
    pub fn alloc(size: usize) -> Result<Self, AclError> {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            let ret = ascendcl_sys::aclrtMalloc(&mut ptr, size, 0);
            check_acl_error(ret)?;
        }
        Ok(DeviceBuffer { ptr, size })
    }

    pub fn copy_from_host(&self, data: &[u8]) -> Result<(), AclError> {
        assert!(data.len() <= self.size);
        unsafe {
            let ret = ascendcl_sys::aclrtMemcpy(
                self.ptr,
                self.size,
                data.as_ptr() as *const _,
                data.len(),
                ascendcl_sys::ACL_MEMCPY_HOST_TO_DEVICE,
            );
            check_acl_error(ret)?;
        }
        Ok(())
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe { ascendcl_sys::aclrtFree(self.ptr); }
    }
}
```

### Phase 3: Integration with Your LLM Server

Map the safe wrapper to your v8 hierarchy traits:

```rust
// In your server:
impl BlockStore for AscendChipBlockStore {
    fn level(&self) -> HierarchyLevel { HierarchyLevel::Chip }

    fn fetch(&self, hashes: &[BlockHash]) -> Vec<Option<KvBlockData>> {
        hashes.iter().map(|hash| {
            if let Some(addr) = self.index.get(hash) {
                let mut host_buf = vec![0u8; self.block_size];
                self.device_buf.copy_to_host_at(*addr, &mut host_buf).ok()?;
                Some(KvBlockData { meta: ..., data: host_buf })
            } else {
                None
            }
        }).collect()
    }
    // ...
}
```

### Phase 4: Device-Side Kernels (Future)

For custom NPU kernels (flash-attention, fused ops):
- **Option A**: Write in C++ AscendC, compile with `bisheng`, load via `aclrtLaunchKernel` from Rust
- **Option B**: Wait for `ascend-rs` to open-source, use their Rust kernel pipeline
- **Option C**: Build your own Rust → MLIR → AscendC pipeline (very high effort, not recommended)

> [!IMPORTANT]
> Option A (C++ kernels + Rust host) is the pragmatic approach. This is exactly how most CUDA Rust projects work — kernels in CUDA C++, orchestration in Rust.

---

## 6. Effort Estimation

| Phase | Effort | Dependencies |
|-------|--------|-------------|
| Phase 1: `ascendcl-sys` | ~1-2 weeks | CANN SDK installed, `bindgen` |
| Phase 2: `ascendcl` safe wrapper | ~2-3 weeks | Phase 1 |
| Phase 3: LLM server integration | ~2-4 weeks | Phase 2, v8 trait impl |
| Phase 4: Custom kernels | Ongoing | C++ AscendC expertise |

**Total for host-side bindings**: ~5-9 weeks

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `ascend-rs` goes open-source and supersedes your work | Design Phase 2 API to be compatible; monitor [ascend-rs.org](https://ascend-rs.org) |
| AscendCL API changes across CANN versions | Version-gate the `build.rs`, test with multiple CANN releases |
| Complex C types in AscendCL headers cause `bindgen` issues | Manual type annotations for problematic types, allowlist approach |
| No Ascend hardware for testing | CANN toolkit includes software simulation mode (some limitations) |
| AscendC kernels have different semantics than CUDA | Abstract behind your `BlockStore` trait — implementation details are hidden |

---

## 8. Recommendations

1. **Start with Phase 1-2** (AscendCL FFI → safe wrapper) as a standalone crate
2. **Write kernels in C++ AscendC** and call from Rust — don't try to write Rust device code
3. **Monitor `ascend-rs`** — if it goes open-source, adopt its kernel side and keep your host-side wrapper
4. **Structure the crate** to mirror your v8 `HierarchyLevel::Chip` abstraction, treating Ascend NPU the same way you treat GPU
5. **Consider contributing** your AscendCL bindings back to the community once stable — there's a clear gap on `crates.io`

---

## 9. Kernel Formats in PyTorch, vLLM, SGLang — and How to Call from Rust

### 9.1 The Layered Architecture (All Three Share This)

All three frameworks follow the same fundamental pattern for Ascend kernel execution:

```
┌──────────────────────────────────────────────────────────────────┐
│  Python Layer (PyTorch / vLLM / SGLang)                          │
│  torch.ops.aten.matmul / torch.ops._C_ascend.custom_op          │
├──────────────────────────────────────────────────────────────────┤
│  torch_npu C++ Layer (libtorch_npu.so)                           │
│  EXEC_NPU_CMD(aclnnMatMul, ...) macro                            │
├──────────────────────────────────────────────────────────────────┤
│  aclnn* API Layer (libopapi.so — shipped in CANN SDK)        ←── KEY FFI TARGET FOR RUST
│  Two-stage pattern:                                              │
│    1. aclnnXxxGetWorkspaceSize(inputs, &workspaceSize, &executor)│
│    2. aclnnXxx(workspace, workspaceSize, executor, stream)       │
├──────────────────────────────────────────────────────────────────┤
│  AscendCL Runtime (libascendcl.so)                               │
│  Device mgmt, memory, streams, kernel dispatch                   │
├──────────────────────────────────────────────────────────────────┤
│  CANN Operator Library (pre-compiled .so kernel binaries)        │
│  kernel_meta/ directory with tiling data + compiled kernels      │
├──────────────────────────────────────────────────────────────────┤
│  NPU Hardware (Da Vinci AI Core)                                 │
└──────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> The `aclnn*` two-stage C API (from `libopapi.so`) is the critical call target for Rust FFI. This is what ALL three frameworks use under the hood. You do NOT need to go through PyTorch or `torch_npu` — you can call `aclnn*` directly from Rust.

### 9.2 PyTorch / torch_npu — Kernel Format Analysis

**Repository**: [Ascend/Pytorch](https://gitee.com/ascend/pytorch) (also mirrored on GitHub)

**How operators are implemented**:
- Located at `torch_npu/csrc/aten/ops/op_api/` (one `.cpp` per op)
- Each op is a thin C++ wrapper calling `aclnn*` APIs
- Uses the `EXEC_NPU_CMD` macro for the two-stage call

**Example — ExpKernelNpuOpApi.cpp**:
```cpp
// This is how torch_npu calls an Ascend kernel:
#include "aclnnExp.h"  // from CANN SDK

at::Tensor exp_npu(const at::Tensor& self) {
    auto result = npu_preparation::apply_tensor(self);
    // Two-stage API call via macro:
    EXEC_NPU_CMD(aclnnExp, self, result);
    return result;
}
```

The `EXEC_NPU_CMD` macro expands to:
```cpp
// Stage 1: Get workspace size
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnExpGetWorkspaceSize(selfDesc, resultDesc, &workspaceSize, &executor);

// Stage 2: Allocate workspace and execute
void* workspace = aclrtMalloc(workspaceSize);
aclnnExp(workspace, workspaceSize, executor, stream);
```

**Kernel binary format**: Pre-compiled into `libopapi.so` / `liboptiling.so` (CANN built-in operators) or custom `.so` for custom operators. Shipped as part of `Ascend-cann-kernels-{chip}.run` installer.

### 9.3 vLLM Ascend — Kernel Format Analysis

**Repository**: [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)

**Custom operator structure** under `csrc/`:
```
csrc/
├── torch_binding.cpp          # Registers ops to torch.ops._C_ascend
├── torch_binding_meta.cpp     # Meta implementations for graph capture
├── build_aclnn.sh             # Build script
└── {op_name}/
    ├── op_host/               # Host-side C++ (aclnn wrapper)
    │   └── {op_name}.cpp      # Calls aclnn* or custom kernel launch
    └── op_kernel/             # Device-side AscendC kernel
        └── {op_name}.cpp      # AscendC kernel code (CopyIn/Compute/CopyOut)
```

**Two types of custom ops in vLLM Ascend**:

1. **Wrapped `aclnn` ops** — just call existing CANN built-in operators:
```cpp
// In torch_binding.cpp, bound to torch.ops._C_ascend.rms_norm
void rms_norm(at::Tensor& output, at::Tensor& input, ...) {
    EXEC_NPU_CMD(aclnnRmsNorm, input, gamma, epsilon, output);
}
```

2. **Custom AscendC ops** — write new kernel code:
```cpp
// op_kernel/fused_add_rmsnorm.cpp  (runs on NPU)
__aicore__ void fused_add_rmsnorm_kernel(...) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQue;
    // ... CopyIn → Compute → CopyOut pipeline
}

// op_host/fused_add_rmsnorm.cpp  (runs on CPU)
void launch_fused_add_rmsnorm(aclrtStream stream, ...) {
    // compile kernel → get .o → launch via aclrtLaunchKernel
}
```

**Build output**: Compiled to `.so` installed at `vllm_ascend/cann_ops_custom/`, dynamically loaded via `torch.ops.load_library()`.

### 9.4 SGLang-Kernel-NPU — Kernel Format Analysis

**Repository**: [sgl-project/SGLang-Kernel-NPU](https://github.com/sgl-project/SGLang-Kernel-NPU)

**Kernel library includes**:
- Multi-Latent Attention (MLA) with Paged KV Cache
- Grouped Query Attention (GQA) Decode
- Flash Linear Attention
- RMSNorm, Fused Add+RMSNorm+Bias
- SwiGLU, Quantized SwiGLU
- LoRA adapters
- AWQ/GPTQ dequantization
- DeepEP-Ascend (Expert Parallelism MoE kernels using HCCS/RDMA)

**Implementation pattern**: Same as vLLM — AscendC kernels compiled to `.so`, linked against `torch_npu`, called via `aclnn*` or `aclrtLaunchKernel`.

**Requirements**: CANN ≥ 8.2.RC1, torch_npu ≥ 2.5.1-7.0.0, Atlas A2/A3 series.

### 9.5 The `aclnn*` Two-Stage API — THE Key FFI Target

This is the most important API surface for Rust bindings. Every Ascend operator follows this pattern:

```c
// Header: aclnn/aclnn_{op_name}.h (from CANN SDK)
// Library: libopapi.so

// Stage 1: Calculate workspace requirements
aclnnStatus aclnnMatMulGetWorkspaceSize(
    const aclTensor* self,           // input tensor descriptor
    const aclTensor* mat2,           // input tensor descriptor
    const aclTensor* out,            // output tensor descriptor
    int8_t cubeMathType,             // compute precision
    uint64_t* workspaceSize,         // [OUT] required workspace bytes
    aclOpExecutor** executor         // [OUT] opaque executor handle
);

// Stage 2: Execute the operator
aclnnStatus aclnnMatMul(
    void* workspace,                 // pre-allocated device memory
    uint64_t workspaceSize,          // from stage 1
    aclOpExecutor* executor,         // from stage 1
    aclrtStream stream               // execution stream
);
```

**Key types for FFI**:

| C Type | Description | Rust FFI |
|--------|-------------|----------|
| `aclTensor*` | Opaque tensor descriptor | `*mut c_void` or newtype |
| `aclScalar*` | Opaque scalar descriptor | `*mut c_void` or newtype |
| `aclIntArray*` | Opaque int array | `*mut c_void` or newtype |
| `aclOpExecutor*` | Opaque executor handle | `*mut c_void` or newtype |
| `aclrtStream` | Execution stream (typedef `void*`) | `*mut c_void` or newtype |
| `aclnnStatus` | Return code (int enum) | `c_int` |
| `aclDataType` | Tensor data type enum | `c_int` enum |
| `aclFormat` | Tensor memory format enum | `c_int` enum |

**Tensor creation** (needed before calling aclnn ops):
```c
// Create tensor descriptor from raw device memory
aclTensor* aclCreateTensor(
    const int64_t* viewDims,         // shape
    uint64_t viewDimsNum,            // ndim
    aclDataType dataType,            // e.g., ACL_FLOAT16
    const int64_t* strides,          // strides
    int64_t offset,                  // byte offset
    aclFormat format,                // e.g., ACL_FORMAT_ND
    const int64_t* storageDims,      // storage shape
    uint64_t storageDimsNum,         // storage ndim
    void* tensorData                 // device memory pointer
);

void aclDestroyTensor(const aclTensor* tensor);
```

### 9.6 Three Strategies for Calling Existing Kernels from Rust

#### Strategy A: Direct `aclnn*` FFI (Recommended)

Bypass PyTorch/torch_npu entirely. Call `aclnn*` C functions directly from Rust.

```
Rust Code → FFI → libopapi.so (aclnn* operators)
                → libascendcl.so (runtime: device, memory, streams)
```

**Rust wrapper example**:
```rust
// In ascendcl-sys crate, link against libopapi.so + libascendcl.so
extern "C" {
    fn aclnnMatMulGetWorkspaceSize(
        self_: *const AclTensor,
        mat2: *const AclTensor,
        out: *const AclTensor,
        cube_math_type: i8,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> i32;

    fn aclnnMatMul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> i32;
}

// Safe wrapper:
pub fn matmul(
    stream: &Stream,
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
) -> Result<(), AclError> {
    let mut ws_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = std::ptr::null_mut();

    // Stage 1
    check_acl(unsafe {
        aclnnMatMulGetWorkspaceSize(
            a.raw(), b.raw(), out.raw(), 0,
            &mut ws_size, &mut executor,
        )
    })?;

    // Allocate workspace
    let workspace = DeviceBuffer::alloc(ws_size as usize)?;

    // Stage 2
    check_acl(unsafe {
        aclnnMatMul(workspace.ptr(), ws_size, executor, stream.raw())
    })?;

    Ok(())
}
```

**Pros**: No Python dependency, no torch overhead, minimal binary size, full control.
**Cons**: Need to build `aclTensor` descriptors manually, need to manage memory yourself.

**Libraries to link**:
```
libascendcl.so    → runtime (device, memory, stream)
libopapi.so       → aclnn* operator APIs
liboptiling.so    → operator tiling (auto-loaded)
libnnopbase.so    → operator base support
```

#### Strategy B: Call Pre-compiled Custom `.so` from vLLM/SGLang

Load the custom operator `.so` files built by vLLM Ascend or SGLang-Kernel-NPU and call their C functions directly.

```
Rust Code → dlopen() → vllm_ascend custom_ops.so → NPU execution
```

```rust
use libloading::{Library, Symbol};

let lib = unsafe { Library::new("path/to/custom_ops.so")? };
let fused_rmsnorm: Symbol<unsafe extern "C" fn(...)> =
    unsafe { lib.get(b"fused_add_rmsnorm")? };
```

**Pros**: Get optimized fused kernels for free (flash attention, MLA, etc.)
**Cons**: Tight coupling to vLLM/SGLang build system and versions; interface may not be stable C ABI.

#### Strategy C: Load `.om` Offline Model via AscendCL Model API

For whole-model execution (not individual operators):

```rust
// Load pre-compiled model
let model_id = unsafe { aclmdlLoadFromFile(b"model.om\0".as_ptr()) };
// Create input/output data structures
// Execute
unsafe { aclmdlExecute(model_id, input_dataset, output_dataset) };
```

**Pros**: Simple for whole-model inference; CANN optimizes the model graph.
**Cons**: No fine-grained control over individual kernels; not suitable for custom scheduling like continuous batching.

### 9.7 Recommended Approach for Your LLM Server

For `rust_llm_server_design_v8.md`, the recommended layered approach is:

```
┌──────────────────────────────────────────────────┐
│         Your Rust LLM Server (v8)                │
│  Scheduling, Batching, KV Cache, CXL hierarchy   │
├──────────────────────────────────────────────────┤
│  ascendcl crate (safe Rust wrapper)              │
│  ┌───────────────────┬──────────────────────┐    │
│  │ Runtime Module     │ Operator Module       │    │
│  │ Device, Stream,    │ matmul(), softmax(),  │    │
│  │ Memory, Context    │ attention(), rmsnorm()│    │
│  └───────────────────┴──────────────────────┘    │
├──────────────────────────────────────────────────┤
│  ascendcl-sys crate (raw FFI)                    │
│  bindgen from acl.h + aclnn headers              │
│  links: libascendcl.so + libopapi.so             │
├──────────────────────────────────────────────────┤
│  CANN SDK (installed on system)                  │
└──────────────────────────────────────────────────┘
```

**Phase plan (updated)**:

| Phase | What | Key APIs | Effort |
|-------|------|----------|--------|
| 1 | `ascendcl-sys` FFI | `acl.h`, `acl_rt.h` | 1-2 weeks |
| 2 | `aclnn-sys` FFI | `aclnn/*.h` headers from `libopapi.so` | 1-2 weeks |
| 3 | Safe `ascendcl` wrapper (runtime) | Device, Stream, Memory, Context | 2-3 weeks |
| 4 | Safe `aclnn` wrapper (operators) | matmul, softmax, attention, rmsnorm, etc. | 2-3 weeks |
| 5 | LLM server integration | v8 `BlockStore`, forward pass | 2-4 weeks |
| 6 | Custom AscendC kernels (C++) | Flash attention, fused ops | Ongoing |

### 9.8 Summary: What Exists vs What You Need

| Component | Exists? | Format | Callable from Rust? |
|-----------|---------|--------|-------------------|
| CANN built-in operators (matmul, softmax, etc.) | ✅ Yes | `aclnn*` C API in `libopapi.so` | ✅ Yes, via FFI |
| torch_npu operator wrappers | ✅ Yes | C++ in `libtorch_npu.so` | ⚠️ C++ mangling; use `aclnn*` directly instead |
| vLLM Ascend custom ops (fused kernels) | ✅ Yes | C++ `.so` with torch bindings | ⚠️ Possible but fragile; better to rewrite calls using `aclnn*` |
| SGLang-Kernel-NPU (MLA, GQA, etc.) | ✅ Yes | AscendC `.so` + torch bindings | ⚠️ Same as vLLM — possible but fragile |
| AscendCL runtime (device, memory, streams) | ✅ Yes | C API in `libascendcl.so` | ✅ Yes, via `bindgen` FFI |
| Rust FFI bindings for any of the above | ❌ No | — | Need to create |
| `ascend-rs` (Rust kernels) | 🔒 Private | Rust → MLIR → C++ → NPU | ❌ Not available |

---

## 10. References

- [ascend-rs.org](https://ascend-rs.org) — Official documentation for the `ascend-rs` framework
- [AscendCL API Documentation](https://www.hiascend.com/en/document) — Official Huawei CANN documentation
- [AscendCL Samples on GitHub](https://github.com/Ascend/samples) — C/C++ sample code
- [CANN SDK](https://www.hiascend.com/software/cann) — Download page for CANN toolkit
- [vllm-ascend](https://github.com/vllm-project/vllm-ascend) — vLLM Ascend plugin (custom ops in `csrc/`)
- [torch_npu](https://gitee.com/ascend/pytorch) — Ascend PyTorch adapter
- [SGLang-Kernel-NPU](https://github.com/sgl-project/SGLang-Kernel-NPU) — SGLang Ascend kernel library
- [CANN aclnn API Reference](https://www.hiascend.com/en/document) — Two-stage operator API documentation
