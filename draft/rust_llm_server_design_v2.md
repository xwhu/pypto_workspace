# Rust LLM Inference Server — Design & Implementation Plan (v2)

> **Target:** 1–2 machines, 8–16 GPU cards, high-performance inference for agent/coding workloads  
> **Philosophy:** Rust for orchestration and scheduling. Reuse CUDA kernels via FFI.  
> **New in v2:** Pipeline Parallelism, multiple quantization schemes, and a **configuration generator** that produces a specialized binary per deployment — no runtime branching over parallelism or quant modes.

---

## Table of Contents

1. [The Generator Idea](#1-the-generator-idea)
2. [Configuration Space](#2-configuration-space)
3. [Generator Design](#3-generator-design)
4. [Architecture Overview](#4-architecture-overview)
5. [Pipeline Parallelism Design](#5-pipeline-parallelism-design)
6. [Quantization Schemes](#6-quantization-schemes)
7. [Kernel Reuse Map](#7-kernel-reuse-map)
8. [Component Design](#8-component-design)
9. [Key Rust Crates](#9-key-rust-crates)
10. [Implementation Phases](#10-implementation-phases)
11. [Directory Layout](#11-directory-layout)
12. [Performance Targets](#12-performance-targets)

---

## 1. The Generator Idea

### The Problem with a Universal Binary

A single binary that handles all combinations at runtime looks like this:

```rust
// Bad: runtime branching everywhere
match config.quant {
    Quant::BF16 => run_gemm_bf16(...),
    Quant::W8A8 => run_gemm_int8(...),
    Quant::W4A16 => run_gemm_awq(...),
    Quant::FP8 => run_gemm_fp8(...),
}
match config.parallelism {
    Parallelism::TP => run_tp_forward(...),
    Parallelism::PP => run_pp_forward(...),
    Parallelism::TPPP => run_tppp_forward(...),
}
```

Problems: dead code in every binary, harder to optimize each path, harder to test, and every kernel variant must be compiled in regardless of use.

### The Generator Approach

Instead, you write a **generator** (a small Rust or Python CLI) that takes a YAML config describing exactly your hardware setup and desired features, then emits a *specialized* Rust project with:

- Compile-time constants baked in (`const TP_SIZE: usize = 8;`)
- Only the kernel variants your config needs compiled
- A `Cargo.toml` with exactly the right feature flags enabled
- A minimal, readable generated `config.rs` — no macros, no trait objects, just direct code

Then you `cargo build --release` and get a lean, specialized binary.

```
llm-gen config.yaml → specialized_project/ → cargo build → llm-server-tp8-pp2-w8a8
```

This is the same philosophy as:
- CUTLASS's Python-based kernel generator (generates C++ specializations)
- JAX's `jit` compilation (specializes on shape + dtype)
- Triton's autotuning (generates optimized variants per config)

---

## 2. Configuration Space

These are the dimensions the generator handles:

### Parallelism

| Mode | TP | PP | Use Case |
|---|---|---|---|
| `tp_only` | 1–8 | 1 | Single node, <70B models |
| `tp_pp` | 1–8 | 2–4 | Two nodes, 70B–405B models |
| `dp_tp` | N replicas | 1 | High throughput, smaller models |

### Quantization

| Scheme | Precision | Memory Saving | Speed | Best For |
|---|---|---|---|---|
| `bf16` | BF16 | 1× baseline | Baseline | Accuracy-sensitive |
| `fp8` | FP8 (E4M3) | ~2× | ~1.5× | H100 native, best trade-off |
| `w8a8` | INT8 weights + INT8 activations | ~2× | ~1.3× | Ampere (A100) |
| `w4a16` | INT4 weights + BF16 activations | ~4× | ~1.2× | Memory-constrained deployments |

### Example configs

```yaml
# config-single-node-coding.yaml
# 8×H100, coding assistant, large context
parallelism:
  tp: 8
  pp: 1
quant: fp8
model_family: llama   # llama / qwen / deepseek
max_seq_len: 32768
max_batch_tokens: 16384
max_prefill_tokens: 4096
block_size: 16
kv_cache_gb: 60
cpu_swap_gb: 32
```

```yaml
# config-two-node-agent.yaml
# 16×H100 across 2 nodes, agent workload
parallelism:
  tp: 8
  pp: 2
quant: fp8
model_family: deepseek
max_seq_len: 131072
max_batch_tokens: 32768
max_prefill_tokens: 8192
block_size: 16
kv_cache_gb: 55
cpu_swap_gb: 64
nccl:
  rdma_device: mlx5_0
  master_addr: 10.0.0.1
  master_port: 29500
```

---

## 3. Generator Design

### What the Generator Produces

The generator takes `config.yaml` and writes a complete Rust project:

```
generated/
  Cargo.toml             ← features matching the config
  build.rs               ← compiles only the needed kernel variants
  src/
    config.rs            ← const TP_SIZE, PP_SIZE, QUANT, etc.
    main.rs              ← wired to the right scheduler + executor variant
  kernels/
    CMakeLists.txt        ← only includes needed .cu files
```

### Generated `config.rs` (example for TP=8, PP=2, FP8)

```rust
// AUTO-GENERATED — do not edit manually
// Source: config-two-node-agent.yaml

pub const TP_SIZE: usize = 8;
pub const PP_SIZE: usize = 2;
pub const WORLD_SIZE: usize = TP_SIZE * PP_SIZE;  // 16

pub const BLOCK_SIZE: usize = 16;
pub const MAX_SEQ_LEN: usize = 131072;
pub const MAX_BATCH_TOKENS: usize = 32768;
pub const MAX_PREFILL_TOKENS: usize = 8192;

pub type WeightDtype = half::bf16;    // weights stored as BF16
pub type ComputeDtype = f8e4m3;       // FP8 compute
pub type KvDtype = half::bf16;        // KV cache stored as BF16

pub const IS_TP_ONLY: bool = false;
pub const IS_PP_ENABLED: bool = true;
pub const PP_RANK_IS_FIRST: bool = false;  // overridden per-process at startup
pub const PP_RANK_IS_LAST: bool = false;

pub const NCCL_RDMA_DEVICE: &str = "mlx5_0";
pub const NCCL_MASTER_ADDR: &str = "10.0.0.1";
pub const NCCL_MASTER_PORT: u16 = 29500;
```

The rest of the codebase imports `crate::config::*`. The compiler sees all of these as constants — dead branches get eliminated, monomorphization is free.

### Generator Implementation

The generator itself is a ~300-line Rust CLI:

```
llm-gen/
  src/
    main.rs         ← parse config.yaml, call each emitter
    emit_config.rs  ← write config.rs
    emit_cargo.rs   ← write Cargo.toml with right features
    emit_cmake.rs   ← write CMakeLists.txt with right kernel list
    emit_main.rs    ← write main.rs with right imports
  templates/        ← Tera templates for each generated file
```

Usage:

```bash
# Generate a project
cargo run --bin llm-gen -- config.yaml --output ./build/my-deployment

# Build it
cd ./build/my-deployment && cargo build --release

# Run it
./target/release/llm-server --model /models/deepseek-v3 --tp-rank 0 --pp-rank 0
```

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     llm-gen (code generator)                        │
│  config.yaml → generated project (config.rs + Cargo.toml + cmake)  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ generates
                    ┌────────────▼────────────┐
                    │    Specialized Binary   │
                    └────────────┬────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                      HTTP API Layer (axum)                          │
│              POST /v1/chat/completions (SSE streaming)              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                      Scheduler (Rust)                               │
│  Continuous batching · Chunked prefill · Prefix cache lookup        │
│  Micro-batch formation (PP) · Preemption                            │
└──────────────┬────────────────────────────┬─────────────────────────┘
               │                            │
┌──────────────▼──────────┐   ┌─────────────▼───────────────────────┐
│    KV Block Manager     │   │     Pipeline Executor (Rust)         │
│  GPU + CPU block pool   │   │  Splits batch into PP_SIZE stages    │
│  Prefix hash table      │   │  Manages inter-stage KV handoff      │
│  LRU eviction           │   │  NCCL send/recv between PP ranks     │
└─────────────────────────┘   └──────────────┬──────────────────────┘
                                             │ one thread per GPU
                              ┌──────────────▼──────────────────────┐
                              │   TP Workers [tp_rank 0..TP_SIZE)   │
                              │   Each GPU: Attention → FFN         │
                              │   NCCL AllReduce for TP             │
                              └──────────────┬──────────────────────┘
                                             │ FFI
┌────────────────────────────────────────────▼──────────────────────┐
│                  CUDA Kernel Library (C/CUDA)                      │
│  FlashAttn-3  PagedAttn  cuBLAS  FP8-GEMM  INT8-GEMM  AWQ  NCCL  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 5. Pipeline Parallelism Design

### What PP Does

PP splits the model's layers across PP_SIZE processes (ranks). Each rank holds `num_layers / PP_SIZE` consecutive transformer layers. A batch is split into micro-batches that flow through the pipeline stages, enabling overlap between stages (GPT-style 1F1B scheduling).

### PP in a 2-Node Setup (TP=8, PP=2)

```
Node 0 (PP rank 0, TP ranks 0–7):  layers 0–39  (first 40 of 80)
Node 1 (PP rank 1, TP ranks 0–7):  layers 40–79 (last 40 of 80)

Micro-batch flow:
  Scheduler → Node 0 → [NCCL send activations] → Node 1 → logits → Sampler
                ↑                                                        |
                └──────────────── output token ──────────────────────────┘
```

### 1F1B Micro-Batch Scheduling

To keep all PP stages busy, the scheduler decomposes each step's batch into `PP_SIZE` micro-batches and pipelines them:

```
Time →    T1        T2        T3        T4        T5
Stage 0:  F(mb1)    F(mb2)    B(mb1)    B(mb2)    idle
Stage 1:  idle      F(mb1)    F(mb2)    B(mb1)    B(mb2)

F = forward pass, B = backward (not needed for inference)
```

For inference-only (no backward), 1F1B simplifies to:

```rust
// PP executor step:
for micro_batch in batch.split(PP_SIZE) {
    if pp_rank == 0 {
        let activations = stage.forward(micro_batch);
        nccl.send(activations, dest = pp_rank + 1).await;
    } else if pp_rank == PP_SIZE - 1 {
        let activations = nccl.recv(src = pp_rank - 1).await;
        let logits = stage.forward(activations);
        sampler.send(logits).await;
    } else {
        let activations = nccl.recv(src = pp_rank - 1).await;
        let activations = stage.forward(activations);
        nccl.send(activations, dest = pp_rank + 1).await;
    }
}
```

### KV Cache with PP

Each PP rank only holds KV blocks for its own layers. The KV Block Manager is instantiated per-rank with `num_heads / PP_SIZE` (actually the full heads for that rank's layers — PP does not split heads, it splits layers).

PP rank 0's KV cache: layers 0–39 KV  
PP rank 1's KV cache: layers 40–79 KV  

The prefix cache hash is computed from tokens (not layers), so prefix hits work identically regardless of PP configuration.

### PP Inter-Stage Communication

Between PP stages, the activation tensor (shape: `[batch, seq, hidden_dim]`) is transferred via NCCL point-to-point:

```rust
// nccl point-to-point (available in NCCL 2.7+)
nccl_comm.send(tensor.as_ptr(), count, dtype, peer = next_rank, stream)?;
nccl_comm.recv(tensor.as_mut_ptr(), count, dtype, peer = prev_rank, stream)?;
```

For intra-node (both PP stages on the same machine), prefer CUDA IPC over NCCL P2P for lower latency. The generator emits the right variant based on config.

---

## 6. Quantization Schemes

### FP8 (Recommended for H100)

H100 has native FP8 tensor cores (E4M3 format). vLLM's FP8 kernels (via `csrc/quantization/fp8/`) implement:
- **Static FP8:** scales computed offline, best throughput
- **Dynamic FP8:** scales computed per-token, better accuracy

What the generator emits for `quant: fp8`:
```rust
// config.rs (generated)
pub type WeightDtype = fp8_e4m3;
pub type ActivDtype = fp8_e4m3;
pub const QUANT_SCALE_MODE: &str = "static";
```

The model's weight tensors are stored as FP8. Each GEMM call dispatches to `cutlass_fp8_gemm` instead of `cublas_gemm_bf16`.

### W8A8 (INT8, for A100)

Weights stored as INT8, activations quantized to INT8 on-the-fly before each GEMM. Uses vLLM's CUTLASS W8A8 kernels (`csrc/quantization/cutlass_w8a8/`). Per-channel weight scales and per-token activation scales stored alongside weights.

### W4A16 (AWQ/GPTQ, memory-constrained)

Weights packed into 4-bit groups (group size 128 by default). Activations remain BF16. Uses vLLM's AWQ/GPTQ kernels (`csrc/quantization/awq/`, `csrc/quantization/gptq/`). The dequantization fused with GEMM. Ideal for fitting large models on fewer GPUs.

### What Changes Per Quant Scheme

| Component | BF16 | FP8 | W8A8 | W4A16 |
|---|---|---|---|---|
| Weight dtype at load | BF16 | FP8 | INT8 | INT4 packed |
| GEMM kernel | cuBLAS BF16 | CUTLASS FP8 | CUTLASS W8A8 | AWQ kernel |
| Quantization scales | None | Per-tensor or per-channel | Per-channel + per-token | Per-group |
| KV cache dtype | BF16 | BF16 (or FP8) | BF16 | BF16 |
| Memory per param | 2B | 1B | 1B | 0.5B |

The generator's CMakeLists only compiles the kernel variant for the configured quant scheme. A W4A16 build does not include FP8 kernel code at all.

---

## 7. Kernel Reuse Map

| Kernel | Source Project | Notes |
|---|---|---|
| FlashAttention 2/3 | `Dao-AILab/flash-attention` | Core attention |
| PagedAttention V2 | `vllm-project/vllm` `csrc/attention/` | Variable-length KV |
| RMSNorm | `vllm-project/vllm` `csrc/layernorm/` | |
| RoPE (rotary pos enc) | `vllm-project/vllm` `csrc/pos_encoding/` | |
| BF16/FP16 GEMM | cuBLAS | `libcublas.so` |
| FP8 GEMM | `vllm-project/vllm` `csrc/quantization/fp8/` | H100 only |
| W8A8 INT8 GEMM | `vllm-project/vllm` `csrc/quantization/cutlass_w8a8/` | |
| W4A16 AWQ | `vllm-project/vllm` `csrc/quantization/awq/` | |
| W4A16 GPTQ | `vllm-project/vllm` `csrc/quantization/gptq/` | |
| Top-p/top-k sampling | `vllm-project/vllm` `csrc/sampling_kernels.cu` | |
| AllReduce (TP) | NCCL `libnccl.so` | |
| P2P send/recv (PP) | NCCL `libnccl.so` | |
| Tokenization | `huggingface/tokenizers` Rust crate | Pure Rust |

The generator's `emit_cmake.rs` builds a `CMakeLists.txt` that includes only the kernel `.cu` files needed by the chosen quant scheme.

---

## 8. Component Design

### 8.1 KV Block Manager (unchanged from v1)

```rust
const BLOCK_SIZE: usize = 16; // from config.rs

struct BlockPool {
    gpu_memory: CudaBuffer,            // pre-allocated: [num_blocks, 2, num_heads, BLOCK_SIZE, head_dim]
    cpu_memory: PinnedHostBuffer,      // for overflow / preemption swap
    free_gpu: VecDeque<u32>,
    free_cpu: VecDeque<u32>,
    hash_to_block: DashMap<u64, u32>,  // prefix cache
    lru: LinkedList<u64>,
}
```

### 8.2 Scheduler

The scheduler handles chunked prefill and is PP-aware: it composes micro-batches of size `batch / PP_SIZE` before dispatching.

```rust
const MAX_PREFILL_TOKENS: usize = crate::config::MAX_PREFILL_TOKENS;
const PP_SIZE: usize = crate::config::PP_SIZE;

fn build_step_batch(&mut self) -> StepBatch {
    let decode_seqs = self.active_decode_seqs();
    let prefill_chunk = self.take_prefill_chunk(MAX_PREFILL_TOKENS);

    // For PP: split into PP_SIZE micro-batches
    StepBatch {
        micro_batches: split_into_micro_batches(decode_seqs, prefill_chunk, PP_SIZE),
    }
}
```

### 8.3 Pipeline Executor

New component (PP_SIZE > 1 only — generated away for TP-only builds):

```rust
struct PipelineExecutor {
    pp_rank: usize,
    tp_workers: Vec<TpWorker>,   // TP_SIZE workers for this PP stage
    nccl_pp_comm: NcclComm,      // point-to-point comm between PP ranks
}

impl PipelineExecutor {
    async fn run_step(&self, step_batch: &StepBatch) {
        for micro_batch in &step_batch.micro_batches {
            let activations = match self.pp_rank {
                0 => {
                    // First stage: embed input tokens
                    let embed = self.embed(micro_batch.token_ids);
                    let out = self.run_layers(embed);
                    self.nccl_pp_comm.send(out, self.pp_rank + 1).await;
                    continue;
                }
                r if r == PP_SIZE - 1 => {
                    // Last stage: receive, run layers, emit logits
                    let act = self.nccl_pp_comm.recv(self.pp_rank - 1).await;
                    let out = self.run_layers(act);
                    self.sampler_tx.send(out).await;
                }
                _ => {
                    // Middle stage: receive, run layers, forward
                    let act = self.nccl_pp_comm.recv(self.pp_rank - 1).await;
                    let out = self.run_layers(act);
                    self.nccl_pp_comm.send(out, self.pp_rank + 1).await;
                }
            };
        }
    }
}
```

For **TP-only builds** (PP_SIZE = 1), the generator emits a simpler `SimpleExecutor` with no PP logic at all.

### 8.4 TP Workers

Each worker holds its shard of the PP stage's layers. The worker dispatch is identical regardless of quant scheme — the difference is which GEMM function pointer is called, and this is resolved at compile time via the generated `config.rs` constants.

```rust
fn linear(&self, x: &CudaTensor, weight: &WeightTensor) -> CudaTensor {
    // WeightTensor is a type alias resolved at compile time:
    //   BF16 build:  WeightTensor = CudaTensor<bf16>      → cublas_gemm_bf16
    //   FP8 build:   WeightTensor = CudaTensor<fp8_e4m3>  → cutlass_fp8_gemm
    //   W8A8 build:  WeightTensor = QuantizedTensor<i8>   → cutlass_w8a8_gemm
    //   W4A16 build: WeightTensor = PackedInt4Tensor       → awq_gemm
    kernels::gemm(x, weight, &self.stream)
}
```

The `kernels::gemm` function is itself generated — it calls the right FFI function based on `WeightDtype` from `config.rs`. No match statement, no vtable.

### 8.5 CUDA Graph Capture

Decode steps (small, fixed batch sizes) are captured as CUDA graphs. With PP enabled, each PP stage captures its own graph. The graph boundaries are at the layer entry/exit points, and the NCCL P2P send/recv is outside the graph (graphs can't capture inter-process communication).

```rust
// Capture decode graphs at warm-up
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128] {
    let graph = cuda_graph_capture(|| {
        self.run_layers_decode(dummy_batch(batch_size));
        // Note: nccl send/recv is NOT inside the graph
    });
    self.graph_cache.insert(batch_size, graph);
}
```

---

## 9. Key Rust Crates

| Crate | Purpose |
|---|---|
| `axum` | HTTP server, SSE streaming |
| `tokio` | Async runtime |
| `safetensors` | Load model weights |
| `tokenizers` | Tokenization (HuggingFace) |
| `cudarc` | Safe Rust CUDA memory + stream management |
| `half` | BF16/FP16 types |
| `sha2` | SHA256 for prefix block hashing |
| `dashmap` | Concurrent prefix cache hashmap |
| `serde` / `serde_json` | Config parsing and API types |
| `tera` | Template engine for code generator |
| `clap` | CLI argument parsing (for llm-gen) |
| `tracing` | Structured logging |
| `prometheus` | Metrics |
| `bindgen` (build dep) | FFI bindings from `llm_kernels.h` |
| `cmake` (build dep) | Drive CUDA kernel compilation |

---

## 10. Implementation Phases

### Phase 1 — Generator + Single-GPU BF16 (3 weeks)

**Goal:** The generator works, and you can serve a single request end-to-end.

- [ ] `llm-gen` CLI: parses `config.yaml`, emits `config.rs` + `Cargo.toml` + `CMakeLists.txt`
- [ ] Kernel build system: cmake + bindgen for FlashAttention + BF16 cuBLAS
- [ ] `CudaTensor` type wrapping `cudarc::DevicePtr`
- [ ] Safetensors weight loader with TP=1 sharding
- [ ] Llama model forward pass (BF16)
- [ ] HTTP server (axum), single request, non-streaming
- [ ] CLI test: `llm-gen config-tp1-bf16.yaml && cargo build && ./llm-server --model llama3-8b`

**Checkpoint:** End-to-end inference on Llama-3.1-8B, one GPU.

---

### Phase 2 — KV Cache + Prefix Caching (2 weeks)

- [ ] BlockPool with GPU pre-allocation
- [ ] SHA256-based prefix hash table (`DashMap<u64, u32>`)
- [ ] LRU eviction
- [ ] PagedAttention kernel wired in
- [ ] Verify >90% cache hit rate on repeated system prompts

**Checkpoint:** Multi-turn agent conversation with measurable cache benefit.

---

### Phase 3 — Continuous Batching + Chunked Prefill (2 weeks)

- [ ] Scheduler main loop with chunked prefill cap
- [ ] Batch builder for variable-length sequences
- [ ] SSE token streaming
- [ ] Preemption + CPU swap
- [ ] Load test: 50 concurrent sessions

**Checkpoint:** Stable latency under concurrent load.

---

### Phase 4 — TP=8 (1 week)

- [ ] NCCL intra-node init (`ncclCommInitAll`)
- [ ] TP weight sharding at load time
- [ ] Per-GPU worker threads, CUDA context isolation
- [ ] AllReduce after attention and FFN
- [ ] CUDA graph capture for decode

**Checkpoint:** Llama-3.1-70B on 8×H100 at expected throughput.

---

### Phase 5 — FP8 Quantization (1 week)

- [ ] Add FP8 kernel variant to cmake and bindgen
- [ ] Generator emits FP8 config + kernel selection
- [ ] Weight loader converts BF16 → FP8 at load time (or reads pre-quantized)
- [ ] Per-channel scale tensors stored alongside weights
- [ ] Benchmark: BF16 vs FP8 throughput + quality check

**Checkpoint:** FP8 build passes accuracy threshold and shows >1.3× throughput vs BF16.

---

### Phase 6 — Pipeline Parallelism: 2 Nodes (2 weeks)

- [ ] PP executor with micro-batch formation
- [ ] NCCL P2P send/recv between PP ranks
- [ ] SSH/MPI process launcher
- [ ] TCP rendezvous for NCCL `ncclUniqueId` broadcast
- [ ] Generator emits PP-aware executor (vs simple executor for TP-only)
- [ ] Test: DeepSeek-V3 across 2×8-GPU nodes

**Checkpoint:** TP=8, PP=2, FP8 serving a 671B MoE model end-to-end.

---

### Phase 7 — W8A8 and W4A16 (1 week each)

- [ ] Generator adds W8A8 / W4A16 config variants
- [ ] Add kernel variant to cmake (CUTLASS W8A8 / AWQ)
- [ ] Weight loader handles INT8 or INT4-packed tensors
- [ ] Scale tensors loaded and applied correctly
- [ ] Accuracy + throughput benchmark vs FP8

**Checkpoint:** W4A16 config serves a 405B model on 8 GPUs that wouldn't fit in BF16.

---

## 11. Directory Layout

```
llm-inference/
├── llm-gen/                        ← the code generator
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs                 ← CLI entry: parse config, call emitters
│   │   ├── emit_config.rs          ← writes config.rs
│   │   ├── emit_cargo.rs           ← writes Cargo.toml
│   │   ├── emit_cmake.rs           ← writes CMakeLists.txt
│   │   └── emit_main.rs            ← writes main.rs
│   └── templates/                  ← Tera templates
│       ├── config.rs.tera
│       ├── Cargo.toml.tera
│       └── CMakeLists.txt.tera
│
├── llm-server/                     ← the server library (consumed by generated projects)
│   ├── Cargo.toml                  ← no features; config comes from generated config.rs
│   ├── kernels/
│   │   ├── src/
│   │   │   ├── attention.cu        ← thin C wrappers for PagedAttn + FlashAttn
│   │   │   ├── layernorm.cu
│   │   │   ├── pos_encoding.cu
│   │   │   ├── sampling.cu
│   │   │   ├── gemm_bf16.cu        ← cuBLAS wrapper
│   │   │   ├── gemm_fp8.cu         ← CUTLASS FP8 wrapper
│   │   │   ├── gemm_w8a8.cu        ← CUTLASS W8A8 wrapper
│   │   │   └── gemm_w4a16.cu       ← AWQ wrapper
│   │   └── include/
│   │       └── llm_kernels.h
│   └── src/
│       ├── api/
│       │   ├── routes.rs
│       │   └── types.rs
│       ├── scheduler/
│       │   ├── mod.rs
│       │   ├── batch.rs
│       │   └── queue.rs
│       ├── kv_cache/
│       │   ├── block_pool.rs
│       │   ├── prefix_cache.rs
│       │   └── swap.rs
│       ├── executor/
│       │   ├── simple.rs           ← TP-only executor (no PP logic)
│       │   ├── pipeline.rs         ← PP executor (generated in when PP_SIZE > 1)
│       │   ├── worker.rs
│       │   └── nccl.rs
│       ├── models/
│       │   ├── mod.rs              ← Model trait
│       │   ├── llama.rs
│       │   ├── deepseek.rs
│       │   └── weights.rs
│       ├── kernels/                ← FFI wrappers (generated by bindgen)
│       │   ├── attention.rs
│       │   ├── gemm.rs
│       │   └── sampling.rs
│       └── cuda/
│           ├── tensor.rs
│           └── stream.rs
│
├── configs/                        ← example configs for common setups
│   ├── tp8-fp8-single-node.yaml
│   ├── tp8-pp2-fp8-two-node.yaml
│   ├── tp4-w4a16-4gpu.yaml
│   └── tp1-bf16-dev.yaml           ← development/debug config
│
└── build/                          ← generated output (gitignored)
    └── tp8-pp2-fp8/
        ├── Cargo.toml
        ├── build.rs
        ├── src/
        │   ├── config.rs           ← generated constants
        │   └── main.rs             ← generated entry point
        └── kernels/
            └── CMakeLists.txt      ← only FP8 + PagedAttn + FlashAttn
```

---

## 12. Performance Targets

For **Llama-3.1-70B on 8×H100 (FP8, TP=8)**:

| Metric | Target |
|---|---|
| Decode throughput | > 4,500 tok/sec (FP8 ~1.5× over BF16's ~3,000) |
| TTFT (512-token prompt, cold) | < 80ms |
| TTFT (512-token prompt, cached) | < 15ms |
| Prefix cache hit rate (agent workload) | > 85% |

For **DeepSeek-V3 (671B MoE) on 2×8×H100 (FP8, TP=8, PP=2)**:

| Metric | Target |
|---|---|
| Decode throughput | > 1,500 tok/sec |
| TTFT | < 300ms |
| PP pipeline bubble overhead | < 5% |

---

## Key Design Decisions — Rationale

**Generator vs feature flags:** Rust feature flags still compile-in all variants and use `#[cfg]` — the compiler eliminates dead code but the source is still complex. The generator approach means each deployment's source literally doesn't contain the other variants. Simpler to read, simpler to debug.

**PP via NCCL P2P, not shared memory:** NCCL P2P works both intra-node (fast, NVLink) and inter-node (RDMA). Shared memory only works intra-node. Using NCCL throughout means the same code handles both cases; the generator just sets the NCCL device string.

**FP8 as the recommended default (not W8A8):** On H100, FP8 has hardware support in tensor cores and requires no offline calibration for static quantization. W8A8 needs calibration data and is better suited to A100. W4A16 is for memory-constrained scenarios (fitting a 405B on 4 GPUs). The generator makes this choice explicit per deployment.

**PP stage boundary at layer granularity:** PP could also split at the token or head level, but layer-granularity is simplest, matches how all major frameworks implement it, and the generated code stays straightforward.

---

*The generator is the key new component in v2. Everything else (KV manager, scheduler, TP workers, kernel FFI) is the same as v1. The generator's job is to make each deployment configuration a first-class, specialized artifact rather than a runtime switch.*
