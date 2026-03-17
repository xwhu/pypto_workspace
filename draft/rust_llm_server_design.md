# Rust LLM Inference Server — Design & Implementation Plan

> **Target:** 1–2 machines, 8–16 GPU cards, high-performance inference for agent/coding workloads  
> **Philosophy:** Rust for orchestration, memory management, and scheduling. Reuse battle-tested CUDA kernels via FFI. No reinventing the wheel.

---

## Table of Contents

1. [Goals & Constraints](#1-goals--constraints)
2. [Architecture Overview](#2-architecture-overview)
3. [Kernel Reuse Strategy](#3-kernel-reuse-strategy)
4. [Component Design](#4-component-design)
5. [Key Rust Crates](#5-key-rust-crates)
6. [Implementation Phases](#6-implementation-phases)
7. [Directory Layout](#7-directory-layout)
8. [Performance Targets](#8-performance-targets)

---

## 1. Goals & Constraints

### Must-Have Features

| Feature | Why |
|---|---|
| **Automatic Prefix Caching** | Agents reuse system prompt + tool history on every turn |
| **Continuous Batching** | Keep GPU utilization high across concurrent requests |
| **Chunked Prefill** | Bound TTFT; mix small prefills with decode in the same step |
| **Tensor Parallelism** | Scale across 8–16 GPUs within and across 2 machines |
| **OpenAI-compatible HTTP API** | Drop-in for existing agent frameworks |
| **KV Cache Tiering** | GPU HBM → CPU DRAM overflow when context is long |

### Explicit Non-Goals (for now)

- Pipeline Parallelism (complex, not needed for 1–2 nodes)
- RLHF / weight synchronization (separate concern)
- Support for every quantization scheme (start with BF16/FP16, add W8A8 later)
- Custom kernel development (reuse existing ones)

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    HTTP API Layer (axum)                         │
│          POST /v1/chat/completions  POST /v1/completions         │
└────────────────────────────┬─────────────────────────────────────┘
                             │ async channel
┌────────────────────────────▼─────────────────────────────────────┐
│                      Scheduler (Rust)                            │
│  • Continuous batching loop (step every ~5–10ms)                 │
│  • Chunked prefill: max_prefill_tokens per step                  │
│  • Priority queue: prefill slots vs decode slots                  │
│  • Prefix cache lookup before allocating new blocks              │
└──────────┬─────────────────────────────────┬─────────────────────┘
           │ allocate/free blocks            │ dispatch batch
┌──────────▼────────────┐        ┌───────────▼─────────────────────┐
│   KV Block Manager    │        │     Model Executor (Rust)       │
│   (Rust)              │        │  • Shards input across TP ranks │
│  • Block pool (GPU +  │        │  • Sends tensors to each Worker │
│    CPU overflow)      │        │  • Collects logits from rank 0  │
│  • Prefix hash table  │        │  • Runs Sampler                 │
│    (SHA256 per block) │        └───────────┬─────────────────────┘
│  • LRU eviction       │                    │ one thread per GPU
└───────────────────────┘        ┌───────────▼─────────────────────┐
                                 │  TP Workers [0..N] (Rust thread)│
                                 │  Each owns one CUDA context      │
                                 │  Runs: Attn → FFN → AllReduce   │
                                 └───────────┬─────────────────────┘
                                             │ FFI calls
┌────────────────────────────────────────────▼─────────────────────┐
│                  CUDA Kernel Library (C/CUDA)                    │
│  FlashAttention-3  │  PagedAttn  │  cuBLAS  │  RMSNorm  │  NCCL  │
└──────────────────────────────────────────────────────────────────┘
```

### Single-Node vs Two-Node

**Single node (8 GPUs):**  
One process, 8 OS threads (one per GPU). NCCL intra-node via NVLink or PCIe.

**Two nodes (8+8 GPUs = 16 GPUs):**  
One process per node. Rank 0 is the master (runs scheduler, HTTP server). Both nodes start 8 worker threads each. NCCL handles AllReduce across the RDMA link. Use `mpirun -n 2` or a simple SSH launcher to start both processes.

---

## 3. Kernel Reuse Strategy

The rule: **write zero CUDA code**. Every kernel comes from an existing, maintained project.

### Kernel Source Map

| Kernel | Source | License | How to Reuse |
|---|---|---|---|
| **Flash Attention 2/3** | `Dao-AILab/flash-attention` | BSD-3 | Compile `csrc/flash_attn` → shared lib, call via C FFI |
| **PagedAttention** | `vllm-project/vllm` | Apache-2 | Compile `csrc/attention/` → shared lib |
| **RMSNorm, SiLU, RoPE** | `vllm-project/vllm` | Apache-2 | Same, `csrc/layernorm`, `csrc/pos_encoding` |
| **GEMM (BF16/FP16)** | cuBLAS | NVIDIA | Link `libcublas.so` via `cublas-sys` crate |
| **W8A8 GEMM (INT8)** | `vllm-project/vllm` | Apache-2 | `csrc/quantization/cutlass_w8a8/` |
| **AllReduce / AllGather** | NCCL | BSD-3 | Link `libnccl.so` via `nccl-sys` crate |
| **Sampling (top-p/top-k)** | `vllm-project/vllm` | Apache-2 | `csrc/sampling_kernels.cu` |
| **Tokenization** | `huggingface/tokenizers` | Apache-2 | Pure Rust crate, no CUDA |

### Build Strategy

Create a `kernels/` subdirectory. Write a `build.rs` that:
1. Runs `cmake` to compile the vLLM and FlashAttention CUDA sources into `libllm_kernels.so`
2. Generates a Rust FFI binding via `bindgen`
3. Links the resulting `.so` into the Rust binary

```
kernels/
  CMakeLists.txt           ← cmake build for all CUDA kernels
  src/
    attention.cu           ← thin C wrappers around FA3 / PagedAttn
    layernorm.cu           ← thin wrappers around vLLM kernels
    pos_encoding.cu
    sampling.cu
  include/
    llm_kernels.h          ← C header for Rust bindgen
```

**The wrappers are 10–20 lines each.** They just adapt the calling convention so bindgen can generate clean Rust bindings. Example:

```c
// attention.cu wrapper
void paged_attention_v2(
    void* out,           // [num_seqs, num_heads, head_size]
    void* query,
    void** key_cache,
    void** value_cache,
    int32_t* block_tables,
    int32_t* context_lens,
    int max_context_len,
    int num_seqs,
    int num_heads,
    int head_size,
    int block_size,
    cudaStream_t stream
);
```

---

## 4. Component Design

### 4.1 KV Block Manager

This is the core innovation — implement it carefully in safe Rust.

```rust
const BLOCK_SIZE: usize = 16; // tokens per block

struct Block {
    block_id: u32,
    ref_count: u32,
    hash: u64,          // SHA256 of token IDs (prefix chain)
    device: Device,     // Gpu(id) or Cpu
}

struct BlockPool {
    // Pre-allocated CUDA memory: [num_blocks, 2, num_heads, block_size, head_dim]
    // Index 0 = K blocks, Index 1 = V blocks
    gpu_memory: CudaBuffer,
    cpu_memory: Vec<u8>,       // pinned host memory for overflow

    free_gpu_blocks: VecDeque<u32>,
    free_cpu_blocks: VecDeque<u32>,

    // Prefix cache
    hash_to_block: HashMap<u64, u32>,   // hash → block_id (GPU only)
    lru_order: LinkedList<u64>,          // for eviction
}
```

**Prefix cache lookup** before scheduling a new request:
```
for each 16-token chunk of the incoming prompt:
    hash = sha256(prev_hash || token_ids[chunk])
    if hash_to_block.contains(hash):
        reuse_block(hash)        ← free! skip compute
        cache_hit_count += 1
    else:
        allocate_new_block()     ← must compute
        break                    ← rest of prompt is also a miss
```

GPU→CPU eviction: when GPU blocks are full, evict LRU blocks by DMA-copying to pinned CPU memory. When a sequence needs them back (e.g., preemption), swap back. This is all async via CUDA streams.

---

### 4.2 Scheduler

The scheduler runs a tight loop on a dedicated tokio task. Each iteration is one "step":

```rust
loop {
    // 1. Accept new requests from HTTP layer via mpsc channel
    drain_new_requests(&mut waiting_queue);

    // 2. Classify each active sequence as prefill or decode
    let (prefill_seqs, decode_seqs) = classify_sequences();

    // 3. Chunked prefill: only process max_prefill_tokens this step
    //    This lets decode sequences run in the same step
    let prefill_batch = take_prefill_chunk(&mut prefill_seqs, MAX_PREFILL_TOKENS);

    // 4. Allocate KV blocks for new tokens (with prefix cache hits)
    allocate_blocks(&prefill_batch, &decode_seqs)?;

    // 5. Build the batch descriptor and send to executor
    let batch = build_batch(prefill_batch, decode_seqs);
    executor.run(batch).await;

    // 6. Process outputs: update sequence states, stream tokens to HTTP
    process_outputs(outputs);

    // 7. Free blocks for finished sequences
    free_finished_sequences();
}
```

**Key scheduler parameters** (tunable):
- `MAX_PREFILL_TOKENS`: 2048 — caps TTFT spike from long prefills
- `MAX_BATCH_TOKENS`: 8192 — total tokens in flight per step  
- `MAX_SEQS`: 256 — max concurrent sequences
- `SWAP_SPACE_GB`: 8 — CPU DRAM for preempted sequences

---

### 4.3 TP Workers

One Rust thread per GPU. Each thread:
1. Owns a CUDA context (set via `cuCtxSetCurrent`)
2. Holds its shard of model weights (loaded at startup)
3. Receives batch descriptors from the executor via a lock-free channel
4. Runs the forward pass
5. AllReduces activations with other GPUs via NCCL after each Attention and FFN layer

```rust
struct TpWorker {
    rank: usize,
    gpu_id: usize,
    nccl_comm: NcclComm,
    weights: ModelWeights,   // sharded for this rank
    stream: CudaStream,
}

impl TpWorker {
    fn forward(&self, batch: &Batch) -> Tensor {
        let mut x = self.embed(batch.token_ids);

        for layer in &self.weights.layers {
            // Attention (each rank handles head_dim/tp_size heads)
            let attn_out = self.attention(&x, batch, layer);
            let attn_out = self.nccl_comm.all_reduce(attn_out, Sum);  // NCCL

            // FFN (each rank handles ffn_dim/tp_size intermediate)
            let ffn_out = self.ffn(&attn_out, layer);
            let ffn_out = self.nccl_comm.all_reduce(ffn_out, Sum);    // NCCL

            x = attn_out + ffn_out;  // residual
        }

        // Only rank 0 returns logits
        if self.rank == 0 { self.lm_head(&x) } else { Tensor::empty() }
    }
}
```

For **2 nodes**: initialize NCCL with `ncclCommInitRank` using a shared `ncclUniqueId` distributed via a TCP rendezvous (rank 0 broadcasts it). After that, NCCL handles all cross-node communication transparently.

---

### 4.4 Model Architecture

Support the most common agentic models first. Each model is a thin wrapper that maps layer names to the kernel calls.

```
models/
  mod.rs        ← Model trait
  llama.rs      ← Llama 3.x, Qwen 2.5, Mistral, DeepSeek-V3
  gemma.rs      ← Gemma 2/3 (different attention variant)
```

The `Model` trait is simple:

```rust
trait Model: Send + Sync {
    fn load(path: &Path, tp_rank: usize, tp_size: usize) -> Self;
    fn forward(&self, batch: &Batch, kv_cache: &KvCache) -> Tensor; // logits
    fn config(&self) -> &ModelConfig;
}
```

Weight loading uses `safetensors` crate. For TP sharding, split `q_proj`, `k_proj`, `v_proj`, `o_proj` along the head dimension, and `gate_proj`, `up_proj`, `down_proj` along the intermediate dimension. This is done at load time.

---

### 4.5 CUDA Graph Capture

For the decode step (where batch sizes are small and stable), capture CUDA graphs to eliminate kernel launch overhead:

```rust
struct GraphCache {
    // Capture a graph for each common decode batch size
    graphs: HashMap<usize, CudaGraph>,  // batch_size → graph
}
```

Capture graphs for batch sizes `[1, 2, 4, 8, 16, 32, 64, 128]` on first warm-up. Decode steps then execute via `cudaGraphLaunch` instead of individual kernel dispatches. This is exactly what nano-vLLM does and it gives a measurable latency reduction for single-request agent workloads.

---

### 4.6 HTTP API

Use `axum` for async HTTP. Implement only what agent frameworks actually use:

```
POST /v1/chat/completions   ← main endpoint, streaming + non-streaming
POST /v1/completions        ← raw completions
GET  /v1/models             ← list loaded models
GET  /health                ← liveness check
GET  /metrics               ← Prometheus: throughput, cache hit rate, queue depth
```

Streaming uses `text/event-stream` (SSE). Each token is pushed to the HTTP response as soon as it's sampled, without waiting for the full sequence.

---

## 5. Key Rust Crates

| Crate | Purpose |
|---|---|
| `axum` | HTTP server, SSE streaming |
| `tokio` | Async runtime |
| `safetensors` | Load model weights (HuggingFace format) |
| `tokenizers` | Tokenization (HuggingFace, pure Rust) |
| `cudarc` | Safe Rust CUDA memory management, stream/event API |
| `cublas-sys` | cuBLAS FFI bindings |
| `nccl-sys` | NCCL FFI bindings (or generate via bindgen) |
| `half` | BF16/FP16 types |
| `sha2` | SHA256 for KV block prefix hashing |
| `dashmap` | Concurrent hashmap for prefix cache |
| `serde` / `serde_json` | Config and API serialization |
| `tracing` | Structured logging |
| `prometheus` | Metrics export |
| `bindgen` (build dep) | Generate FFI bindings from `llm_kernels.h` |
| `cmake` (build dep) | Drive CUDA kernel compilation |

**What to avoid:**
- `tch` / `torch` bindings — brings in all of PyTorch
- `candle` — useful but duplicates work if you're already calling CUDA kernels directly
- `burn` — too early stage for production GPU kernels

---

## 6. Implementation Phases

### Phase 1 — Working Single-GPU Server (2–3 weeks)

**Goal:** End-to-end inference on one GPU, no batching yet.

- [ ] `kernels/` build system: cmake + bindgen, compile FlashAttention + vLLM kernels
- [ ] `cudarc`-based tensor type: `CudaTensor { ptr, shape, dtype, device }`  
- [ ] Weight loader: read safetensors → shard for TP=1 → allocate CUDA memory
- [ ] Llama model forward pass using the kernel wrappers
- [ ] Greedy sampler (argmax)
- [ ] CLI inference test: load model, tokenize input, run forward, decode output
- [ ] Basic HTTP server (axum): single request, no streaming

**Checkpoint:** Can serve a Llama-3.1-8B request end-to-end.

---

### Phase 2 — KV Cache + Prefix Caching (1–2 weeks)

**Goal:** Efficient memory management for multi-turn conversations.

- [ ] `BlockPool` with GPU allocation, block table management
- [ ] `KVBlockManager` with SHA256 prefix hash table
- [ ] LRU eviction (no CPU swap yet)
- [ ] PagedAttention kernel wired in (replaces naive attention)
- [ ] Multi-turn benchmark: verify cache hit rate on agent-style workloads

**Checkpoint:** Second turn of a conversation with the same system prompt shows >90% cache hit rate.

---

### Phase 3 — Continuous Batching + Chunked Prefill (1–2 weeks)

**Goal:** Serve multiple concurrent users without head-of-line blocking.

- [ ] Scheduler main loop: waiting queue → prefill slots → decode slots
- [ ] Chunked prefill: `max_prefill_tokens` cap per step
- [ ] Batch builder: packs variable-length sequences into padded/ragged tensors
- [ ] SSE streaming: push tokens to HTTP response per decode step
- [ ] Preemption: swap-out a decode sequence to CPU when GPU memory is full
- [ ] Load test: 50 concurrent agent conversations, measure throughput/TTFT

**Checkpoint:** Stable throughput under concurrent load; TTFT is bounded.

---

### Phase 4 — Tensor Parallelism: 8 GPUs (1 week)

**Goal:** Scale to full single-node.

- [ ] NCCL init: `ncclCommInitAll` for all GPUs on the node
- [ ] Weight sharding at load time for TP=8
- [ ] Worker thread per GPU with CUDA context isolation
- [ ] AllReduce after attention and FFN projections
- [ ] CUDA graph capture for decode (batch sizes 1–128)
- [ ] Benchmark: compare TP=1 vs TP=8 throughput on 70B model

**Checkpoint:** Llama-3.1-70B running at expected throughput on 8×H100.

---

### Phase 5 — Two-Node Support (1 week)

**Goal:** 16 GPUs across 2 machines.

- [ ] Process launcher: SSH-based or MPI, starts the binary on both nodes
- [ ] NCCL `ncclCommInitRank` with shared `ncclUniqueId` via TCP rendezvous
- [ ] Scheduler runs only on rank 0; workers on both nodes receive broadcast batches
- [ ] Test with TP=16 on a 2×8-GPU model (e.g., DeepSeek-V3)

**Checkpoint:** DeepSeek-V3 (671B MoE) or a 405B dense model running across 2 nodes.

---

### Phase 6 — Polish & Observability (ongoing)

- [ ] CPU swap-in/swap-out for KV cache (pinned host memory, async CUDA memcpy)
- [ ] W8A8 quantization path (INT8 GEMMs via vLLM's CUTLASS W8A8 kernels)
- [ ] Prometheus metrics: `kv_cache_hit_rate`, `queue_depth`, `tokens_per_second`, `ttft_ms`
- [ ] Config file (TOML): model path, TP size, block pool size, chunked prefill params
- [ ] OpenTelemetry tracing for request lifecycle

---

## 7. Directory Layout

```
llm-server/
├── Cargo.toml
├── build.rs                    ← drives cmake, runs bindgen
│
├── kernels/                    ← C/CUDA kernel library
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── llm_kernels.h       ← C API (consumed by bindgen)
│   └── src/
│       ├── attention.cu        ← wraps FlashAttention-3 + PagedAttn
│       ├── layernorm.cu        ← wraps vLLM RMSNorm/LayerNorm
│       ├── pos_encoding.cu     ← wraps vLLM RoPE
│       └── sampling.cu         ← wraps vLLM top-p / top-k
│
└── src/
    ├── main.rs                 ← startup, arg parsing, model load
    │
    ├── api/
    │   ├── mod.rs
    │   ├── routes.rs           ← axum route handlers
    │   └── types.rs            ← OpenAI API request/response types
    │
    ├── scheduler/
    │   ├── mod.rs              ← main scheduling loop
    │   ├── batch.rs            ← batch builder (ragged tensor packing)
    │   └── queue.rs            ← waiting queue, sequence state machine
    │
    ├── kv_cache/
    │   ├── mod.rs
    │   ├── block_pool.rs       ← GPU + CPU block allocation
    │   ├── prefix_cache.rs     ← hash table, LRU eviction
    │   └── swap.rs             ← async GPU↔CPU swapping
    │
    ├── executor/
    │   ├── mod.rs              ← dispatch batches to TP workers
    │   ├── worker.rs           ← per-GPU thread, CUDA context
    │   ├── nccl.rs             ← NCCL comm wrappers
    │   └── graph_cache.rs      ← CUDA graph capture/replay
    │
    ├── models/
    │   ├── mod.rs              ← Model trait
    │   ├── llama.rs            ← Llama / Qwen / Mistral / DeepSeek
    │   ├── gemma.rs
    │   └── weights.rs          ← safetensors loading + TP sharding
    │
    ├── kernels/
    │   ├── mod.rs              ← generated FFI bindings (from build.rs)
    │   ├── attention.rs        ← safe Rust wrappers
    │   ├── layernorm.rs
    │   └── sampling.rs
    │
    └── cuda/
        ├── mod.rs
        ├── tensor.rs           ← CudaTensor type (wraps cudarc DevicePtr)
        └── stream.rs           ← CUDA stream management
```

---

## 8. Performance Targets

For **Llama-3.1-70B on 8×H100 (80GB)**, the targets to validate against vLLM:

| Metric | Target | Measurement |
|---|---|---|
| Decode throughput | > 3,000 tokens/sec | 50 concurrent decode sequences |
| TTFT (512-token prompt, cold) | < 100ms | Single request |
| TTFT (512-token prompt, cached) | < 20ms | Second turn, 90%+ cache hit |
| Prefix cache hit rate | > 85% | Agent workload (repeated system prompt) |
| GPU utilization | > 75% | Under mixed prefill+decode load |
| KV cache memory efficiency | > 95% | vs theoretical max (PagedAttention baseline) |

---

## Key Design Decisions — Rationale

**"Why not just wrap Python vLLM?"**  
Python's async GIL and per-step scheduling overhead adds 5–15ms of CPU latency per step at high concurrency. Rust eliminates this: the scheduler, block manager, and batch builder all run in microseconds.

**"Why not mistral.rs as a base?"**  
mistral.rs is the closest existing codebase. It's worth studying heavily, and you can borrow its CUDA memory management patterns. However, it doesn't implement PagedAttention or continuous batching — those are what you're building. Starting from scratch (with cudarc) gives cleaner abstractions.

**"Why reuse vLLM kernels specifically?"**  
vLLM's CUDA kernels are among the most heavily optimized and battle-tested open-source LLM kernels available. They handle all the edge cases (variable sequence lengths, chunked prefill, multi-query attention). Using them means you inherit years of correctness work for free.

**"Block size of 16 tokens — is that right?"**  
Yes. The tradeoff is compute efficiency (larger = fewer kernel launches) vs. cache granularity (smaller = finer prefix reuse). 16 is the same default used by vLLM and nano-vLLM for the same reason. You can expose it as a config parameter but 16 is the right default for agent workloads.

---

*This plan is designed to be executable: each phase has a clear checkpoint, the kernel reuse strategy avoids all CUDA authoring, and the Rust layer is focused exclusively on the scheduling, memory management, and orchestration logic where it provides the most value.*
