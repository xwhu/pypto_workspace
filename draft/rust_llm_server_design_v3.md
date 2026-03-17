# Rust LLM Inference Server — Design & Implementation Plan (v3)

> **Target:** 1–2 machines, 8–16 GPU/NPU cards, high-performance inference for agent/coding workloads
> **Philosophy:** Rust for orchestration and scheduling. Reuse CUDA/NPU kernels via FFI.
> **New in v3:** Hardware abstraction layer (CUDA + Ascend NPU), overlap scheduling, radix-tree prefix cache, token-budget scheduling, disaggregated prefill/decode, expert parallelism for MoE models, and fine-grained TP control.

---

## Table of Contents

1. [What's New in v3](#1-whats-new-in-v3)
2. [Configuration Space](#2-configuration-space)
3. [Generator Design](#3-generator-design)
4. [Architecture Overview](#4-architecture-overview)
5. [Hardware Abstraction Layer](#5-hardware-abstraction-layer)
6. [Scheduler Design](#6-scheduler-design)
7. [KV Cache: Radix-Tree Prefix Cache](#7-kv-cache-radix-tree-prefix-cache)
8. [Pipeline Parallelism Design](#8-pipeline-parallelism-design)
9. [Expert Parallelism (MoE)](#9-expert-parallelism-moe)
10. [Disaggregated Serving](#10-disaggregated-serving)
11. [Quantization Schemes](#11-quantization-schemes)
12. [Kernel Reuse Map](#12-kernel-reuse-map)
13. [Component Design](#13-component-design)
14. [Key Rust Crates](#14-key-rust-crates)
15. [Implementation Phases](#15-implementation-phases)
16. [Directory Layout](#16-directory-layout)
17. [Performance Targets](#17-performance-targets)

---

## 1. What's New in v3

### From Reference Project Analysis

Five reference projects were analyzed: **LMDeploy**, **Mini-SGLang**, **Mistral.rs**, **Nano-vLLM**, and **Nano-vLLM-v1**. Key insights extracted:

| Source | Insight Adopted |
|---|---|
| Mini-SGLang | **Overlap scheduling**: process last batch's results while GPU computes next |
| Mini-SGLang | **Radix-tree prefix cache**: incremental matching replaces flat hash table |
| Nano-vLLM-v1 | **Token-budget scheduling**: explicit per-iteration budget prevents OOM |
| Nano-vLLM-v1 | **Chunked prefill with layout tracking**: `computed | new_computed | new` |
| LMDeploy | **Disaggregated serving**: separate prefill/decode processes for head-of-line unblocking |
| LMDeploy | **Fine-grained TP**: separate TP degrees for attention vs. MLP |
| LMDeploy | **Ascend NPU backend**: hardware abstraction via device traits |
| LMDeploy | **Expert parallelism**: MoE expert sharding across ranks |
| Mistral.rs | **Multiple KV cache strategies**: full / rotating / hybrid per layer |
| Mistral.rs | **Ring-topology TP**: reduces AllReduce bandwidth |
| Mistral.rs | **Bucketing scheduler**: groups sequences by length to improve batch homogeneity |

### Summary of v3 Additions

1. **Hardware Abstraction Layer (HAL):** Trait-based device abstraction enabling CUDA and Ascend NPU (via CANN) with a single codebase, no `#[cfg]` sprawl.
2. **Overlap Scheduler:** Two async loops — GPU runs forward pass while CPU prepares next batch.
3. **Radix-Tree Prefix Cache:** Replaces the flat `DashMap<u64, u32>` with a proper trie structure supporting incremental prefix matching.
4. **Token-Budget Scheduling with Layout Tracking:** Explicit iteration budget; tracks `computed | new_computed | new` blocks.
5. **Disaggregated Prefill/Decode (optional):** Two processes, same binary, different roles; connected via shared-memory KV transfer.
6. **Expert Parallelism (EP):** Generator emits EP-aware MoE expert distribution when `ep > 1`.
7. **Fine-Grained TP:** Optional `attn_tp` / `mlp_tp` config for models where attention and MLP have different compute profiles.

---

## 2. Configuration Space

### Parallelism

| Mode | TP | PP | EP | Use Case |
|---|---|---|---|---|
| `tp_only` | 1–8 | 1 | 1 | Single node, <70B dense |
| `tp_pp` | 1–8 | 2–4 | 1 | Two nodes, 70B–405B dense |
| `tp_ep` | 1–8 | 1 | 1–8 | MoE (DeepSeek, Mixtral) single node |
| `tp_pp_ep` | 1–8 | 2–4 | 1–8 | MoE across two nodes |
| `dp_tp` | N replicas | 1 | 1 | High throughput, smaller models |
| `disagg` | any | any | any | Disaggregated prefill/decode |

### Quantization

| Scheme | Precision | Memory | Speed | Best For |
|---|---|---|---|---|
| `bf16` | BF16 | 1× | Baseline | Accuracy-sensitive |
| `fp8` | FP8 (E4M3) | ~2× | ~1.5× | H100 / Ascend 910B native |
| `w8a8` | INT8 | ~2× | ~1.3× | A100 / Ascend 910 |
| `w4a16` | INT4 | ~4× | ~1.2× | Memory-constrained |

### Hardware Target

| Target | Notes |
|---|---|
| `cuda` | NVIDIA GPUs (H100, A100, A10) |
| `ascend` | Huawei Ascend NPU via CANN (910B, 910C) |
| `cpu` | Debug / tokenizer-only mode |

### Example Configs

```yaml
# config-h100-coding.yaml — 8×H100, FP8, coding assistant
hardware: cuda
parallelism: { tp: 8, pp: 1, ep: 1 }
quant: fp8
model_family: llama
max_seq_len: 32768
max_batch_tokens: 16384
max_prefill_tokens: 4096
block_size: 16
kv_cache_gb: 60
cpu_swap_gb: 32
scheduler:
  overlap: true
  chunked_prefill: true
  max_decode_seqs: 256
```

```yaml
# config-ascend-agent.yaml — 8×Ascend 910B, W8A8, agent workload
hardware: ascend
parallelism: { tp: 8, pp: 1, ep: 1 }
quant: w8a8
model_family: qwen
max_seq_len: 131072
max_batch_tokens: 32768
max_prefill_tokens: 8192
block_size: 16
kv_cache_gb: 55
cpu_swap_gb: 32
scheduler:
  overlap: true
  chunked_prefill: true
  max_decode_seqs: 512
hccl:
  master_addr: 10.0.0.1
  master_port: 29500
```

```yaml
# config-deepseek-moe.yaml — 16×H100, FP8, DeepSeek-V3 671B
hardware: cuda
parallelism: { tp: 8, pp: 2, ep: 8 }
quant: fp8
model_family: deepseek
max_seq_len: 131072
max_batch_tokens: 32768
max_prefill_tokens: 8192
block_size: 16
kv_cache_gb: 55
cpu_swap_gb: 64
scheduler:
  overlap: true
  chunked_prefill: true
  disagg: false
nccl:
  rdma_device: mlx5_0
  master_addr: 10.0.0.1
  master_port: 29500
```

---

## 3. Generator Design

The generator is unchanged in concept but expanded to emit hardware-specific code.

### Generated Files

```
generated/
  Cargo.toml             ← features = ["cuda"] or ["ascend"], quant features
  build.rs               ← compiles kernel variants for the target hardware
  src/
    config.rs            ← const TP_SIZE, PP_SIZE, EP_SIZE, QUANT, HARDWARE, etc.
    main.rs              ← wired to the right scheduler + executor + HAL
  kernels/
    CMakeLists.txt        ← CUDA build  (hardware=cuda)
    CMakeLists_cann.txt   ← CANN build  (hardware=ascend)
```

### Generated `config.rs` (example for Ascend, TP=8, W8A8)

```rust
// AUTO-GENERATED — do not edit manually
// Source: config-ascend-agent.yaml

pub const TP_SIZE: usize = 8;
pub const PP_SIZE: usize = 1;
pub const EP_SIZE: usize = 1;
pub const WORLD_SIZE: usize = TP_SIZE * PP_SIZE;

pub const BLOCK_SIZE: usize = 16;
pub const MAX_SEQ_LEN: usize = 131072;
pub const MAX_BATCH_TOKENS: usize = 32768;
pub const MAX_PREFILL_TOKENS: usize = 8192;
pub const MAX_DECODE_SEQS: usize = 512;

pub const OVERLAP_SCHEDULING: bool = true;
pub const CHUNKED_PREFILL: bool = true;
pub const DISAGG_MODE: bool = false;

pub type WeightDtype = i8;
pub type ActivDtype = i8;
pub type KvDtype = half::bf16;

pub const HARDWARE: Hardware = Hardware::Ascend;
pub const COMM_BACKEND: CommBackend = CommBackend::Hccl;  // vs Nccl for CUDA

pub const HCCL_MASTER_ADDR: &str = "10.0.0.1";
pub const HCCL_MASTER_PORT: u16 = 29500;
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
│               Overlap Scheduler (Rust + Tokio)                      │
│  Token-budget · Chunked prefill · Bucketing · Preemption            │
│  CPU Loop A: prepare next batch ←→ GPU Loop B: forward + sample     │
└──────────────┬────────────────────────────┬─────────────────────────┘
               │                            │
┌──────────────▼──────────┐   ┌─────────────▼───────────────────────┐
│  Radix-Tree KV Manager  │   │     Pipeline/Simple Executor         │
│  Trie prefix cache      │   │  PP: splits into micro-batches        │
│  Block pool (GPU+CPU)   │   │  EP: routes tokens to expert ranks   │
│  LRU eviction           │   │  HAL: device-agnostic kernel calls   │
└─────────────────────────┘   └──────────────┬──────────────────────┘
                                             │ one thread per device
                              ┌──────────────▼──────────────────────┐
                              │   TP Workers [tp_rank 0..TP_SIZE)   │
                              │   Each device: Attention → FFN/MoE  │
                              │   Comm AllReduce for TP             │
                              └──────────────┬──────────────────────┘
                                             │ HAL FFI
┌────────────────────────────────────────────▼──────────────────────┐
│              Hardware Abstraction Layer (HAL)                       │
│  CUDA path:   FlashAttn-3  PagedAttn  cuBLAS  FP8  NCCL           │
│  Ascend path: FlashAttn-Ascend  PagedAttn  CANN  HCCL             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 5. Hardware Abstraction Layer

### Design Goal

One scheduler, one executor, one KV manager — all hardware-agnostic. The HAL is the only place that knows whether we're on CUDA or Ascend.

### HAL Traits

```rust
/// Device memory handle — thin wrapper over CUdeviceptr or Ascend ACL buffer
pub trait DeviceBuffer: Send + Sync {
    fn as_ptr(&self) -> *const u8;
    fn as_mut_ptr(&mut self) -> *mut u8;
    fn len_bytes(&self) -> usize;
}

/// Device allocator: allocates/frees device memory
pub trait DeviceAllocator: Send + Sync {
    fn alloc(&self, bytes: usize) -> Box<dyn DeviceBuffer>;
    fn free(&self, buf: Box<dyn DeviceBuffer>);
    fn device_id(&self) -> usize;
}

/// Collective communication backend (NCCL or HCCL)
pub trait CommBackend: Send + Sync {
    fn all_reduce_sum(&self, buf: &mut dyn DeviceBuffer, stream: &dyn Stream);
    fn send(&self, buf: &dyn DeviceBuffer, dest: usize, stream: &dyn Stream);
    fn recv(&self, buf: &mut dyn DeviceBuffer, src: usize, stream: &dyn Stream);
}

/// Kernel dispatch: hardware-specific kernel calls
pub trait KernelDispatch: Send + Sync {
    fn paged_attention(
        &self,
        q: &dyn DeviceBuffer,
        k_cache: &dyn DeviceBuffer,
        v_cache: &dyn DeviceBuffer,
        block_tables: &dyn DeviceBuffer,
        out: &mut dyn DeviceBuffer,
        params: &AttentionParams,
        stream: &dyn Stream,
    );

    fn gemm(
        &self,
        a: &dyn DeviceBuffer,
        b: &dyn DeviceBuffer,
        out: &mut dyn DeviceBuffer,
        params: &GemmParams,
        stream: &dyn Stream,
    );

    fn rms_norm(
        &self,
        input: &dyn DeviceBuffer,
        weight: &dyn DeviceBuffer,
        out: &mut dyn DeviceBuffer,
        eps: f32,
        stream: &dyn Stream,
    );

    fn rope(
        &self,
        q: &mut dyn DeviceBuffer,
        k: &mut dyn DeviceBuffer,
        positions: &dyn DeviceBuffer,
        params: &RopeParams,
        stream: &dyn Stream,
    );

    fn sampling_top_p_top_k(
        &self,
        logits: &dyn DeviceBuffer,
        out: &mut dyn DeviceBuffer,
        params: &SamplingParams,
        stream: &dyn Stream,
    );
}

/// Stream: CUDA stream or Ascend ACL stream
pub trait Stream: Send + Sync {
    fn synchronize(&self);
}
```

### CUDA Implementation

```rust
pub struct CudaKernels {
    device: i32,
}

impl KernelDispatch for CudaKernels {
    fn paged_attention(&self, ...) {
        unsafe { ffi::cuda::paged_attention_v2(...) }
    }
    fn gemm(&self, ...) {
        match WeightDtype::KIND {  // compile-time const from config.rs
            DtypeKind::BF16  => unsafe { ffi::cuda::cublas_gemm_bf16(...) },
            DtypeKind::FP8   => unsafe { ffi::cuda::cutlass_fp8_gemm(...) },
            DtypeKind::INT8  => unsafe { ffi::cuda::cutlass_w8a8_gemm(...) },
            DtypeKind::INT4  => unsafe { ffi::cuda::awq_gemm(...) },
        }
    }
    // ...
}
```

### Ascend NPU Implementation

```rust
pub struct AscendKernels {
    device: i32,
}

impl KernelDispatch for AscendKernels {
    fn paged_attention(&self, ...) {
        // Ascend's CANN IncreFlashAttention kernel
        unsafe { ffi::ascend::incre_flash_attention(...) }
    }
    fn gemm(&self, ...) {
        match WeightDtype::KIND {
            DtypeKind::BF16 => unsafe { ffi::ascend::aclnn_matmul_bf16(...) },
            DtypeKind::INT8 => unsafe { ffi::ascend::aclnn_quant_matmul_w8a8(...) },
            // FP8 available on Ascend 910B+ (E4M3 via aclnnCastToFp8)
            DtypeKind::FP8  => unsafe { ffi::ascend::aclnn_matmul_fp8(...) },
            _ => panic!("unsupported quant for Ascend"),
        }
    }
    // ...
}
```

### HAL Construction (compile-time dispatch)

The generator emits in `main.rs`:

```rust
// Generated for hardware = ascend
let kernels: Arc<dyn KernelDispatch> = Arc::new(AscendKernels::new(device_id));
let comm: Arc<dyn CommBackend> = Arc::new(HcclComm::init(tp_rank, tp_size));
```

Or for CUDA:

```rust
// Generated for hardware = cuda
let kernels: Arc<dyn KernelDispatch> = Arc::new(CudaKernels::new(device_id));
let comm: Arc<dyn CommBackend> = Arc::new(NcclComm::init(tp_rank, tp_size));
```

---

## 6. Scheduler Design

### Overlap Scheduling (from Mini-SGLang)

The key insight: on every step, the CPU can prepare the *next* batch while the GPU executes the *current* batch. Two tasks run concurrently — only synchronize at batch boundaries.

```
Time →   T1              T2              T3
GPU:     forward(B0)     forward(B1)     forward(B2)
CPU:     prepare(B1)     prepare(B2)     prepare(B3)
         sample(B0−1)    sample(B1−1)    sample(B2−1)
         ─────────────────────────────────────────────
         [sync]          [sync]          [sync]
```

```rust
/// Overlap scheduler state machine
pub struct OverlapScheduler {
    // Two alternating slots: current and next
    current_batch: Option<ForwardInput>,
    next_batch: Option<ForwardInput>,

    // Queues
    waiting: VecDeque<Request>,
    running: Vec<Sequence>,
    swapped: Vec<Sequence>,

    // Budget
    token_budget: usize,   // max tokens per iteration = MAX_BATCH_TOKENS
    max_decode_seqs: usize,

    kv_manager: Arc<RadixKvManager>,
}

impl OverlapScheduler {
    /// Called from the CPU loop — prepares next_batch from queues
    async fn schedule(&mut self) -> ForwardInput {
        // Step 1: decode pass — running sequences get 1 token each
        let decode_seqs = self.schedule_decode();
        let decode_tokens: usize = decode_seqs.len();

        // Step 2: prefill pass — fill remaining budget with chunked prefill
        let remaining = self.token_budget.saturating_sub(decode_tokens);
        let prefill_chunk = self.schedule_prefill_chunk(remaining);

        ForwardInput {
            decode_seqs,
            prefill_chunk,
        }
    }

    /// Overlap loop — CPU and GPU tasks interleaved via channels
    pub async fn run(mut self, executor: Arc<dyn Executor>) {
        let (batch_tx, batch_rx) = tokio::sync::mpsc::channel(1);
        let (result_tx, result_rx) = tokio::sync::mpsc::channel(1);

        // GPU task: forward pass
        let exec = executor.clone();
        tokio::spawn(async move {
            while let Some(batch) = batch_rx.recv().await {
                let result = exec.forward(batch).await;
                result_tx.send(result).await.unwrap();
            }
        });

        // CPU task: scheduling and result processing
        loop {
            let next_batch = self.schedule().await;
            // Process previous step's results while GPU runs
            if let Some(result) = result_rx.try_recv().ok() {
                self.process_results(result).await;
            }
            batch_tx.send(next_batch).await.unwrap();
        }
    }
}
```

### Token-Budget Scheduling with Block Layout Tracking (from Nano-vLLM-v1)

```
Token layout per sequence per iteration:

|    computed    | new_computed |     new     |
|← cached hits →|← this step →|← future  →|
                  ↑             ↑
              tokens to      tokens to
              compute now     allocate
```

```rust
struct TokenLayout {
    computed_tokens: usize,      // already in KV cache (prefix hit)
    new_computed_tokens: usize,  // tokens we will compute this step
    new_tokens_remaining: usize, // tokens not yet computed (chunked prefill)
}

impl BlockManager {
    /// Predict token layout BEFORE allocating blocks
    fn get_token_layout(&self, seq: &Sequence) -> TokenLayout {
        let (cached, _) = self.radix_cache.find_prefix(&seq.tokens);
        let budget_this_step = seq.budget_this_step; // set by scheduler
        TokenLayout {
            computed_tokens: cached,
            new_computed_tokens: budget_this_step,
            new_tokens_remaining: seq.prompt_len.saturating_sub(cached + budget_this_step),
        }
    }
}
```

### Bucketing (from Mistral.rs)

Optionally group sequences by length bucket to improve batch homogeneity (reduces padding waste):

```rust
const DECODE_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

fn bucket_for(seq_len: usize) -> usize {
    DECODE_BUCKETS.partition_point(|&b| b < seq_len)
        .min(DECODE_BUCKETS.len() - 1)
}
```

On Ascend NPU, bucketing is especially valuable because the NPU's DMA controller is optimized for fixed-size tensors. Variable-length batches require padding to the max length in the batch, but bucketing tightly bounds that overhead.

---

## 7. KV Cache: Radix-Tree Prefix Cache

### Why Radix Tree over Flat Hash Map (from Mini-SGLang)

A flat `HashMap<u64, BlockId>` (as in v2) hashes each full block of tokens independently. This means:
- A system prompt of 1,024 tokens (64 blocks of 16) requires 64 hash lookups
- If two requests share the first 800 tokens but differ at token 801, they share 50 blocks — but the hash table doesn't know that without scanning all blocks

A radix tree stores the *token sequence* as a path through the tree. A single prefix match traversal finds all shared blocks in O(shared_prefix_len / BLOCK_SIZE) steps, naturally handles partial matches, and enables exact split-and-reuse.

### Data Structure

```rust
/// One node per block boundary in the radix tree
struct RadixNode {
    /// Token content of this block (BLOCK_SIZE tokens)
    tokens: [u32; BLOCK_SIZE],
    /// Physical block ID on the device
    block_id: u32,
    /// Children: keyed on the first token of the next block
    children: HashMap<u32, Arc<RwLock<RadixNode>>>,
    /// Reference count: 0 means evictable
    ref_count: AtomicUsize,
    /// Last access time for LRU
    last_used: AtomicU64,
}

pub struct RadixKvManager {
    root: Arc<RwLock<RadixNode>>,
    /// GPU block pool
    gpu_pool: BlockPool,
    /// CPU block pool (for swap-out)
    cpu_pool: BlockPool,
    /// LRU eviction heap (block_id → last_used)
    eviction_heap: Mutex<BinaryHeap<EvictEntry>>,
}
```

### Core Operations

```rust
impl RadixKvManager {
    /// Returns (num_cached_tokens, cached_block_ids)
    /// The returned blocks are ref-counted — caller must call release() when done
    pub fn find_prefix(&self, tokens: &[u32]) -> (usize, Vec<u32>) {
        let mut node = self.root.read();
        let mut cached_tokens = 0;
        let mut block_ids = vec![];

        for chunk in tokens.chunks(BLOCK_SIZE) {
            if chunk.len() < BLOCK_SIZE { break; }  // partial block: no hit
            let first_tok = chunk[0];
            match node.children.get(&first_tok) {
                Some(child_arc) => {
                    let child = child_arc.read();
                    if child.tokens == chunk {
                        child.ref_count.fetch_add(1, Ordering::SeqCst);
                        child.last_used.store(now_ns(), Ordering::Relaxed);
                        block_ids.push(child.block_id);
                        cached_tokens += BLOCK_SIZE;
                        node = child;
                    } else {
                        break;  // token mismatch — prefix ends here
                    }
                }
                None => break,
            }
        }
        (cached_tokens, block_ids)
    }

    /// After a sequence finishes, insert newly-computed blocks into the tree
    pub fn insert_prefix(&self, tokens: &[u32], block_ids: &[u32]) {
        // Walk the tree, creating nodes for blocks not yet cached
        // ...
    }

    /// Evict blocks with ref_count == 0 using LRU
    pub fn evict_to_free(&self, target_blocks: usize) -> usize {
        // Pop from eviction_heap until target_blocks freed
        // ...
    }
}
```

### Block Pool

```rust
struct BlockPool {
    /// Device memory: shape [num_blocks, 2, num_kv_heads, BLOCK_SIZE, head_dim]
    memory: Box<dyn DeviceBuffer>,
    free: VecDeque<u32>,
    total: usize,
}
```

---

## 8. Pipeline Parallelism Design

*(Unchanged from v2 — see v2 for full PP design. Summary below.)*

PP splits layers across PP_SIZE ranks. Each rank holds `num_layers / PP_SIZE` consecutive layers. Micro-batches flow through the pipeline:

```
Node 0 (PP rank 0, TP ranks 0–7):  layers 0–39
Node 1 (PP rank 1, TP ranks 0–7):  layers 40–79

Activation shape transferred between nodes:  [batch, seq, hidden_dim]
Transport:  NCCL P2P (inter-node) or CUDA/ACL IPC (intra-node)
```

**KV cache with PP:** Each PP rank allocates only the KV blocks for its own layers. The radix tree prefix cache is computed from tokens (not layers), so prefix cache hits are identical regardless of PP configuration — all PP ranks check the same token sequence for prefix matches.

**v3 addition:** The PP executor now uses the HAL's `CommBackend` trait for P2P transfers, so the same executor code works for both NCCL (CUDA) and HCCL (Ascend).

---

## 9. Expert Parallelism (MoE)

### What EP Does

In a MoE model (DeepSeek-V3, Mixtral), each FFN layer has `E` experts, and each token routes to `top_k` of them. EP distributes experts across `EP_SIZE` ranks — each rank holds `E / EP_SIZE` experts. Tokens are dispatched to the rank that holds their selected experts.

### EP Data Flow

```
  TP AllReduce         Expert Dispatch (EP AllToAll)
       │                        │
Attn → Hidden → Router → [dispatch to EP ranks] → Expert GEMM → [gather] → Hidden
```

### Generated Config (TP=8, EP=8, PP=2 for DeepSeek-V3)

```rust
pub const EP_SIZE: usize = 8;
pub const NUM_EXPERTS: usize = 256;      // from model config
pub const EXPERTS_PER_RANK: usize = NUM_EXPERTS / EP_SIZE;  // 32
pub const TOP_K_EXPERTS: usize = 8;      // DeepSeek top-8 routing
```

### EP Executor Sketch

```rust
async fn run_moe_layer(
    &self,
    hidden: &CudaTensor,      // [batch, seq, hidden_dim]
    router_weights: &WeightTensor,
    expert_weights: &[WeightTensor],  // only EXPERTS_PER_RANK experts
    stream: &dyn Stream,
) -> CudaTensor {
    // 1. Compute routing scores (all ranks participate)
    let scores = self.kernels.router_gemm(hidden, router_weights, stream);
    let (expert_ids, gate_weights) = topk(scores, TOP_K_EXPERTS);

    // 2. AllToAll dispatch: send each token to its expert's EP rank
    let (recv_hidden, recv_expert_ids) = self.comm_ep.all_to_all_dispatch(
        hidden, expert_ids, stream,
    ).await;

    // 3. Local expert computation on this rank's experts
    let expert_out = self.run_local_experts(recv_hidden, recv_expert_ids, stream);

    // 4. AllToAll gather: return results to originating ranks
    let gathered = self.comm_ep.all_to_all_gather(
        expert_out, recv_expert_ids, stream,
    ).await;

    // 5. Weighted sum by gate weights
    self.kernels.weighted_sum(gathered, gate_weights, stream)
}
```

### EP on Ascend

On Ascend, HCCL provides AllToAll for expert dispatch. The same trait abstraction handles both backends:

```rust
pub trait CommBackend: Send + Sync {
    // ...
    fn all_to_all(
        &self,
        send: &dyn DeviceBuffer,
        recv: &mut dyn DeviceBuffer,
        send_counts: &[usize],
        stream: &dyn Stream,
    );
}
```

---

## 10. Disaggregated Serving

*(Optional — generated when `disagg: true` in config)*

### Motivation (from LMDeploy)

Prefill (prompt processing) and decode (token generation) have very different compute profiles:
- Prefill: compute-bound, large matrix multiplies, wants long context
- Decode: memory-bandwidth bound, tiny batches, wants low latency

Mixing them causes **head-of-line blocking**: a long prefill stalls decode iterations. Disaggregated serving runs two separate processes — a **Prefill Engine** and a **Decode Engine** — connected via KV transfer.

### Architecture

```
Client → HTTP Router
               │
     ┌─────────┴──────────┐
     ▼                    ▼
Prefill Engine       Decode Engine
  TP=8 GPUs           TP=8 GPUs
  Computes KV          Receives KV
  Sends KV →──────────→ Holds KV
                        Generates tokens
```

### KV Transfer

Within a single machine: **CUDA IPC** (zero-copy shared memory).
Across machines: **RDMA** via GPUDirect.

```rust
/// Prefill engine: after computing KV for a request, transfer to decode engine
async fn transfer_kv(&self, seq_id: u64, block_ids: &[u32]) {
    let kv_data = self.kv_manager.export_blocks(block_ids);
    match DISAGG_TRANSPORT {
        Transport::CudaIpc  => self.ipc_sender.send(seq_id, kv_data).await,
        Transport::Rdma     => self.rdma_sender.send(seq_id, kv_data).await,
    }
}

/// Decode engine: receive KV from prefill engine, insert into local cache
async fn receive_kv(&self, seq_id: u64) {
    let kv_data = self.ipc_receiver.recv(seq_id).await;
    self.kv_manager.import_blocks(seq_id, kv_data);
}
```

### Generator: Disagg Mode

When `disagg: true`, the generator emits `main.rs` that accepts `--role prefill|decode` and starts the appropriate engine. Both engines share the same `llm-server` library — just different entry points.

---

## 11. Quantization Schemes

*(Identical to v2 for CUDA. Ascend additions below.)*

### Ascend-Specific Quantization Notes

| Scheme | Ascend Support | Kernel |
|---|---|---|
| `bf16` | ✓ (910B+) | `aclnnMatMul` with BF16 dtype |
| `w8a8` | ✓ (910 and up) | `aclnnQuantMatMul` — weight and activation INT8 |
| `fp8` | ✓ (910B+, E4M3) | `aclnnCastToFp8` + `aclnnMatMulFp8` |
| `w4a16` | Partial | Community kernels; Ascend official roadmap item |

### Generator: Ascend quant dispatch

```rust
// config.rs (generated for hardware=ascend, quant=w8a8)
pub type WeightDtype = i8;
pub type ActivDtype  = i8;
pub type KvDtype     = half::bf16;
pub const QUANT_SCALE_MODE: QuantScaleMode = QuantScaleMode::PerChannel;
```

---

## 12. Kernel Reuse Map

| Kernel | CUDA Source | Ascend Source | Notes |
|---|---|---|---|
| Flash Attention prefill | `Dao-AILab/flash-attention` | `Ascend/AscendSpeed flash_attn` | Core attention |
| Paged Attention (decode) | `vllm-project/vllm csrc/attention/` | `aclnnIncreFlashAttention` (CANN built-in) | Variable-length KV |
| RMSNorm | `vllm-project/vllm csrc/layernorm/` | `aclnnRmsNorm` (CANN) | |
| RoPE | `vllm-project/vllm csrc/pos_encoding/` | `aclnnApplyRotaryPosEmb` (CANN) | |
| BF16/FP16 GEMM | cuBLAS `libcublas.so` | `aclnnMatMul` via CANN | |
| FP8 GEMM | `vllm csrc/quantization/fp8/` | `aclnnMatMulFp8` (CANN 8.0+) | H100 / 910B only |
| W8A8 INT8 GEMM | `vllm csrc/quantization/cutlass_w8a8/` | `aclnnQuantMatMul` (CANN) | |
| W4A16 AWQ | `vllm csrc/quantization/awq/` | Community (`AscendSpeed`) | |
| Top-p/top-k sampling | `vllm csrc/sampling_kernels.cu` | `aclnnTopK` + custom sample | |
| AllReduce (TP) | NCCL `libnccl.so` | HCCL `libhccl.so` | |
| P2P send/recv (PP) | NCCL `libnccl.so` | HCCL `libhccl.so` | |
| AllToAll (EP) | NCCL `libnccl.so` | HCCL `libhccl.so` | |
| Tokenization | `huggingface/tokenizers` Rust crate | Same | Pure Rust, device-agnostic |

---

## 13. Component Design

### 13.1 Overlap Scheduler (new)

See §6 for full design. Key interface:

```rust
pub struct OverlapScheduler { ... }

impl OverlapScheduler {
    pub async fn run(self, executor: Arc<dyn Executor>);
    async fn schedule(&mut self) -> ForwardInput;
    async fn process_results(&mut self, result: ForwardOutput);
}
```

### 13.2 Radix-Tree KV Manager (replaces BlockPool + DashMap)

See §7 for full design. Key interface:

```rust
pub struct RadixKvManager { ... }

impl RadixKvManager {
    pub fn find_prefix(&self, tokens: &[u32]) -> (usize, Vec<u32>);
    pub fn insert_prefix(&self, tokens: &[u32], block_ids: &[u32]);
    pub fn allocate_blocks(&self, n: usize) -> Vec<u32>;
    pub fn free_blocks(&self, block_ids: &[u32]);
    pub fn evict_to_free(&self, target: usize) -> usize;
    pub fn release_prefix(&self, block_ids: &[u32]);  // dec ref counts
}
```

### 13.3 Executor (updated for HAL)

```rust
pub trait Executor: Send + Sync {
    async fn forward(&self, input: ForwardInput) -> ForwardOutput;
    async fn warmup_graphs(&self);
}

// Generated variant: SimpleExecutor (PP=1) or PipelineExecutor (PP>1)
pub struct SimpleExecutor {
    workers: Vec<TpWorker>,
    kernels: Arc<dyn KernelDispatch>,
    comm_tp: Arc<dyn CommBackend>,
}
```

### 13.4 TP Worker (updated for HAL)

```rust
struct TpWorker {
    tp_rank: usize,
    layers: Vec<TransformerLayer>,
    kernels: Arc<dyn KernelDispatch>,
    comm: Arc<dyn CommBackend>,
    stream: Box<dyn Stream>,
}

impl TpWorker {
    fn forward_layer(&self, layer: &TransformerLayer, hidden: &dyn DeviceBuffer) -> Box<dyn DeviceBuffer> {
        // Attention
        let normed = self.kernels.rms_norm(hidden, &layer.attn_norm, self.stream.as_ref());
        let q = self.kernels.gemm(&normed, &layer.wq, ...);
        let k = self.kernels.gemm(&normed, &layer.wk, ...);
        let v = self.kernels.gemm(&normed, &layer.wv, ...);
        self.kernels.rope(&mut q, &mut k, &positions, ...);
        let attn_out = self.kernels.paged_attention(&q, &k_cache, &v_cache, ...);
        let proj = self.kernels.gemm(&attn_out, &layer.wo, ...);
        self.comm.all_reduce_sum(&proj, self.stream.as_ref());  // TP sync

        // FFN
        let normed = self.kernels.rms_norm(&proj, &layer.ffn_norm, ...);
        let ffn_out = self.run_ffn(&normed, &layer);
        self.comm.all_reduce_sum(&ffn_out, self.stream.as_ref());

        ffn_out
    }
}
```

### 13.5 CUDA Graph Capture (unchanged)

Decode steps are captured as CUDA graphs. On Ascend, similar functionality is available via **ACL Graph** (`aclCreateGraph` / `aclGraphCompile`). The generator emits the right capture code based on `HARDWARE`.

---

## 14. Key Rust Crates

| Crate | Purpose |
|---|---|
| `axum` | HTTP server, SSE streaming |
| `tokio` | Async runtime (overlap scheduler) |
| `safetensors` | Load model weights |
| `tokenizers` | Tokenization (HuggingFace) |
| `cudarc` | Safe Rust CUDA memory + stream management |
| `half` | BF16/FP16 types |
| `sha2` | SHA256 for prefix block hashing |
| `dashmap` | Concurrent hash maps (radix tree children) |
| `serde` / `serde_json` | Config parsing and API types |
| `tera` | Template engine for code generator |
| `clap` | CLI argument parsing |
| `tracing` | Structured logging |
| `prometheus` | Metrics |
| `bindgen` (build dep) | FFI bindings from `llm_kernels.h` and CANN headers |
| `cmake` (build dep) | Drive CUDA/CANN kernel compilation |
| `parking_lot` | Faster RwLock for radix tree nodes |

**Ascend-specific:**

| Crate / Binding | Purpose |
|---|---|
| `acl-sys` (generated by bindgen) | Raw FFI to Ascend ACL runtime |
| `hccl-sys` (generated by bindgen) | Raw FFI to HCCL collective comm |
| `cann-rs` (thin safe wrapper) | Safe Rust wrapper for ACL buffers and streams |

---

## 15. Implementation Phases

### Phase 1 — Generator + Single-Device BF16 (3 weeks)

- [ ] `llm-gen` CLI: parses config, emits `config.rs` + `Cargo.toml` + CMakeLists
- [ ] HAL traits defined; `CudaKernels` stub + `AscendKernels` stub (panic for now)
- [ ] Kernel build system: cmake + bindgen for FlashAttention + BF16 cuBLAS
- [ ] `CudaTensor` / `AscendBuffer` implementing `DeviceBuffer`
- [ ] Safetensors weight loader with TP=1 sharding
- [ ] Llama model forward pass using HAL trait calls (BF16)
- [ ] HTTP server (axum), single request, non-streaming

**Checkpoint:** End-to-end inference on Llama-3.1-8B, one GPU.

---

### Phase 2 — Radix-Tree KV Cache + Overlap Scheduler (3 weeks)

- [ ] `RadixKvManager` with `find_prefix` / `insert_prefix` / `evict_to_free`
- [ ] `BlockPool` (GPU pre-allocation)
- [ ] `OverlapScheduler` with two-task loop (batch_tx / result_rx channels)
- [ ] Token-budget scheduling with layout tracking
- [ ] Chunked prefill cap (`MAX_PREFILL_TOKENS` constant)
- [ ] PagedAttention kernel wired in
- [ ] Verify >90% cache hit rate on repeated system prompts

**Checkpoint:** Multi-turn agent conversation with measurable cache benefit; overlap scheduling reducing latency.

---

### Phase 3 — Continuous Batching + Streaming (2 weeks)

- [ ] Scheduler main loop with chunked prefill cap
- [ ] Variable-length sequence batch builder
- [ ] SSE token streaming
- [ ] Preemption + CPU swap-out
- [ ] Load test: 50 concurrent sessions

**Checkpoint:** Stable latency under concurrent load.

---

### Phase 4 — TP=8, CUDA (1 week)

- [ ] NCCL intra-node init (`ncclCommInitAll`)
- [ ] TP weight sharding at load time
- [ ] Per-GPU worker threads, CUDA context isolation
- [ ] AllReduce after attention and FFN via `CommBackend` trait
- [ ] CUDA graph capture for decode

**Checkpoint:** Llama-3.1-70B on 8×H100 at expected throughput.

---

### Phase 5 — Ascend NPU Backend (2 weeks)

- [ ] `acl-sys` bindgen from CANN headers
- [ ] `hccl-sys` bindgen
- [ ] `AscendKernels` implementing all `KernelDispatch` methods
- [ ] `HcclComm` implementing `CommBackend`
- [ ] `AscendBuffer` implementing `DeviceBuffer`
- [ ] Bucketing scheduler enabled for Ascend (fixed-size tensor optimization)
- [ ] ACL Graph capture (equivalent to CUDA graph)
- [ ] Generator: `hardware: ascend` path for CMakeLists_cann.txt

**Checkpoint:** Llama-3.1-8B serving on Ascend 910B at expected throughput.

---

### Phase 6 — FP8 Quantization (1 week)

- [ ] FP8 kernel variant for CUDA (CUTLASS) and Ascend (CANN aclnnMatMulFp8)
- [ ] Generator emits FP8 config; build system includes FP8 kernels
- [ ] Weight loader converts BF16 → FP8 or reads pre-quantized
- [ ] Per-channel scale tensors
- [ ] Benchmark: BF16 vs FP8 on both hardware

**Checkpoint:** FP8 build shows >1.3× throughput vs BF16 on both H100 and Ascend 910B.

---

### Phase 7 — Pipeline Parallelism (2 weeks)

- [ ] `PipelineExecutor` with micro-batch formation
- [ ] `CommBackend::send/recv` for inter-stage activation transfer
- [ ] SSH/MPI process launcher
- [ ] TCP rendezvous for comm init
- [ ] Generator: `pp > 1` emits pipeline executor; `pp == 1` emits simple executor

**Checkpoint:** TP=8, PP=2, FP8 serving a 671B model end-to-end.

---

### Phase 8 — Expert Parallelism for MoE (2 weeks)

- [ ] `CommBackend::all_to_all` for NCCL and HCCL
- [ ] `MoeLayer` using AllToAll dispatch + gather
- [ ] Generator: `ep > 1` emits EP-aware MoE executor
- [ ] DeepSeek model family (`model_family: deepseek`)
- [ ] Test: DeepSeek-V3 EP=8, TP=8 on single node

**Checkpoint:** MoE expert routing working at full throughput.

---

### Phase 9 — W8A8 + W4A16 + Disaggregated Serving (2 weeks)

- [ ] W8A8 kernel variant (CUTLASS for CUDA; CANN for Ascend)
- [ ] W4A16 AWQ variant (CUDA only; Ascend partial)
- [ ] Disaggregated prefill/decode: `--role prefill|decode`, KV IPC transfer
- [ ] HTTP router (minimal) for disagg mode

**Checkpoint:** W4A16 config serves a 405B model on 8 GPUs.

---

## 16. Directory Layout

```
llm-inference/
├── llm-gen/                          ← code generator
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs
│   │   ├── emit_config.rs
│   │   ├── emit_cargo.rs
│   │   ├── emit_cmake.rs             ← CUDA cmake
│   │   ├── emit_cmake_cann.rs        ← Ascend/CANN cmake (new)
│   │   └── emit_main.rs
│   └── templates/
│       ├── config.rs.tera
│       ├── Cargo.toml.tera
│       ├── CMakeLists.txt.tera
│       └── CMakeLists_cann.txt.tera  ← (new)
│
├── llm-server/                       ← server library
│   ├── Cargo.toml
│   ├── kernels/
│   │   ├── cuda/                     ← CUDA kernel wrappers
│   │   │   ├── attention.cu
│   │   │   ├── layernorm.cu
│   │   │   ├── pos_encoding.cu
│   │   │   ├── sampling.cu
│   │   │   ├── gemm_bf16.cu
│   │   │   ├── gemm_fp8.cu
│   │   │   ├── gemm_w8a8.cu
│   │   │   └── gemm_w4a16.cu
│   │   ├── ascend/                   ← Ascend CANN wrappers (new)
│   │   │   ├── attention.cpp         ← wraps aclnnIncreFlashAttention
│   │   │   ├── layernorm.cpp
│   │   │   ├── gemm_bf16.cpp
│   │   │   ├── gemm_w8a8.cpp
│   │   │   └── gemm_fp8.cpp
│   │   └── include/
│   │       ├── llm_kernels.h         ← CUDA kernel declarations
│   │       └── llm_kernels_ascend.h  ← Ascend kernel declarations (new)
│   └── src/
│       ├── hal/                      ← Hardware Abstraction Layer (new)
│       │   ├── mod.rs                ← DeviceBuffer, KernelDispatch, CommBackend traits
│       │   ├── cuda.rs               ← CudaKernels, NcclComm, CudaBuffer
│       │   └── ascend.rs             ← AscendKernels, HcclComm, AscendBuffer
│       ├── api/
│       │   ├── routes.rs
│       │   └── types.rs
│       ├── scheduler/
│       │   ├── mod.rs
│       │   ├── overlap.rs            ← OverlapScheduler (new)
│       │   ├── batch.rs              ← token budget + layout tracking (updated)
│       │   └── queue.rs
│       ├── kv_cache/
│       │   ├── radix_tree.rs         ← RadixKvManager (replaces v2 block_pool+prefix_cache)
│       │   ├── block_pool.rs
│       │   └── swap.rs
│       ├── executor/
│       │   ├── simple.rs             ← TP-only
│       │   ├── pipeline.rs           ← PP executor
│       │   ├── moe.rs                ← EP-aware MoE layer (new)
│       │   ├── worker.rs
│       │   └── comm.rs               ← CommBackend usage
│       ├── models/
│       │   ├── mod.rs
│       │   ├── llama.rs
│       │   ├── deepseek.rs           ← MoE model
│       │   ├── qwen.rs               ← (new)
│       │   └── weights.rs
│       ├── disagg/                   ← Disaggregated serving (new)
│       │   ├── prefill_engine.rs
│       │   ├── decode_engine.rs
│       │   └── kv_transfer.rs        ← IPC / RDMA KV transfer
│       └── cuda/
│           ├── tensor.rs
│           └── stream.rs
│
├── configs/
│   ├── tp8-fp8-cuda-single-node.yaml
│   ├── tp8-pp2-fp8-cuda-two-node.yaml
│   ├── tp8-w8a8-ascend-single-node.yaml   ← (new)
│   ├── tp4-w4a16-4gpu.yaml
│   ├── tp8-ep8-fp8-moe-single-node.yaml   ← (new)
│   └── tp1-bf16-dev.yaml
│
└── build/                            ← generated output (gitignored)
    └── tp8-fp8-cuda/
        ├── Cargo.toml
        ├── build.rs
        ├── src/
        │   ├── config.rs
        │   └── main.rs
        └── kernels/
            └── CMakeLists.txt
```

---

## 17. Performance Targets

### Llama-3.1-70B on 8×H100 (FP8, TP=8)

| Metric | Target |
|---|---|
| Decode throughput | > 4,500 tok/sec |
| TTFT (512-token prompt, cold) | < 80ms |
| TTFT (512-token prompt, cached) | < 15ms |
| Prefix cache hit rate (agent) | > 85% |
| Overlap scheduling latency reduction | > 15% vs non-overlap |

### DeepSeek-V3 (671B MoE) on 2×8×H100 (FP8, TP=8, PP=2, EP=8)

| Metric | Target |
|---|---|
| Decode throughput | > 1,500 tok/sec |
| TTFT | < 300ms |
| PP pipeline bubble overhead | < 5% |
| EP routing overhead | < 8% vs dense equivalent |

### Qwen-72B on 8×Ascend 910B (W8A8, TP=8)

| Metric | Target |
|---|---|
| Decode throughput | > 3,000 tok/sec |
| TTFT (512-token prompt, cold) | < 120ms |
| TTFT (512-token prompt, cached) | < 25ms |
| Memory utilization | > 80% of device HBM |

---

## Key Design Decisions — Rationale

**HAL trait objects vs compile-time dispatch:** Trait objects (`Arc<dyn KernelDispatch>`) add one pointer indirection per kernel call. For attention and GEMM, the kernel itself takes microseconds to milliseconds — one pointer dereference is immeasurable. The benefit is that the scheduler, KV manager, and executor code is truly hardware-agnostic; only `main.rs` (generated) is hardware-specific. Alternative: use `cfg` and monomorphize everything — this eliminates the indirection but spreads hardware ifdefs through every file.

**Radix tree vs flat hash table:** The flat table in v2 hashes each block independently. Radix tree traversal is O(prefix_blocks) vs O(1) per block lookup, but traversal has excellent cache locality (tree nodes are small) and enables splitting at mid-block boundaries. Real workloads (agents with system prompts + few-shot examples) have very long shared prefixes — radix tree wins in practice (Mini-SGLang benchmarks show 15–30% memory savings over flat cache for agent workloads).

**Overlap scheduling trade-off:** Overlap scheduling requires the result of step N to be processed by the CPU before the GPU finishes step N+1. If the CPU is slower than the GPU (possible for very small batches with CUDA graphs), the overlap scheduling degenerates to sequential. Add a `OVERLAP_MIN_BATCH_SIZE` threshold in the generated config — below this, fall back to synchronous scheduling.

**Ascend HCCL vs NCCL semantics:** HCCL's API is designed to mirror NCCL's API closely. The `CommBackend` trait methods map 1:1 to both. The main difference is initialization (`ncclUniqueId` vs Ascend's rendezvous URL) — handled in the generated `main.rs`.

**EP AllToAll vs expert replication:** For models with many small experts (Mixtral: 8 experts, top-2), replication can avoid AllToAll. For models with many large experts (DeepSeek: 256 experts, top-8), EP with AllToAll is necessary. The generator makes the right choice based on `num_experts` and `ep > 1` in config.

**Generator vs feature flags (same rationale as v2):** Each deployment config produces a codebase that literally only contains the relevant paths. Easier to read, debug, and reason about than a single codebase with `#[cfg(feature = "ascend")]` branches throughout.

---

*v3 adds hardware abstraction and advanced scheduling without changing the generator philosophy: each deployment is a specialized artifact. The HAL ensures Ascend NPU support doesn't complicate the scheduler or KV manager — those remain hardware-agnostic Rust.*
