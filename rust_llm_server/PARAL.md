# Parallelism Design & Implementation Progress

## Overview

This document captures the design decisions and implementation status for adding
Tensor Parallelism (TP), Pipeline Parallelism (PP), and Data Parallelism (DP) to
the Rust LLM serving project.

## Architecture

### Parallelism Model

Three orthogonal dimensions of parallelism, combined into a 3D grid:

```
world_rank = dp_rank * (tp_size * pp_size) + pp_rank * tp_size + tp_rank
```

- **TP (Tensor Parallelism)**: Splits individual layers across devices. Each device
  holds a shard of each weight matrix. Requires AllReduce after row-sharded projections.
- **PP (Pipeline Parallelism)**: Splits layers into stages across devices. Hidden states
  are sent/received between stages via point-to-point communication.
- **DP (Data Parallelism)**: Independent replicas serving different requests. No
  inter-replica communication needed for inference. Each replica gets its own HTTP port
  (`base_port + dp_rank`).

### Process Model

One OS process per NPU device. Multi-process coordination uses:
- Standard env vars: `RANK`, `WORLD_SIZE`, `LOCAL_RANK`
- HCCL (Huawei Collective Communication Library) for device-to-device communication
- File-based root info sharing for HCCL group initialization

### Launch Pattern

```bash
# Example: TP=4, PP=2 → 8 NPUs total
for rank in $(seq 0 7); do
  WORLD_SIZE=8 RANK=$rank LOCAL_RANK=$rank \
    ./rust-llm-server --tp 4 --pp 2 --weights /path/to/weights --backend ascend &
done
```

---

## Tensor Parallelism (TP) Design

### Weight Sharding Strategy (Megatron-LM column/row parallel)

| Weight | Strategy | Reason |
|--------|----------|--------|
| `q_proj`, `k_proj`, `v_proj` | `ShardColumns` | Each rank owns a subset of heads |
| `o_proj` | `ShardRows` | Partial outputs, summed via AllReduce |
| `gate_proj`, `up_proj` | `ShardColumns` | Each rank owns subset of intermediate dim |
| `down_proj` | `ShardRows` | Partial outputs, summed via AllReduce |
| `q_norm`, `k_norm` | `Replicate` | Per-head norms, small, no benefit sharding |
| `input_layernorm`, `post_attention_layernorm` | `Replicate` | Norms over full hidden size |
| `embed_tokens` | `Replicate` | Avoids AllGather after lookup |
| `lm_head` | `Replicate` | Avoids AllGather before sampling (151K vocab × FP16 = ~300MB, acceptable for TP≤8) |
| `model.norm` | `Replicate` | Final norm, small |

### AllReduce Insertion

The execution plan compiler (`plan.rs`) already inserts `AllReduceSum` after:
- `o_proj` matmul (after attention row-sharded projection)
- `down_proj` matmul (after MLP row-sharded projection)

Both operations use in-place HCCL AllReduce (sendBuf == recvBuf).

### KV Cache Adjustment

Each TP rank has `num_kv_heads / tp_size` KV heads, so KV cache per rank is:
```
per_layer_bytes = num_blocks × block_size × (num_kv_heads / tp_size) × head_dim × 2
```

The `BlockManager` is unaffected — it manages logical block IDs, independent of head count.

---

## Pipeline Parallelism (PP) Design

### Layer Assignment

Layers are distributed evenly across PP stages, with remainder layers going to earlier stages:
```rust
pp_layer_range(total_layers: 36, pp_size: 4)
// Stage 0: layers 0..9
// Stage 1: layers 9..18
// Stage 2: layers 18..27
// Stage 3: layers 27..36
```

### Stage Boundaries

The plan compiler already handles:
- **First stage** (`pp_rank == 0`): has `Embedding`, no `Recv` at start
- **Last stage** (`pp_rank == pp_size - 1`): has `FinalNorm + LMHead + Sample`, no `Send` at end
- **Middle stages**: have `Recv` at start, `Send` at end

Hidden state shape transferred between stages: `[batch_size, seq_len, hidden_size]` (FP16).

### PP Weight Loading

PP-aware loading skips weights not needed for this stage:
- Skip `embed_tokens` if `pp_rank != 0`
- Skip `lm_head`, `model.norm` if `pp_rank != pp_size - 1`
- Only load layers in `pp_layer_range()`

### Pipeline Bubble

Basic PP has a bubble of `(pp_size - 1) / pp_size` idle compute per step.
Micro-batching (1F1B schedule) can reduce this — deferred to a future optimization phase.

---

## Data Parallelism (DP) Design

DP for inference requires no gradient synchronization. Each replica is fully independent:
- Separate process, separate Engine, separate KV cache, separate BlockManager
- HTTP server port: `base_port + dp_rank`
- External load balancer (nginx/HAProxy) distributes requests

No code coupling between DP replicas. The `distributed::DistributedConfig` correctly
maps world ranks to dp_rank for port assignment and future DP-aware scheduling.

---

## Communication Backend (HCCL)

### Crate Structure

```
rustBindings/
├── hccl-sys/           # Raw FFI: libhccl.so bindings
│   ├── src/lib.rs      # extern "C" declarations
│   └── src/types.rs    # HcclComm, HcclRootInfo, HcclDataType, HcclReduceOp, HcclResult
└── ascend/
    └── src/comm.rs     # Safe wrapper: HcclCommunicator (RAII)
```

### HcclCommunicator API

```rust
// Initialization
let root_info = get_root_info()?;          // rank 0 only
write_root_info_to_file(&root_info, path)?; // rank 0 writes
let root_info = read_root_info_from_file(path)?; // other ranks poll
let comm = HcclCommunicator::init_rank(n_ranks, &root_info, rank)?;

// Operations (all async on stream)
comm.all_reduce_sum(&stream, &buf, count, dtype)?;   // in-place
comm.send(&stream, &buf, count, dtype, dst_rank)?;
comm.recv(&stream, &buf, count, dtype, src_rank)?;
comm.all_gather(&stream, &send_buf, &recv_buf, count, dtype)?;
comm.broadcast(&stream, &buf, count, dtype, root)?;
// Drop: HcclCommDestroy called automatically
```

### Process Groups

Each parallelism dimension gets its own HCCL communicator:
- **TP group**: ranks with same `(pp_rank, dp_rank)`, different `tp_rank`
- **PP group**: ranks with same `(tp_rank, dp_rank)`, different `pp_rank`

Initialization: rank 0 of each group generates `HcclRootInfo`, writes to `/tmp/hccl_root_info/hccl_root_{group_id}.bin`. Other ranks poll with 60s timeout.

### Stream Strategy

Compute and communication share the same AscendCL stream for automatic serialization.
No explicit sync events needed. Future optimization: overlapping compute/comm with separate streams.

---

## Execution Plan Wiring

The `CompiledPlan::execute_paged()` now accepts `Option<&AscendCommOps>`:

```rust
ExecStep::AllReduceSum { tensor } =>
    comm_ops.all_reduce_sum_inplace(pool.get(*tensor))

ExecStep::Send { tensor, dst_rank } =>
    comm_ops.send_tensor(pool.get(*tensor), *dst_rank)

ExecStep::Recv { tensor, src_rank, .. } =>
    pool.put(*tensor, comm_ops.recv_tensor(&hidden_shape, DType::Float16, *src_rank))
```

Hidden shape for `Recv` is inferred as `[input_ids.len(), config.hidden_size]` at runtime.

---

## File Reference

### New Files

| File | Purpose |
|------|---------|
| `rustBindings/hccl-sys/Cargo.toml` | hccl-sys crate manifest (links libhccl.so) |
| `rustBindings/hccl-sys/build.rs` | CANN SDK search + link flags |
| `rustBindings/hccl-sys/src/lib.rs` | Raw FFI: HcclAllReduce, HcclSend, HcclRecv, etc. |
| `rustBindings/hccl-sys/src/types.rs` | HcclComm, HcclRootInfo (4120 bytes), enums |
| `rustBindings/ascend/src/comm.rs` | Safe HcclCommunicator (RAII, Drop) |
| `src/ops/ascend_comm.rs` | AscendCommOps: all_reduce_sum_inplace, send_tensor, recv_tensor, all_gather |
| `src/distributed/mod.rs` | DistributedConfig: rank mapping, group computation, env var loading |
| `src/distributed/process_group.rs` | init_process_groups(): file-based HCCL root info sharing |

### Modified Files

| File | Key Changes |
|------|-------------|
| `rustBindings/ascend/Cargo.toml` | Added `hccl = ["dep:hccl-sys"]` feature |
| `rustBindings/ascend/src/lib.rs` | Added `#[cfg(feature = "hccl")] pub mod comm` |
| `rust_llm_server/Cargo.toml` | Added `hccl-sys` optional dep; `ascend` feature enables `ascend/hccl` |
| `src/model/parallel.rs` | Added `dp_size`, `dp_rank` fields to `ParallelConfig` |
| `src/model/network.rs` | Added `Qwen3Model::new_sharded(config, parallel)` |
| `src/model/weights.rs` | Added `load_weights_sharded()`, `extract_shard()`, `shard_strategy_for_name()` |
| `src/engine/plan.rs` | Wired AllReduceSum/Send/Recv in `execute_paged()` to `AscendCommOps` |
| `src/engine/engine.rs` | Added `comm_ops` field + `set_comm_ops()`; TP/PP-aware KV cache allocation |
| `src/ops/mod.rs` | Added `#[cfg(feature = "ascend")] pub mod ascend_comm` |
| `src/main.rs` | Added `--dp`, distributed init from env vars, HCCL group init, DP port offset |

---

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | `hccl-sys` FFI bindings crate | ✅ Done |
| 2 | Safe HCCL wrapper (`ascend/comm.rs`) | ✅ Done |
| 3 | `AscendCommOps` + wire into execution plan | ✅ Done |
| 4 | Weight sharding for TP (`load_weights_sharded`, `new_sharded`) | ✅ Done |
| 5 | KV cache TP adjustment (`kv_heads / tp_size`) | ✅ Done |
| 6 | Multi-process launch & process groups (`distributed/`) | ✅ Done |
| 7 | Data parallelism (`dp_size`/`dp_rank`, port offset) | ✅ Done |
| 8 | PP micro-batching (1F1B pipeline schedule) | ⏳ Deferred (optimization) |
| 9 | lm_head TP handling (AllGather before sample) | ⏳ Deferred (lm_head is Replicate for now) |

**Test count**: 42 tests, all passing.

---

## Testing Plan

### Unit Tests (no NPU required, currently passing)

- Rank mapping math: `test_rank_mapping_tp4_pp2`, `test_rank_mapping_tp2_pp2_dp2`
- Group rank computation: `test_tp_group_ranks`, `test_pp_group_ranks`
- Weight sharding: `test_extract_shard_columns`, `test_extract_shard_rows`
- Shard strategy name lookup: `test_shard_strategy_for_name`
- Sharded model shapes: `test_qwen3_8b_sharded_tp4`, `test_qwen3_8b_sharded_pp2`
- Plan compilation with TP/PP: existing `test_compile_tp`, `test_compile_pp`

### Integration Tests (require Ascend NPU)

1. **HCCL smoke test** (2 NPUs): AllReduce a known tensor, verify result
2. **TP=2 correctness** (2 NPUs): Qwen3-0.6B output matches single-device (float tolerance)
3. **PP=2 correctness** (2 NPUs): Same comparison
4. **TP=2 + PP=2** (4 NPUs): Combined parallelism verification
5. **DP=2 throughput** (2 NPUs): Verify ~2x throughput with independent replicas

---

## Known Limitations & Future Work

1. **PP micro-batching**: Basic PP has a bubble of `(pp_size-1)/pp_size`. Implementing 1F1B
   scheduling would reduce idle time during prefill.

2. **lm_head replication**: `lm_head` is replicated rather than column-sharded. This wastes
   memory for large vocabularies. For Qwen3's 151K vocab at FP16, this is ~300MB per rank —
   acceptable for TP≤8 but could be optimized with AllGather before sampling.

3. **Single-sequence DP routing**: The HTTP server currently handles one sequence at a time.
   A smarter scheduler could route requests across DP ranks more efficiently.

4. **Continuous batching across requests**: The current FIFO scheduler is single-sequence.
   Batching multiple sequences (with different lengths) needs padding/dynamic shapes.

5. **DP group communicator**: The `ProcessGroups` struct currently omits a `dp_comm`. This
   would be needed if gradient sync (training) or DP-aware KV cache sharing were required.
