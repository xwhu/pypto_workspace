# Lingqu Data System

## Overview

The Lingqu system provides four basic distributed data services over the Unified Bus (UB) network.  
These four services form the foundation of the data plane. More advanced data services are built on top of them.

Core principle:
- Keep each base service focused on one data abstraction.
- Build higher-level capabilities (transactional workflows, caching layers, metadata services, analytics pipelines, etc.) by composing these base services.

## 1) `lingqu_shmem`

**Type:** Distributed shared memory service over UB network.

**Purpose:**
- Expose a global shared-memory abstraction across distributed nodes.
- Support low-latency data exchange and synchronization patterns for tightly coupled workloads.

**Typical usage:**
- Fast control/state sharing between distributed runtimes.
- Shared data regions for parallel compute stages.

### `lingqu_shmem` API and Memory Model (OpenSHMEM-aligned)

`lingqu_shmem` follows the OpenSHMEM programming model for the control plane:
- Initialization/finalization, PE discovery, synchronization, and one-sided communication semantics are kept compatible with OpenSHMEM APIs.
- This allows existing OpenSHMEM-style runtime logic to be reused with minimal adaptation in Lingqu environments.

After a shared-memory logical address is established by the control plane, `lingqu_shmem` is accessible from **Level 0 through Level 7** of the Lingqu hierarchy.

Within **Level 0-Level 2**, the mapped `lingqu_shmem` region is treated as part of **GM space**:
- The region is visible as global memory from the execution/data-plane perspective.
- `gm_tensor` objects can be placed in this region as **external memory references (external memref)**.
- These tensors can be directly accessed by PTO ISA load/store instructions, including `TLOAD` and `TSTORE`, without extra software copy steps.

This design cleanly separates responsibilities:
- OpenSHMEM-compatible APIs handle distributed coordination and memory setup (control plane).
- PTO ISA instructions (`TLOAD`/`TSTORE`) provide direct tensor data movement once mapping is established (data plane).

## 2) `lingqu_block`

**Type:** Distributed block access service.

**Purpose:**
- Provide remote block-level access to UB-attached block devices.
- Target devices connected to UB network, such as UB-SSU (for example, 1815E).

**Typical usage:**
- Direct block reads/writes for storage engines.
- Building higher-level file or object abstractions on top of block primitives.

### `lingqu_block` Access and Async Command Model

UB-attached block-level devices, such as SSD/SSU (for example, 1815E), are directly accessible from **Level 0 through Level 7** of the Lingqu hierarchy.

Each block device location is denoted by:
- `(UB_ADDRESS, LBA)`
- `UB_ADDRESS`: identifies the target device on the Unified Bus.
- `LBA` (Logical Block Address): identifies the target logical block on that device.

`lingqu_block` provides asynchronous read/write APIs:
- A read or write command is issued directly to the SSD/SSU device without CPU-mediated copy.
- The API accepts tensor arguments to integrate I/O directly with runtime dataflow.

Read semantics:
- A read command targets an output tensor argument allocated in the caller's memory hierarchy.
- The UB-level data path transfers block content directly into the output tensor buffer.
- When the read completes, the output tensor is marked as ready.
- Completion is reported to the Level-2 orchestrator runtime, which resolves dependencies by treating the read command as the **producer** of that output tensor.

Write semantics:
- A write command takes an input tensor argument as its payload.
- The write command is treated as a **consumer** of the input tensor.
- When write completion is reported, the orchestrator treats it as completion of input-tensor consumption, allowing the producer ring slot to be retired.

### `lingqu_block` Addressing and QoS

Block commands carry full `(UB_ADDRESS, LBA, length, flags)` descriptors:
- `UB_ADDRESS`: UB fabric endpoint address identifying the target SSU/SSD node.
- `LBA`: starting logical block address on that device.
- `length`: transfer size in bytes (must be a multiple of the device block size).
- `flags`: access hints such as priority, prefetch behavior, and completion notification mode.

The command submission is non-blocking: the issuing runtime enqueues the command in the device command ring and proceeds. Completion arrives asynchronously via a UB completion queue, which is polled by the scheduler.

Hierarchical dispatch:
- At L0–L2, commands are issued directly from AIC/AIV core programs via the PTO ISA.
- At L3–L7, commands are issued from CPU runtime threads using the `lingqu_block` user-space API.
- Regardless of issue level, completion always flows back to the Level-2 orchestrator for DAG resolution.

## 3) `lingqu_dfs`

**Type:** Distributed file system service over UB network.

**Purpose:**
- Provide file-level distributed storage semantics over Unified Bus.
- Support DFS implementations such as 3FS running on UB infrastructure.

**Typical usage:**
- Shared namespace and file I/O for distributed applications.
- Durable file storage for logs, checkpoints, and datasets.

### `lingqu_dfs` Namespace and API Model

`lingqu_dfs` provides a distributed file-system directory namespace to represent and organize files:
- Files are addressed through hierarchical paths and directory structures.
- A single global namespace is shared across all participating Lingqu nodes; any node at L3–L7 can access any file by path.
- The namespace provides familiar file-level abstractions for applications and runtime components.

Implementation approach:
- `lingqu_dfs` adopts popular distributed file-system implementations (for example, 3FS or HDFS-compatible FSes) and adapts them to run over Unified Bus protocol mechanisms.
- Key adaptation points:
  - **Metadata path**: directory lookup and inode operations are served over UB RPC channels, replacing TCP-based metadata server communication.
  - **Data path**: file block transfers use direct UB DMA to move data between client buffers and storage nodes, bypassing host-CPU memory copies.
  - **Consistency model**: strong consistency within a UB domain is achieved via UB-native atomic operations; relaxed cross-domain consistency modes are configurable.

Service API scope in the Lingqu hierarchy:
- `lingqu_dfs` APIs are accessible from **Level 3 (host) through Level 7**.
- Access is exposed as **POSIX-style file APIs** (open/read/write/close, pread/pwrite, fsync, directory traversal, stat, and file metadata operations).

### `lingqu_dfs` Access Patterns

Typical usage patterns from different hierarchy levels:

| Hierarchy Level | Typical Access Pattern |
|---|---|
| L3 (HOST) | Worker reads/writes local dataset files; checkpoint writes |
| L4 (POD) | Pod orchestrator reads shared configuration files |
| L5–L7 (cluster/global) | Cross-cluster metadata lookup; global dataset management |

The DFS is the primary persistence layer for the PyPTO distributed runtime: training checkpoints, model shards, activation tensors, and profiling logs are all routed through `lingqu_dfs`.

## 4) `lingqu_db`

**Type:** Redis API compatible distributed database, optimized for UB network.

**Purpose:**
- Offer key-value database semantics via Redis-compatible APIs.
- Optimize network and storage paths for high-throughput, low-latency UB environments.

**Typical usage:**
- Metadata/state service for distributed systems.
- Cache and fast lookup tier for application services.

### `lingqu_db` API and Performance Optimization Model

For DB service, Lingqu adopts an optimized implementation of **Redis-compatible DB access APIs** over the UB network, with targeted simplifications at every layer of the stack.

#### Supported API Surface

`lingqu_db` exposes a subset of Redis-style key-value operations focused on high-frequency access patterns:
- **String commands**: `GET`, `SET`, `MGET`, `MSET`, `DEL`, `EXPIRE`, `TTL`.
- **Hash commands**: `HGET`, `HSET`, `HMGET`, `HMSET`, `HDEL`, `HGETALL`.
- **List commands**: `LPUSH`, `RPUSH`, `LPOP`, `RPOP`, `LRANGE`.
- **Pub/Sub primitives**: `PUBLISH`, `SUBSCRIBE` for lightweight event notification.

Rarely used Redis commands, heavyweight scripting (Lua), and complex cluster management APIs are omitted to reduce implementation surface and minimize runtime overhead per request.

#### Function Simplification

- Each command maps to a single, fixed-size RPC descriptor that avoids dynamic allocation on the critical path.
- Command parsing is reduced to direct field extraction from a compact binary format, eliminating the overhead of Redis RESP (text protocol) parsing.
- Inline keys and short values (≤ 64 bytes) are carried directly in the command descriptor; large values use a separate UB-registered DMA buffer reference.

#### RPC Protocol Simplification

- The on-wire protocol uses a compact binary header (12 bytes) carrying: command opcode, key hash, value length, and a 32-bit correlation ID.
- Request/response matching uses the correlation ID with a lock-free per-thread table, eliminating global serialization.
- Batched command pipelining is first-class: a single UB send can carry multiple command descriptors with a shared completion notification.

#### Data Transfer Optimization

- For large values (> 64 bytes), a zero-copy DMA path transfers data directly between client tensor buffers and server storage.
- Small values use inline encoding to keep everything within the initial RPC message, avoiding extra round-trip latency.
- UB multicast is used for `PUBLISH` to fan out to multiple `SUBSCRIBE` receivers in a single UB send operation.
- Server-side write coalescing batches multiple `SET`/`HSET` updates into single storage writes where ordering constraints allow.

#### System Goals

- **Lower per-request overhead**: eliminate text protocol parsing, reduce lock contention, use fixed-size descriptors.
- **Increase throughput**: batching, pipelining, and UB multicast reduce total message count per logical operation.
- **Improve QPS under high concurrency**: lock-free correlation table + per-thread command queues scale with core count without contention bottlenecks.
- **Predictable latency**: inline small values and direct DMA for large values eliminate variable-latency copies on the hot path.

## Layering Model

These four base services are complementary:
- `lingqu_shmem`: memory abstraction
- `lingqu_block`: block abstraction
- `lingqu_dfs`: file abstraction
- `lingqu_db`: database/key-value abstraction

Advanced services can be composed from one or multiple base layers.  
Examples include distributed caching with persistence, data lifecycle management, workflow checkpointing services, and application-specific storage middleware.
