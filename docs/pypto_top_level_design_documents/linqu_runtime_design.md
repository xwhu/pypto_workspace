# Linqu Distributed Runtime: Full System Design

This document provides a comprehensive design for the **Linqu Distributed Runtime** — the software system that executes PyPTO programs across a hierarchical cluster of machines. It synthesizes and extends the Linqu machine hierarchy model, the PyPTO scope/function grammar, the multi-layer ring buffer memory management system, and the distributed cluster orchestration architecture.

---

## 1. Design Goals and Principles

### 1.1 Hierarchical Symmetry

The runtime software architecture must **mirror** the physical Linqu machine hierarchy (Levels 0–6). Every runtime component — task identity, memory management, scheduling, and communication — is structured around the same hierarchy that the hardware and compiler use. There is no separate "software topology" distinct from the machine model.

### 1.2 Deterministic O(1) Resource Management

Resource lifetime and reclamation use **ring buffers** indexed by scope depth, not garbage collection. Allocation and retirement are O(1) per operation. Inner scopes can retire independently of outer scopes.

### 1.3 Zero-Configuration Cluster Discovery

Nodes identify themselves and discover peers using **deterministic, rule-based IP-to-coordinate mapping** and **gossip-based membership**. No static IP lists, no centralized DNS, no manual topology files.

### 1.4 Code and Data Residency

Minimize network overhead via a **two-phase model**: register code/data blobs once (by content hash), invoke many times by handle. Remote nodes are treated as **persistent environments**, not ephemeral task executors.

### 1.5 Logical Isolation (Multi-Tenancy)

A single physical cluster supports multiple **Logical Linqu Systems** identified by namespace strings. Different logical systems sharing the same hardware cannot observe or interfere with each other's memory, tasks, or ring buffers.

### 1.6 Unified Programming Model

The same `pl.Level.*` hierarchy parameter, `pl.at()` scope grammar, and `@pl.function(level=...)` decorator work consistently from core-level kernels (Level 0) up to cross-rack cluster programs (Level 6). The compiler generates **hierarchy labels** on all functions, reserved for the runtime to dispatch at the appropriate level.

---

## 2. The Linqu Machine Hierarchy

### 2.1 Levels Defined Bottom-Up

| Level | Name | Description | pl.Level aliases |
|-------|------|-------------|------------------|
| **0** | **Core / Core-group** | **Core:** single execution unit (AIV or AIC). **Core-group:** scheduling unit grouping multiple cores with local affinity (e.g. 1 AIC + 2 AIV), sharing TPUSH/TPOP ring; runs InCoreFunctionGroup. | `AIV`, `AIC`, `CORE_GROUP` |
| **1** | **Chip die** | One die; contains multiple cores/core-groups. May be **omitted** in single-die chip models. | `CHIP_DIE`, `L2CACHE` |
| **2** | **Chip** | One chip; contains one or more dies or directly multiple cores. Unified memory address space (UMA). | `CHIP`, `PROCESSOR`, `UMA` |
| **3** | **Host** | A single OS instance; one or more chips; runs orchestration. | `HOST`, `NODE` |
| **4** | **Cluster-level-0** | First cluster tier; usually a single server or pod; high bandwidth, tight coupling. | `CLUSTER_0`, `POD` |
| **5** | **Cluster-level-1** | Second cluster tier; usually a supernode; high-bandwidth domain across nodes. | `CLUSTER_1`, `CLOS1` |
| **6** | **Cluster-level-2** | Third cluster tier; cross-rack or wider-area with contracted bandwidth. | `CLUSTER_2`, `CLOS2` |

### 2.2 Recursive Enclosure

Each level is a logical machine that **encloses several instances** of the level below, recursively forming the complete system:

```
Cluster-level-2  encloses several →  Cluster-level-1
Cluster-level-1  encloses several →  Cluster-level-0
Cluster-level-0  encloses several →  Host
Host             encloses several →  Chip
Chip          (opt) encloses several →  Chip die
Chip / Chip die  encloses several →  Core / Core-group
```

### 2.3 Bandwidth and Coupling Gradient

As level number increases, the domain grows and interconnect bandwidth becomes more constrained:

- **Level 0** (Core-group): intra-cluster DMA, TPUSH/TPOP — highest bandwidth, lowest latency.
- **Level 1–2** (Chip die / Chip): on-chip interconnect, shared L2 cache or UMA memory.
- **Level 3** (Host): PCIe, NVLink, or similar host-to-device links.
- **Level 4** (Pod): high-bandwidth intra-server or intra-pod network (e.g. NVLink, InfiniBand within rack).
- **Level 5** (Supernode): high-bandwidth domain across nodes (e.g. fat-tree spine within supernode).
- **Level 6** (Cross-rack): contracted bandwidth, multi-hop routing.

The runtime uses this gradient to make communication decisions: prefer local data transfer at lower levels; serialize and RPC at higher levels.

---

## 3. Existing Runtime (`simpler`), Project Scope, and Extension Plan

### 3.1 The `simpler` Runtime: Level 0–2 (Do Not Modify)

The existing runtime, implemented in the **`pypto_workspace/simpler`** repository, already handles **Levels 0–2** of the Linqu hierarchy:

- **Level 0** (Core / Core-group): InCore functions, AIC/AIV kernels, and InCoreFunctionGroups execute on cores and core-groups. TPUSH/TPOP co-scheduling within a core-group.
- **Level 2** (Chip): **Orchestration functions** run at the chip level. The `simpler` runtime manages task submission, ring buffers (single-ring today), tensor lifetime, scope-exit semantics, and producer/consumer tracking — all within a single chip.
- **Level 1** (Chip die): **Omitted** on current chip models. In the future, the `simpler` repo will be enhanced to support Level 1 when multi-die chips become available. This is an internal `simpler` concern.

**Guiding rule: Do not change `simpler`.** The `simpler` design and codebase are treated as a **fixed, existing capability**. This project **adapts to** the `simpler` interfaces and **builds on top of** them, never modifying them.

### 3.2 This Project: Build the Level 2–6 Runtime

This project's scope is to build a **coherent runtime that covers Levels 2 through 6** (Chip through Cluster-level-2) in one unified design:

- **Level 2** (Chip): the interface layer — a future `ChipBackend` adapter (within this project) will call `simpler`'s existing API through a stable ABI (dynamic linking, no compile-time dependency) to dispatch chip-level work. In Phase 0, L2 dispatch is stubbed. The adapter does **not** modify `simpler`.
- **Level 3** (Host): multi-chip coordination within a single OS instance. This is the first new capability.
- **Levels 4–6** (Cluster-level-0/1/2): multi-host coordination across pods, supernodes, and racks. These are built as extensions of Level 3.

The architecture is:

```
┌─────────────────────────────────────────────────────────────────────┐
│  This project: Linqu Distributed Runtime (Levels 2–6)               │
│  Level 6: Cluster-level-2 (cross-rack)                              │
│  Level 5: Cluster-level-1 (supernode)                               │
│  Level 4: Cluster-level-0 (pod)                                     │
│  Level 3: Host (multi-chip coordination)                            │
│  Level 2: Chip (adaptation layer — future ChipBackend adapter)      │
├─────────────────────────────────────────────────────────────────────┤
│  simpler runtime (Levels 0–2, DO NOT MODIFY)                        │
│  Level 2: Chip orchestration (task ring, buffer ring, scope mgmt)   │
│  Level 1: Chip die (omitted for now; simpler will add later)        │
│  Level 0: Core / Core-group (InCore, AIC, AIV, TPUSH/TPOP)         │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Adaptation to `simpler`

The Linqu runtime **adapts to** the `simpler` runtime at the Level 2 boundary:

1. **Calling convention:** When the Linqu runtime needs to execute a chip-level (Level 2) function, a future `ChipBackend` adapter will call `simpler`'s existing orchestration API through dynamic linking (`dlopen`/`dlsym`). The Linqu runtime does **not** re-implement task rings, buffer rings, or scope-exit logic for Level 0–2 — those remain `simpler`'s responsibility. In Phase 0, this dispatch is stubbed. The `ChipBackend` adapter lives within `pypto_runtime_distributed` and does **not** modify `simpler`.

2. **Identity mapping:** `simpler` uses its own task identity scheme internally (e.g. `task_id` within a chip). The Linqu runtime maps this to the full `TaskKey(logical_system, L6..L0, scope_depth, task_id)` coordinate. The `simpler`-internal `task_id` becomes the `L0/L2` portion of the full key.

3. **Ring buffer layering:** For Levels 0–2, `simpler` manages its own ring buffers. For Levels 3–6, the Linqu runtime manages **its own** ring buffers (`task_ring[L][d]`, `buffer_ring[L][d]` for `L ≥ 3`). The two ring systems are independent; the Linqu runtime does not modify `simpler`'s rings.

4. **Scope nesting:** `simpler` manages scope depth within a chip. The Linqu runtime manages **higher-level scopes** (e.g. a host-level scope that contains multiple chip-level scopes). When the Linqu runtime exits a Level 3 scope, it performs Level 3 ring retirement; the chip-level scope exits inside `simpler` are handled by `simpler` independently.

5. **Memory domain boundary (h2d/d2h):** The L3 runtime (host CPU) and L2 runtime (`simpler` on device) operate in **different memory domains** — host DRAM vs. device Global Memory. Data crosses this boundary via explicit `h2d_copy` (host→device) and `d2h_copy` (device→host) DMA operations. The L3 `ChipBackend` adapter is responsible for managing these transfers. Tensor handles in the Linqu runtime (opaque integers) are mapped to actual device GM addresses by the adapter during `h2d_copy`. See §7.4 (Tier 2) for the detailed communication model.

6. **Future Level 1 support:** When `simpler` adds Level 1 (chip-die) support, it will be transparent to the Linqu runtime. The Linqu runtime only interacts with `simpler` at the Level 2 boundary.

### 3.4 Hardware Verification Environment Limitation

The full 7-layer Linqu system (Levels 0–6) is the **target architecture**, but the physical hardware for all seven layers is **not yet available**:

- **Levels 0–2**: Available today via `simpler` on current chips (Level 1 omitted on single-die models).
- **Levels 3–6**: While individual servers exist, the full multi-level cluster hardware with distinct Level 4/5/6 network fabrics is **not yet ready** for verification.

Therefore, the **first implementation** of the Linqu runtime targets a hardware environment with **Level 3 only** — a single host (one OS instance) managing one or more chips via `simpler`. This is the most constrained but immediately verifiable environment.

### 3.5 Forward-Compatible Software Design Requirement

Despite the Level 3–only hardware verification environment, the **software design must be compatible with a hardware environment that has all 7 layers** (Levels 0–6). This means:

1. **All data structures are parameterized by hierarchy level.** Ring buffers are `task_ring[L][d]` and `buffer_ring[L][d]` where `L` ranges over Levels 3–6 (managed by this project) plus the adaptation interface to `simpler` for Level 0–2. In the first implementation, only `L = 3` has non-zero capacity in the Linqu runtime; Levels 4–6 have zero-capacity rings but the indexing and data structures exist.

2. **All APIs accept `pl.Level` for any level.** The `pl.at(level=...)`, `@pl.function(level=...)`, and runtime dispatch interfaces accept any `pl.Level` value. For `L ∈ {0, 1, 2}`: dispatch is delegated to `simpler`. For `L = 3`: handled by the Linqu runtime. For `L ∈ {4, 5, 6}`: raise a clear `NotYetSupported` error at **runtime** (not compile time) in the first implementation.

3. **Task identity uses the full coordinate.** `TaskKey` always includes all level indices, even if most are zero in the first implementation.

4. **The RPC protocol header includes all level fields.** The `LinquHeader` struct carries `l6_idx`, `l5_idx`, `l4_idx`, `l3_idx` from day one. In the first implementation, `l4–l6` are always zero, but the wire format is stable.

5. **The PeerRegistry and gossip protocol are designed for multi-level.** In the first implementation, discovery is trivial (single host, no network gossip needed), but the data structures exist and are tested with mock multi-level topologies.

6. **The compiler generates hierarchy labels for all levels.** Even if the runtime ignores Level 4–6 labels today, the compiler emits them.

### 3.6 Compiler Contract: Hierarchy Labels

The compiler **must** generate a **hierarchy label** (a `pl.Level` tag) on every function it outlines or assigns to a level. This label is attached to the function in the IR and emitted metadata. Even if the runtime does not yet support a given level, the label is **present and reserved** so that:

- Future runtime versions can dispatch and schedule using these labels without changing the codegen contract.
- The same compiled binary can run on runtimes with different levels of multi-level support.
- Tools (profilers, debuggers) can display the hierarchy level of each function.

### 3.7 Extension Roadmap

| Phase | Levels Exercised | Hardware Environment | Capability |
|-------|-----------------|---------------------|------------|
| **Phase 0 (first impl.)** | 0–2 via `simpler`, 3 via Linqu runtime | Single host, one or more chips | Core/core-group execution via `simpler`; chip orchestration via `simpler`; **host-level multi-chip coordination** via Linqu runtime. Software design forward-compatible with all 7 layers. |
| **Phase 1** | 0–2 via `simpler` (with L1), 3 via Linqu | Host with multi-die chips | Level 1 added inside `simpler` when multi-die chips ship. Transparent to Linqu runtime. |
| **Phase 2** | 0–2 via `simpler`, 3–4 via Linqu | Multi-host pod/server | Cluster-level-0: intra-pod multi-host coordination. Activate gossip discovery, distributed SCOPE_EXIT, SPMD fan-out. |
| **Phase 3** | 0–2 via `simpler`, 3–5 via Linqu | Supernode | Cluster-level-1: hierarchical dispatch through Level-5 leaders. |
| **Phase 4** | 0–2 via `simpler`, 3–6 via Linqu | Full cluster | Cluster-level-2: cross-rack dispatch, aggregation nodes, contracted-bandwidth-aware scheduling. |

---

## 4. Task Identity and Coordinate System

### 4.1 The Task Key

Every task in the system is uniquely identified by a **hierarchical coordinate**:

```
TaskKey = (logical_system, L6, L5, L4, L3, L2, L1, L0, scope_depth, task_id)
```

In practice, many levels may be elided (e.g. if Level 1 is omitted, or if the program runs on a single host). The minimal form used within a single chip is:

```
TaskKey = (scope_depth, task_id)
```

This dual tag is the **canonical identifier** for all task lookup, producer/consumer references, ring buffer indexing, and retirement decisions. `task_id` alone is **never** used for correctness decisions; it is only unique within a scope layer.

### 4.2 User-Defined Coordinate Mapping

For cluster-level operation (Levels 3–6), each node computes its position in the hierarchy from its IP address using a **user-defined** mapping function:

```python
def get_my_coordinates(ip_address: str) -> dict:
    parts = [int(x) for x in ip_address.split('.')]
    return {
        "level_5": parts[1],   # Cluster-level-1 (supernode)
        "level_4": parts[2],   # Cluster-level-0 (pod/rack)
        "level_3": parts[3],   # Host
    }
```

The mapping function is **not hard-coded** into the runtime. Different cluster deployments may use different IP assignment schemes. The runtime loads this function at startup.

### 4.3 Logical and Physical System Names

To support multi-tenancy on shared physical infrastructure, each node carries two namespace strings:

- **`linqu_physical_system_name`**: identifies the hardware (e.g. `"dc-north-rack-05"`). Used for operational monitoring, failure diagnosis, and hardware-level addressing.
- **`linqu_logical_system_name`**: identifies the application or tenant (e.g. `"llm-training-v2"`). Used for **runtime isolation** — ring buffers, code caches, task keys, and membership tables are scoped to the logical system.

A single physical cluster can host multiple logical Linqu systems simultaneously. A node may participate in multiple logical systems.

---

## 5. Memory and Task Management: The Multi-Layer Ring Stack

### 5.1 Overview

Instead of a global heap or garbage collector, the runtime manages memory using a **multi-layer ring stack**. For every hierarchy level `L` and every scope depth `d` within that level, the runtime maintains:

- `task_ring[L][d]`: ring buffer of task metadata and execution status.
- `buffer_ring[L][d]`: ring buffer of output tensor / data buffer slots.
- `last_task_alive[L][d]`: the retirement head pointer for that ring.

### 5.2 Scope Depth and Ring Layers

Programs use `pl.scope()` to create nested scopes. Each scope depth `d` maps to a ring layer:

**On `scope.enter`:**
1. `current_scope_depth += 1`
2. Bind all new allocations to ring layer `d = current_scope_depth`.

**On task creation at depth `d`:**
1. Allocate task slot from `task_ring[L][d]`.
2. Allocate output buffers from `buffer_ring[L][d]`.
3. Assign task key as `(d, task_id_in_layer)`.
4. Initialize `fanout_count = 1` (the mandatory scope-exit token).
5. Register producer/consumer metadata.

**On `scope.exit`:**
1. Mark tasks/tensors in this scope frame as out-of-scope.
2. Apply scope-exit token: for each task in scope, if `task_freed == 0`, increment `ref_count += 1`.
3. Trigger layer-local retirement scan for depth `d`.
4. `current_scope_depth -= 1`.

### 5.3 Retirement Rules

A task and its output buffers in `buffer_ring[L][d]` are reclaimable when **both** conditions hold:

1. **Scope token applied**: either `scope.exit()` has been reached, or `pl.free(tensor)` was called for the tensor's producer task.
2. **Reference count satisfied**: `ref_count == fanout_count` (all consumers have completed).

Retirement is **layer-local**: progress in an inner scope (depth `d+1`) is **not** blocked by a stalled task in an outer scope (depth `d`). Each ring advances its own `last_task_alive` pointer independently.

### 5.4 The `pl.free(tensor)` Optimization

`pl.free(tensor)` allows the programmer to explicitly end a tensor's scope lifetime before the lexical scope exits. This is useful for tensors in broad, high-level scopes whose memory would otherwise be pinned until the scope naturally closes.

**Runtime API: `pto_rt.free(outbuf)`**

1. Lookup `(task_scope_level, task_id)` from `outbuf` via tensor map.
2. If `task_freed == true`, return (idempotent).
3. Set `task_freed = true`.
4. Increment `ref_count += 1`.
5. Return.

**Interaction with `scope.exit()`:**

At scope exit, when iterating tasks:
- If `task_freed == false`: apply scope token (`ref_count += 1`).
- If `task_freed == true`: skip (token was already applied by `pl.free`).

This guarantees **exactly-once** scope token application.

### 5.5 Producer/Consumer Metadata

All references use the dual-tag `TaskKey(scope_level, task_id)`, never `task_id` alone:

| Record | From (legacy) | To (multi-layer) |
|--------|---------------|-------------------|
| Tensor producer | `producer_task_id` | `producer_task_key = (scope_level, task_id)` |
| Tensor consumer list | `consumer_task_id[]` | `consumer_task_key[] = (scope_level, task_id)[]` |
| Tensor map | N/A | `outbuf -> (task_scope_level, task_id)` |
| Debug/profile events | `task_id` | `(scope_level, task_id)` |

### 5.6 Correctness Invariants

The following invariants must hold at all times:

1. **No early free**: never reclaim unless `ref_count == fanout_count`.
2. **Exactly-once scope token**: `pl.free()` + `scope.exit()` applies the token exactly once per task.
3. **Layer isolation**: `last_task_alive[d]` only controls layer `d`.
4. **Deterministic behavior**: same program order yields same reclamation decisions.
5. **Task identity uniqueness**: `TaskKey(scope_level, task_id)` is globally unique at runtime.
6. **Compatibility**: programs without `pl.free()` behave identically to current single-ring runtime.

---

## 6. The Unified Programming Grammar

### 6.1 Hierarchy Level Enum: `pl.Level`

A first-class enum aligns with the Linqu hierarchy and is used in both explicit and implicit function declarations:

| Symbol | Linqu Level | Readability Aliases |
|--------|-------------|---------------------|
| `pl.Level.AIV` | 0 (Core) | — |
| `pl.Level.AIC` | 0 (Core) | — |
| `pl.Level.CORE_GROUP` | 0 (Core-group) | — |
| `pl.Level.CHIP_DIE` | 1 | `pl.Level.L2CACHE` |
| `pl.Level.CHIP` | 2 | `pl.Level.PROCESSOR`, `pl.Level.UMA` |
| `pl.Level.HOST` | 3 | `pl.Level.NODE` |
| `pl.Level.CLUSTER_0` | 4 | `pl.Level.POD` |
| `pl.Level.CLUSTER_1` | 5 | `pl.Level.CLOS1` |
| `pl.Level.CLUSTER_2` | 6 | `pl.Level.CLOS2` |

### 6.2 Explicit Function Declaration

```python
@pl.function(level=pl.Level.AIV)
def my_aiv_kernel(...): ...

@pl.function(level=pl.Level.CHIP)
def my_orchestration(...): ...

@pl.function(level=pl.Level.CLUSTER_0)
def my_cluster_function(...): ...
```

The compiler tags the function with its hierarchy label. The runtime dispatches it to the appropriate level.

### 6.3 Implicit Function Declaration: `pl.at()`

A single context manager with an optional `optimization` parameter:

```python
with pl.at(level=pl.Level.CORE_GROUP):
    # One function at core-group level.
    ...

with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
    # Compiler splits loops, interchanges, outlines one or more functions.
    ...

with pl.at(level=pl.Level.CLUSTER_0):
    # One function dispatched to all nodes in cluster-level-0.
    ...
```

**Parameters:**

| Parameter | Meaning |
|-----------|---------|
| `level` | Target hierarchy level. Block is outlined as function(s) at this level. |
| `optimization` | Optional. Transformation before outlining: `pl.chunked_loop_optimizer`, `pl.fully_unroll_static_loop`, etc. Omit for a single outlined function. |

**Backward compatibility:**
- `with pl.incore` → `with pl.at(level=pl.Level.CORE_GROUP)`
- `with pl.auto_incore` → `with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer)`

### 6.4 Function Role Extension: `role={orchestrator|worker}`

To support explicit role separation in L3-L7 software runtimes, both explicit and implicit grammar forms are extended with a role argument:

```python
@pl.function(level=pl.Level.HOST, role=pl.Role.ORCHESTRATOR)
def host_orchestrator(...): ...

@pl.function(level=pl.Level.HOST, role=pl.Role.WORKER)
def host_worker(...): ...

with pl.at(level=pl.Level.POD, role=pl.Role.ORCHESTRATOR):
    ...
```

Role semantics:
- `ORCHESTRATOR`: builds DAG/task graph, submits child tasks (same-level workers or next-level orchestrators), manages tensor/task ring allocation.
- `WORKER`: executes concrete compute/data tasks; may invoke lower-level orchestrator dispatch via RPC.

Default behavior:
- If role is omitted, runtime treats the function as `ORCHESTRATOR` for backward compatibility with existing compiler outputs.

---

## 7. Distributed Runtime Architecture (Levels 3–6)

### 7.1 Component Stack

The distributed runtime consists of four layers:

```
┌──────────────────────────────────────────────────────────┐
│  Orchestrator (Level 3+ code path)                        │
│  - Executes main control flow                             │
│  - Manages pl.scope, dispatches SPMD blocks               │
│  - Sends REG_CODE / REG_DATA / CALL_TASK / SCOPE_EXIT     │
├──────────────────────────────────────────────────────────┤
│  Node Daemon (Level 3, runs on every host)                │
│  - Listens for tasks                                      │
│  - Manages local buffer_ring[L][d] and task_ring[L][d]    │
│  - Executes code blobs locally on chips                   │
│  - Reports heartbeats via gossip                          │
├──────────────────────────────────────────────────────────┤
│  Rendezvous Ledger (Distributed, gossip-based)            │
│  - Membership table: (logical_system, coord) → IP/status  │
│  - No central server; peer-to-peer convergence            │
├──────────────────────────────────────────────────────────┤
│  Transport (Binary RPC over TCP/UDP)                      │
│  - Zero-copy binary protocol (FlatBuffers or equivalent)  │
│  - Handle-based: register once, invoke many               │
└──────────────────────────────────────────────────────────┘
```

### 7.2 The Orchestrator

The **Orchestrator** runs the top-level PyPTO program. It is the entity that:

1. Executes `pl.scope()` enter/exit at the cluster level, managing `current_scope_depth` for distributed scopes.
2. Resolves `pl.at(level=pl.Level.CLUSTER_0)` by querying the Rendezvous Ledger for live nodes at the target level and dispatching the outlined function to all of them.
3. Sends `REG_CODE` and `REG_DATA` messages to register code blobs and data buffers on remote nodes.
4. Sends `CALL_TASK` messages containing only the blob hash, data handles, and SPMD index — not the full binary.
5. Sends `SCOPE_EXIT` signals when a scope closes, triggering distributed ring layer retirement.

### 7.3 The Node Daemon

Every host runs a lightweight **Node Daemon** that:

1. On startup, calls the user-defined `get_my_coordinates(ip)` to compute its position in the hierarchy.
2. Joins the cluster by probing deterministic seed IPs (see §8) and participating in gossip.
3. Maintains local `task_ring[L][d]` and `buffer_ring[L][d]` for the hierarchy levels it manages.
4. Stores received code blobs in a `CodeCache[blob_hash] → FunctionPointer` map.
5. Stores received data in a `DataCache[data_handle] → BufferPointer` map.
6. On `CALL_TASK` receipt: looks up the blob hash in CodeCache, looks up data handles in DataCache, executes the function with the given `spmd_idx`.
7. On `SCOPE_EXIT` receipt at depth `d`: retires all tasks and buffers at that scope depth, advancing `last_task_alive[L][d]` and freeing ring slots.

### 7.3A L3-L7 Thread Model and Role Separation (Enhancement)

For L3-L7 CPU runtime instances, initialization uses a three-role thread model:

1. **Orchestrator thread (1 thread):**
   - Builds DAG graph of submitted tasks.
   - Allocates task/tensor ring entries.
   - Submits tasks to same-level workers or to next-level orchestrators.

2. **Scheduler threads (`Lx_NUM_SCHEDULER_THREADS`, default = 1):**
   - Track task state transitions.
   - Resolve DAG dependencies.
   - Dispatch ready tasks to worker queues.

3. **Worker threads (`Lx_NUM_WORKER_THREADS`, test default = 4):**
   - Execute worker functions on CPU.
   - For orchestrator-targeted child work, send RPC dispatch to lower-level runtime.

This is intentionally aligned with the simpler L0-L2 control model (orchestrator/scheduler/ring/worker), with one key difference:
- **L3-L7 workers run on CPU threads** in host processes.
- **L0-L2 workers run on AIC/AIV cores** through the simpler runtime.

#### 7.3A-1 Tensor Output Buffer Allocation at Task-Submit Time

`make_tensor(n)` / `create_tensor(n)` creates a Tensor descriptor with **metadata only**:
handle, element count, and a readiness flag.  **No backing storage is allocated** by
`make_tensor`.  The underlying buffer is a zero-size placeholder at this stage.

Storage is allocated from the level's **memory ring** (HeapRing equivalent) inside
`submit_worker()`, as the very first step before the task is enqueued:

```
make_tensor(n)        →  Tensor{handle, count, data=∅, ready=false}   (no alloc)
submit_worker(…, out) →  out.data = memory_ring.alloc(n)              (alloc here)
                         enqueue task in task ring
worker executes       →  writes into out.data[0..n-1]
                         marks out.ready = true
```

This matches the simpler (L0-L2) runtime's protocol where `alloc_tensor` claims a
HeapRing slot and records its GM address in the Tensor descriptor, and the worker
writes the result directly into that pre-allocated slot.

**Rationale:** Deferring allocation to submit time avoids reserving ring space for tasks
that have not yet been committed to the DAG, and keeps the memory ring a simple
monotonic allocator with deterministic back-pressure.

#### 7.3A-2 Task Dependency DAG Tracks Tensor Parameters Only

The scheduler resolves dependencies by inspecting `TaskDesc::input_handles`, which
lists **only tensor-typed parameters** of the task function.  Scalars, integers,
strings, file paths, and other non-tensor arguments are **not tracked** in the DAG.

```
submit_worker(fn, inputs=[Tensor A, Tensor B], outputs=[Tensor C])
    → TaskDesc.input_handles  = {A.handle, B.handle}   ← DAG edges
    → TaskDesc.output_handles = {C.handle}
```

Non-tensor arguments (hierarchy coordinates, loop indices, file paths, etc.) are passed
to the worker via lambda captures or struct fields and are **invisible to the scheduler**.
The scheduler promotes a task from PENDING → READY solely when every handle in
`input_handles` has `is_ready() == true`.  No scalar conditions are evaluated.

This is consistent with the simpler runtime, where `Tensor` parameters carry buffer
addresses and dependency lists, while scalar arguments (SPMD index, configuration
values) are passed as plain integers in the task payload.

#### 7.3A-2a Tree Reduction Pattern: Parallel Worker Submission via Tensor-Map DAG

A canonical orchestration pattern for aggregation is **binary tree reduction**.  The
orchestrator submits **all rounds of the tree upfront** (in a single sequential pass),
without waiting between rounds.  The tensor map drives the dispatch order automatically.

```
Orchestrator builds the full tree DAG in one pass:

  Phase 1 — leaf workers (no tensor deps; all immediately READY)
    submit_worker(leaf_reader, inputs=[], outputs=[L0])     ← rt_l3
    submit_worker(leaf_reader, inputs=[], outputs=[L1])     ← rt_l3
    ...
    submit_worker(leaf_reader, inputs=[], outputs=[L15])    ← rt_l3

  Phase 2 — round 1 pair-sum workers (deps on leaf outputs)
    submit_worker(pair_sum, inputs=[L0,  L1 ], outputs=[R1_0])   ← rt_l4
    submit_worker(pair_sum, inputs=[L2,  L3 ], outputs=[R1_1])   ← rt_l4
    ...
    submit_worker(pair_sum, inputs=[L14, L15], outputs=[R1_7])   ← rt_l4

  Phase 3 — round 2 pair-sum workers (deps on round-1 outputs)
    submit_worker(pair_sum, inputs=[R1_0, R1_1], outputs=[R2_0]) ← rt_l4
    ...

  Phase 4, 5 — further rounds ...

  Root — final worker
    submit_worker(pair_sum, inputs=[R3_0, R3_1], outputs=[ROOT]) ← rt_l4

  Orchestrator then waits: root_future.get()
```

Key properties of this pattern:

- The orchestrator submits all `(N − 1)` internal-node workers **before any of them
  execute**.  No explicit synchronisation between tree rounds is written in the
  orchestrator.
- Each `pair_sum_worker` declares exactly **2 tensor-typed inputs** in
  `TaskDesc::input_handles`.  The scheduler holds it `PENDING` until both inputs
  are `is_ready() == true`, then promotes it to `READY` and dispatches to a worker
  thread automatically.
- Leaf workers and sibling subtrees at the same round execute **in parallel** (up to
  the worker thread count).  Deeper rounds stall naturally until earlier rounds
  produce their output tensors — purely through the tensor-map dependency mechanism.
- The orchestrator function itself is **pure structure** (no compute, no polling):
  it runs to completion rapidly, having only built the DAG and submitted all workers.

For N leaves and a binary tree:

| Total workers submitted | `N` leaf readers + `N−1` internal nodes |
|------------------------|------------------------------------------|
| Tree depth             | `⌈log₂ N⌉` rounds                       |
| Scheduler decisions    | one PENDING→READY check per internal node |
| Orchestrator waits     | once, on the root future                |

#### 7.3A-3 Shared Data Structures and Synchronisation Within the Same Process

At each hierarchy level (L3–L7), the orchestrator thread, scheduler thread(s), and
worker threads all reside in **the same OS process** and therefore share a common
address space.  This allows zero-copy sharing of runtime state:

| Shared structure      | Owning / writing role    | Reading roles            | Protection mechanism |
|-----------------------|--------------------------|--------------------------|----------------------|
| `TensorRegistry`      | Orchestrator (insert), Worker (ready flag) | Scheduler (is_ready) | `std::mutex` + `shared_ptr<atomic<bool>>` for ready |
| `pending_tasks_`      | Orchestrator (append), Scheduler (drain) | Scheduler              | `std::mutex pending_mu_` |
| `ready_queue_`        | Scheduler (enqueue)      | Worker (dequeue)         | `std::mutex worker_mu_` + `condition_variable` |
| `orch_queue_`         | External callers (push)  | Orchestrator (pop)       | `std::mutex orch_mu_` + `condition_variable` |
| `TaskDesc::state`     | All roles                | All roles                | `std::atomic<TaskState>` with CAS |
| `Tensor::ready`       | Worker (store release)   | Scheduler (load acquire) | `shared_ptr<atomic<bool>>` with acquire/release |

**Key synchronisation rules (following the simpler L0-L2 approach):**

- **Per-tensor readiness** uses `std::atomic<bool>` with `memory_order_release` on
  write (by the worker) and `memory_order_acquire` on read (by the scheduler).  This
  is the exact pattern used by simpler's `STORE_RELEASE` / `LOAD_ACQUIRE` on ring
  pointer and task state fields in device GM.

- **PENDING → READY promotion** uses `atomic<TaskState>::compare_exchange_strong`
  (CAS) so that multiple scheduler threads cannot both promote the same task.

- **Queue wakeups** use `std::mutex` + `std::condition_variable` to avoid busy-wait.
  Workers call `sched_cv_.notify_all()` after marking tensors ready, mirroring the
  way simpler's scheduler wakes up when the orchestrator advances `current_task_index`.

- **Cross-level tensor readiness** (a tensor produced by an L3 worker unblocking an L4
  task) propagates automatically because the `Tensor::ready` shared_ptr is shared
  across runtime instances.  The L4 scheduler detects the flag change on its next
  wakeup (short `wait_for` timeout), without any explicit cross-runtime notification.

- **TensorHandle values must be globally unique** across all hierarchy levels within
  the same process.  Each level uses the same global atomic handle counter to avoid
  collisions: if L3 and L4 both allocate from their own per-level counter starting at 1,
  a tensor produced at L3 (handle=1) and the output tensor at L4 (also handle=1) would
  collide in L4's TensorRegistry, causing the scheduler to skip registering the L3
  dependency and the DAG edge to be silently lost.  Using a single process-wide atomic
  counter guarantees uniqueness without coordination overhead.

- **Mutex scopes are kept short**: the scheduler collects tasks to dispatch under
  `pending_mu_`, then releases the lock before calling `enqueue_ready`, to avoid
  holding two locks simultaneously.

### 7.4 Three-Tier Communication Architecture

The Linqu hierarchy has **three fundamentally different communication tiers**, each with distinct memory models, synchronization mechanisms, and performance characteristics:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Tier 3: Message Passing (L4–L6 ↔ L3)                              │
│  Unix Socket / TCP / RDMA                                           │
│  Serialized messages: CALL_TASK, TASK_COMPLETE, SHUTDOWN            │
│  Each node: independent process, independent address space          │
│  No shared state — all coordination via explicit messages           │
├─────────────────────────────────────────────────────────────────────┤
│  Tier 2: Host-Device DMA (L3 ↔ L2)                                 │
│  h2d_copy / d2h_copy                                                │
│  L3 orchestrator runs on Host CPU (host memory)                     │
│  L2 scheduler + workers run on Device (device GM)                   │
│  Data transfer: explicit DMA between host memory and device GM      │
│  Control: simpler's shared memory header in device GM,              │
│           orchestrator writes via MMIO / DMA; scheduler reads       │
├─────────────────────────────────────────────────────────────────────┤
│  Tier 1: Shared Device GM (L0–L2, intra-chip)                      │
│  Atomic operations + memory barriers (LOAD_ACQUIRE / STORE_RELEASE) │
│  Orchestrator + Scheduler + Workers share same GM address space     │
│  Zero-copy: pointer passing via ring buffers                        │
│  CAS spinlocks for concurrent fanout list updates                   │
└─────────────────────────────────────────────────────────────────────┘
```

#### Tier 1: Shared Device GM (L0–L2, managed by `simpler`)

Within a single chip, the `simpler` runtime operates entirely in **device Global Memory (GM)**. The orchestrator, scheduler, and all worker cores share the same physical address space:

- **TaskRing, HeapRing, DepListPool** are allocated in GM and accessed by all components via direct pointer dereference.
- **Synchronization** uses hardware atomic operations: `__atomic_compare_exchange_n` (CAS) for fanout locks, `__atomic_fetch_add` for refcount updates, `STORE_RELEASE` / `LOAD_ACQUIRE` for ring pointer visibility.
- **Output buffers** are allocated from the HeapRing and their **real GM addresses** are written back into `Tensor.buffer.addr`. Workers read input data directly from GM pointers — no copies.
- The scheduler reads `current_task_index` via `LOAD_ACQUIRE` to detect newly submitted tasks, and writes `last_task_alive` via `STORE_RELEASE` for the orchestrator to detect freed ring slots.

This tier has the lowest latency and highest throughput. It is the domain of `simpler` and is **not modified** by the Linqu project.

#### Tier 2: Host-Device DMA (L3 ↔ L2)

The L3 (Host) level runs on the **Host CPU** with **host memory**, while L2 (Chip) operates in **device GM**. These are **different memory domains** — the host CPU cannot directly dereference device GM pointers, and vice versa. Data crosses this boundary via explicit DMA operations:

- **`h2d_copy` (host-to-device):** The L3 orchestrator prepares input tensors in host memory, then issues a DMA transfer to copy them into device GM before dispatching a chip-level task. The `simpler` runtime at L2 sees the data at the device GM address.
- **`d2h_copy` (device-to-host):** When a chip-level task produces output in device GM, the L3 orchestrator issues a DMA transfer to copy results back to host memory for further processing or forwarding to L4.
- **Control path:** The `simpler` orchestrator state (ring pointers, task descriptors) resides in device GM. The L3 host-side code writes to it via memory-mapped I/O (MMIO) or DMA, not via direct pointer access. The `PTO2SharedMemoryHeader` provides the interface for this cross-domain control flow.

This means L3's NodeDaemon acts as a **bridge**:
1. Receives `CALL_TASK` from L4 via Unix socket (Tier 3 message).
2. Prepares parameters in host memory.
3. Calls `h2d_copy` to push input data to device GM.
4. Invokes `simpler`'s orchestration API to submit the task to the chip.
5. Waits for chip-level completion.
6. Calls `d2h_copy` to pull results back to host memory.
7. Sends `TASK_COMPLETE` back to L4 via Unix socket (Tier 3 message).

#### Tier 3: Message Passing (L4–L6 ↔ L3, inter-process)

Above L3, every hierarchy level runs as an **independent OS process** with its **own address space**. There is no shared memory between L4 and L3, between L5 and L4, or between L6 and L5. All coordination is via explicit messages over IPC (Unix domain sockets in verification) or network (TCP/RDMA in production):

- **CALL_TASK:** Serialized message containing kernel name, tensor handles, and scalar parameters. The receiving process creates its own `LinquOrchestratorState` and executes the kernel in its own address space.
- **TASK_COMPLETE:** Serialized completion status sent back to the dispatching level.
- **Tensor handles are opaque integers**, not memory addresses. A handle at L4 refers to a logical tensor; the L3 process maps it to an actual host memory buffer independently.

This is the domain of the Linqu distributed runtime (`LinquOrchestratorState`, `RemoteDispatcher`, `NodeDaemon`).

#### Communication Model Summary Table

| Level boundary | Tier | Memory model | Transport | Synchronization | Data transfer |
|----------------|------|-------------|-----------|-----------------|---------------|
| L0 intra-core-group | 1 | Shared GM | TPUSH/TPOP DMA | Hardware ring | None (on-chip) |
| L0–L2 intra-chip | 1 | Shared GM | Direct pointer | Atomics + barriers | Zero-copy via GM pointers |
| **L2–L3 chip↔host** | **2** | **Host mem ↔ Device GM** | **h2d_copy / d2h_copy** | **DMA completion + MMIO** | **Explicit DMA** |
| L3–L4 host↔pod | 3 | Independent processes | Unix socket / TCP | Message acknowledgment | Serialized payload |
| L4–L5 pod↔supernode | 3 | Independent processes | TCP / RDMA | Message acknowledgment | Serialized payload |
| L5–L6 supernode↔cluster | 3 | Independent processes | TCP | Message acknowledgment | Serialized + compressed |

#### Implications for Runtime Design

1. **Linqu's `LinquOrchestratorState` (L3–L6) does NOT need atomic operations or shared-memory synchronization.** Its `fanout_refcount`, `fanin_refcount`, and task state are local variables in a single process. The `std::mutex` used for the scheduler is for thread safety (recv thread vs. main thread), not for cross-process coordination.

2. **Linqu's TensorMap uses opaque handles, not GM addresses.** This is correct by design — at L3–L6, there is no shared buffer space, so multi-dimensional overlap detection (which operates on physical memory addresses) is unnecessary. Handle-based exact-match tracking is the right approach.

3. **The L3 NodeDaemon is the critical bridge layer.** It must translate between Tier 3 (message-based) and Tier 2 (DMA-based) communication. Future Phase 1 implementation will integrate `simpler`'s `PTO2OrchestratorState` into the L3 daemon with proper `h2d_copy` / `d2h_copy` calls.

4. **Back-pressure at L3–L6 is message-driven**, not shared-memory-driven. When an L4 orchestrator's ring is full, it spins waiting for `TASK_COMPLETE` messages from L3 (which trigger `on_task_complete` → `propagate_completion` → `check_task_consumed` → `try_advance_ring_pointers`). This is fundamentally different from `simpler`'s approach where the orchestrator spins on `LOAD_ACQUIRE(last_task_alive)` in shared memory.

---

## 8. Cluster Discovery and Topology

### 8.1 Rule-Based IP-to-Coordinate Mapping

Linqu network hierarchy boundaries are **not** defined by subnet boundaries. Instead, IP addresses are **pre-planned and assigned** according to the hierarchy. The runtime uses a **user-defined rule formula** to compute the index at each level from an IP address.

Example schema for IPv4 `A.B.C.D`:

| Level | Name | IP segment | Formula |
|-------|------|------------|---------|
| 6 | Cluster-level-2 | (implicit or from A) | `idx_L6 = f(A)` |
| 5 | Cluster-level-1 | Octet B | `idx_L5 = B` |
| 4 | Cluster-level-0 | Octet C | `idx_L4 = C` |
| 3 | Host | Octet D | `idx_L3 = D` |

The mapping function is provided by the user at deployment time:

```python
def get_my_coordinates(ip_address: str) -> dict:
    parts = [int(x) for x in ip_address.split('.')]
    return {
        "level_6": 0,          # single L2 cluster in this deployment
        "level_5": parts[1],
        "level_4": parts[2],
        "level_3": parts[3],
    }
```

Every node can compute its own coordinates — and the coordinates of any other node — using this function. No central topology server is required.

### 8.2 Deterministic Peer Mapping

Because the IP-to-coordinate function is known to all nodes, peer addressing is **mathematical**:

- To find the "next host" in the same pod: `neighbor_ip = f"10.{my_L5}.{my_L4}.{my_L3 + 1}"`
- To find all hosts in pod 5 of supernode 1: enumerate `10.1.5.*`

The runtime does not need DNS or a name service for address resolution.

### 8.3 Zero-Configuration Rendezvous

Even though coordinates are deterministic, the runtime needs to know which nodes are **currently alive**. This is solved with a **decentralized gossip-based membership protocol** (e.g. SWIM):

**Phase A: Bootstrap ("First Citizen")**

By convention, the first valid IP in each Level-4 subnet is the **implicit rendezvous seed** for that scope:
- Node `10.1.5.20` knows its Level-4 seed is `10.1.5.1`.
- On boot, the node probes `10.1.5.1`. If `.1` is down, it tries `.2`, `.3`, etc.
- Once any active peer responds, they swap membership lists.

**Phase B: Gossip Protocol**

Once bootstrapped, nodes use periodic gossip to maintain the membership table:
- Every ~1 second, a node picks 2–3 random peers and exchanges its "Node List" for the same `linqu_logical_system_name`.
- Information about new or failed nodes spreads exponentially.
- Even in a 1,000-node cluster, full convergence happens in seconds.

**Phase C: Failure Detection**

- If Node A hasn't received a gossip from Node B in `T_suspect` seconds, it marks B as "Suspect."
- After `T_dead` more seconds without response, it marks B as "Dead" and propagates this.
- The Orchestrator's `pl.at(Level.CLUSTER_0)` call automatically skips dead nodes.

### 8.4 The Rendezvous Ledger Structure

Each node maintains a local `PeerRegistry` — a thread-safe map:

```
Key:   (linqu_logical_system_name, L6_idx, L5_idx, L4_idx, L3_idx)
Value: {
    ip: str,
    physical_system_name: str,
    status: "live" | "suspect" | "dead",
    last_heartbeat: timestamp,
    resources: ResourceVector,  # e.g. {"chips": 8, "mem_gb": 512}
}
```

The Orchestrator queries this map when resolving `pl.at(level=...)`:
1. Filter entries by `linqu_logical_system_name`.
2. Filter by target level and index range.
3. Exclude "dead" entries.
4. Return the list of live IPs for dispatch.

### 8.5 Hierarchical Aggregation for Scale

In very large clusters (Level 6), flat gossip becomes expensive. The runtime uses **aggregation nodes**:

- **Level 4 Leader**: one designated node in each Cluster-level-0 acts as a sub-registrar. It maintains the full membership list for its pod and reports a summary to Level 5.
- **Level 5 Leader**: aggregates Level-4 summaries for its supernode, reports up to Level 6.

Queries are level-scoped: an orchestrator targeting Level 4 only queries the Level-4 leader, not every individual host.

---

## 9. The Efficient RPC Protocol

### 9.1 Design Principles

- **Zero-copy**: use FlatBuffers or Cap'n Proto so fields can be read directly from the network buffer without deserialization into new objects.
- **Handle-based**: code and data are registered once and referenced by hash/handle thereafter.
- **Hierarchy-aware**: every message carries the sender's Linqu coordinate so the receiver knows the message's position in the hierarchy.
- **Tiny metadata path**: the common case (`CALL_TASK`) is a small header + handles, not a full binary blob.

### 9.2 Common Message Header

Every RPC message begins with a fixed 24-byte header:

```c
struct LinquHeader {
    uint32_t magic;            // Protocol version identifier
    uint64_t logical_sys_id;   // Hash of linqu_logical_system_name
    uint8_t  l6_idx;           // Cluster-level-2 index
    uint8_t  l5_idx;           // Cluster-level-1 index
    uint8_t  l4_idx;           // Cluster-level-0 index
    uint8_t  l3_idx;           // Host index
    uint16_t msg_type;         // Message type enum
    uint32_t payload_size;     // Payload bytes following header
};
```

### 9.3 Message Types

| msg_type | Name | Direction | Payload | Frequency |
|----------|------|-----------|---------|-----------|
| 0x01 | `HEARTBEAT` | Node → Peers | `(phys_name, logic_name, resources, timestamp)` | Periodic (~1s) |
| 0x02 | `REG_CODE` | Orchestrator → Node | `(blob_hash, code_binary)` | Once per code version |
| 0x03 | `REG_DATA` | Orchestrator → Node | `(data_handle, buffer_bytes, level, scope_depth)` | Once per dataset |
| 0x04 | `CALL_TASK` | Orchestrator → Node | `(blob_hash, data_handle[], spmd_idx, scope_depth, task_id)` | Every invocation |
| 0x05 | `SCOPE_EXIT` | Orchestrator → Nodes | `(scope_depth)` | Every scope close |
| 0x06 | `RETRY_WITH_CODE` | Node → Orchestrator | `(blob_hash)` | On cache miss |
| 0x07 | `TASK_COMPLETE` | Node → Orchestrator | `(task_key, status)` | Every task completion |

### 9.4 The "Register Once, Invoke Many" Lifecycle

```
Step 1  SETUP:      Orchestrator calls rt.deploy_environment(cluster_id)
                    → Node daemons are already running; gossip membership converges.

Step 2  CODE PREP:  Orchestrator calls rt.register(my_kernel_blob)
                    → REG_CODE sent with full binary + blob_hash.
                    → Node stores in CodeCache[blob_hash] → FunctionPointer.

Step 3  DATA PREP:  h = rt.put(large_tensor, level=Level.HOST, scope_depth=d)
                    → REG_DATA sent with buffer bytes.
                    → Node stores in buffer_ring[L][d] at the given scope.
                    → Returns data_handle h.

Step 4  RUN:        rt.call(my_kernel_blob, h, spmd_idx=i)
                    → CALL_TASK sent: only (blob_hash, [h], i, d, task_id).
                    → Tiny message: ~32 bytes of metadata.
                    → Node looks up CodeCache and DataCache, executes locally.

Step 5  CLEANUP:    Orchestrator exits pl.scope() at depth d.
                    → SCOPE_EXIT(d) broadcast to all nodes.
                    → Nodes clear buffer_ring[L][d] and task_ring[L][d].
```

### 9.5 Lazy Code Loading

When a node receives `CALL_TASK` but doesn't have the `blob_hash` in its CodeCache (e.g. it joined the cluster after registration), it responds with `RETRY_WITH_CODE`. The Orchestrator then sends `REG_CODE` for that hash, and re-sends the `CALL_TASK`. This lazy loading pattern keeps initial startup fast and handles late-joining nodes gracefully.

### 9.6 Code Outlining Integration

The compiler provides each outlined function with:
- A **static content hash** of the code block.
- A **capture list** (the variables the code reads from the enclosing scope).

The Orchestrator checks `if remote_node.has_hash(block_hash): skip_send()`. This further reduces transfers for code that hasn't changed between invocations.

---

## 10. SPMD Dispatch: `pl.at()` at Cluster Level

### 10.1 Execution Semantics

When the Orchestrator encounters `pl.at(level=pl.Level.CLUSTER_0)`, it:

1. **Resolves** the target scope: queries the PeerRegistry for all live nodes in the current Cluster-level-0 group (filtered by `linqu_logical_system_name`).
2. **Registers** code and data if not already cached on target nodes.
3. **Broadcasts** `CALL_TASK` to each target node, assigning a unique `spmd_idx` to each.
4. **Waits** for all `TASK_COMPLETE` responses (or handles failures).

### 10.2 Optimization via `pl.at(level=..., optimization=...)`

At cluster level, the `optimization` parameter can control how the work is distributed:

- `optimization=pl.chunked_loop_optimizer`: the runtime (or compiler) splits a large iteration space across nodes in chunks, similar to how chunked-loop splitting works at core level.
- No `optimization`: the block is dispatched as a single function per node.

### 10.3 Hierarchical Dispatch

For multi-level dispatch (e.g. `pl.at(level=pl.Level.CLUSTER_1)` that spans a supernode):

1. The Orchestrator sends to **Level 5 Leaders** (one per Cluster-level-0).
2. Each Level 5 Leader re-dispatches to its Level 4 nodes.
3. Each Level 4 node executes the function locally.

This hierarchical fan-out reduces the Orchestrator's bandwidth requirements for large clusters.

---

## 11. Ring Buffer Hierarchy at Cluster Level

### 11.1 Per-Level, Per-Depth Rings

At every hierarchy level, the runtime maintains ring buffers indexed by scope depth. For cluster operation:

```
task_ring[HOST][0], task_ring[HOST][1], ...      (Host-local rings)
task_ring[CLUSTER_0][0], task_ring[CLUSTER_0][1], ...  (Pod-level rings)
task_ring[CLUSTER_1][0], task_ring[CLUSTER_1][1], ...  (Supernode-level rings)
```

Data allocated within a `pl.at(level=pl.Level.CLUSTER_0)` scope at depth `d` is placed in `buffer_ring[CLUSTER_0][d]`.

### 11.2 Distributed Scope Exit

When the Orchestrator exits a `pl.scope()` at cluster level:

1. It broadcasts `SCOPE_EXIT(d)` to all nodes in the scope.
2. Each node locally applies scope-exit semantics: iterates tasks in `task_ring[L][d]`, applies scope tokens (respecting `task_freed` flags), triggers retirement scan.
3. Each node clears its local `buffer_ring[L][d]` slots once tasks are retired.
4. The Orchestrator decrements its own `current_scope_depth`.

This is the **distributed analog** of single-chip scope exit: each node handles its local portion independently.

### 11.3 Cross-Level Data Transfer

When a tensor produced at Level 3 (Host) is consumed by a function at Level 0 (Core), the runtime performs a multi-level transfer that crosses the Tier 2 communication boundary (see §7.4):

1. **Host ring (L3, host memory):** tensor lives in `buffer_ring[HOST][d]`, allocated in host DRAM.
2. **h2d_copy (Tier 2 boundary):** the L3 NodeDaemon issues an `h2d_copy` to transfer the tensor from host memory to device GM. The chip-side runtime allocates space in `buffer_ring[CHIP][d']` within device GM.
3. **Chip (L2, device GM):** the `simpler` orchestrator sees the data at the device GM address and submits work to cores.
4. **Core (L0):** the InCore function reads from device GM — zero-copy within the chip (Tier 1).

For the reverse direction (L0 output → L3 consumption):
1. The core writes output to device GM via `buffer_ring[CHIP][d']`.
2. The L3 NodeDaemon issues a `d2h_copy` to transfer results from device GM back to host memory.
3. The tensor is now available in `buffer_ring[HOST][d]` for the L3 orchestrator or forwarding to L4.

The hierarchy labels on functions tell the runtime which transfers are needed. The L3 NodeDaemon is responsible for managing these `h2d_copy`/`d2h_copy` operations automatically based on the data flow direction.

---

## 12. Profiling and Ring Capacity Tuning

### 12.1 Per-Layer Metrics

The runtime must collect the following metrics for each ring layer (hierarchy level `L`, scope depth `d`):

| Metric | Description |
|--------|-------------|
| `task_ring_capacity[L][d]` | Total slots allocated |
| `task_ring_peak_used[L][d]` | Maximum slots simultaneously occupied |
| `task_ring_peak_occupancy_pct[L][d]` | `peak_used / capacity × 100` |
| `task_ring_block_count[L][d]` | Number of times allocation blocked (ring full) |
| `task_ring_block_time_us[L][d]` | Total microseconds spent blocked |
| `buffer_ring_capacity_bytes[L][d]` | Total bytes allocated |
| `buffer_ring_peak_used_bytes[L][d]` | Maximum bytes simultaneously occupied |
| `buffer_ring_peak_occupancy_pct[L][d]` | `peak_used / capacity × 100` |
| `buffer_ring_block_count[L][d]` | Number of times allocation blocked |
| `buffer_ring_block_time_us[L][d]` | Total microseconds spent blocked |
| `retire_scan_calls[L][d]` | Number of retirement scans triggered |
| `retire_scan_reclaimed_tasks[L][d]` | Tasks reclaimed by retirement |
| `retire_scan_reclaimed_bytes[L][d]` | Bytes reclaimed |

### 12.2 Global Rollout Metrics

- Total blocked time across all layers.
- Maximum concurrent active scope depth.
- Program-level peak memory (sum over all layers).
- Per-operation / per-scope attribution of blocking hotspots.

### 12.3 Profiling Interface

Runtime flags:
- `runtime.profile_ring = true`
- `runtime.profile_ring_detail_level = {basic | verbose}`

Output formats:
- Machine-readable JSON (for CI and auto-tuning).
- Human-readable summary table.

Example JSON output:

```json
{
  "run_id": "xxx",
  "layers": [
    {
      "level": "HOST",
      "depth": 0,
      "task_ring_capacity": 4096,
      "task_ring_peak_used": 3012,
      "task_ring_block_count": 0,
      "buffer_ring_capacity_bytes": 1073741824,
      "buffer_ring_peak_used_bytes": 812646400,
      "buffer_ring_block_count": 3,
      "buffer_ring_block_time_us": 14320
    }
  ],
  "global": {
    "total_block_time_us": 20111,
    "max_active_scope_depth": 4,
    "peak_total_buffer_bytes": 1879048192
  }
}
```

### 12.4 Capacity Tuning Workflow

1. Run representative workloads with profiling enabled.
2. Collect p95/p99 of peak usage per layer.
3. Set deployment ring capacity with safety margin:
   - `task_capacity[L][d] = ceil(p99_task_peak[L][d] × margin)` (margin: 1.1–1.3).
   - `buffer_capacity[L][d] = ceil(p99_buffer_peak[L][d] × margin)`.
4. Re-validate: target `block_count == 0` for latency-sensitive paths.
5. Repeat for different model shapes / batch sizes; store profiles as deployment presets.

If `block_count` remains non-zero after capacity increase, inspect whether:
- Logical lifetime is too long (missing `pl.free` opportunities).
- Retire ordering is suboptimal.
- Do not only increase ring sizes; fix the root cause.

---

## 13. CI Gating and Regression Policy

### 13.1 Gating Rules

| Severity | Condition |
|----------|-----------|
| **Hard fail** | `buffer_ring_block_count_total == 0` for latency-critical pipelines |
| **Hard fail** | `task_ring_block_count_total == 0` |
| **Soft fail / warning** | Any layer `peak_occupancy_pct > 95%` |
| **Soft fail / warning** | Global `total_block_time_us` regresses by > X% from baseline |
| **Trend guard** | p99 `peak_total_buffer_bytes` regression > Y% over 7-day window |

### 13.2 Baseline Strategy

- Store a versioned profiling baseline per (model_shape, batch_size, cluster_config).
- Compare PR results against nearest baseline preset.
- Require explicit approval tag for intentional capacity increases.

### 13.3 Regression Triage

1. If block count increases: inspect missing/late `pl.free` opportunities; inspect ring layer mapping for changed scope behavior.
2. If occupancy increases without blocks: evaluate whether shape mix changed; adjust preset capacities only after confirming no logic regressions.
3. If only one layer regresses: tune that layer first; avoid global over-provisioning.

---

## 14. Implementation Plan

The plan is structured around two realities:

1. **`simpler` owns Levels 0–2 and must not be modified.** The Linqu runtime adapts to `simpler`'s existing interfaces for chip-level and core-level execution.
2. **The first implementation** (Phase 0) runs on a **Level 3 (single-host) environment only**, but the software is designed so that **all data structures, APIs, protocols, and identity formats are forward-compatible with the full 7-layer system**. Subsequent phases activate additional levels as hardware becomes available.

### Phase 0: First Implementation — Level 3 (Single-Host), Building on `simpler`

**Hardware target:** One host (one OS instance), one or more chips. No multi-host cluster. No Level 4–6 network fabric.

**Guiding rule:** Do not modify `simpler`. Use `simpler`'s API to execute Level 0–2 functions. Build Linqu runtime for Levels 2–6 around it.

| Task | Description |
|------|-------------|
| 0.1 | **Document the `simpler` adaptation interface (read-only study).** Read `simpler`'s source to document the APIs that the future `ChipBackend` adapter will need to call. This is a read-only reference document — no code in `pypto_runtime_distributed` depends on or links `simpler` in Phase 0. |
| 0.2 | **Design the `ChipBackend` adapter interface (design only in Phase 0).** Define the Tier 2 bridge interface (see §7.4) that will translate Linqu runtime dispatch calls (`level=pl.Level.CHIP`) into `simpler` API calls via dynamic linking. The adapter will manage `h2d_copy` / `d2h_copy` DMA operations, map opaque tensor handles to device GM addresses, and invoke `simpler`'s orchestration API. In Phase 0, L3→L2 dispatch is **stubbed**. The adapter will be implemented in a future phase within `pypto_runtime_distributed` — it does NOT modify `simpler`. |
| 0.3 | Implement the `RingLayer` class managing `task_ring[L][d]` and `buffer_ring[L][d]` for **Levels 3–6**. Parameterized by hierarchy level `L` (3–6), but only allocate non-zero capacity for `L = 3` in this phase. Levels 4–6 have zero-capacity rings (data structures exist, just empty). For Levels 0–2, `simpler` manages its own rings — the Linqu runtime does **not** duplicate them. |
| 0.4 | Implement the `ScopeManager` for **host-level and above scopes** (Level 3+). Tracks `current_scope_depth` for Level 3+ scopes, handles `scope.enter` / `scope.exit` / `pl.free` signals at these levels. Chip-level scope depth (within a `simpler` orchestration function) is managed by `simpler` independently. |
| 0.5 | Implement the `TaskKey` identity model with **full coordinate** `(logical_system, L6..L0, scope_depth, task_id)`. The `L0/L2` portion maps to `simpler`'s internal task identity (via the future `ChipBackend` adapter). In this phase, L4–L6 are always zero, but the struct and all comparisons use the full key. |
| 0.6 | Implement the retirement scan for Level 3+ rings: layer-local, respects `task_freed` flag, advances `last_task_alive[L][d]`. Level 0–2 retirement is handled by `simpler`. |
| 0.7 | Implement the `LinquCoordinate` struct and `get_my_coordinates(ip)` interface. In this phase, the host populates L3 (and queries `simpler` for chip/core topology to fill L0/L2); L4–L6 default to zero. |
| 0.8 | Implement `linqu_physical_system_name` and `linqu_logical_system_name` node identity. Even on a single host, the naming is set so that multi-host deployment requires no identity-format change. |
| 0.9 | Define and implement the `LinquHeader` binary message format (24-byte header) with **all level fields** (`l6_idx`, `l5_idx`, `l4_idx`, `l3_idx`). In this phase, l4–l6 are zero, but the wire format is stable. |
| 0.10 | Implement the `PeerRegistry` data structure, keyed by `(logical_system, full_coordinate)`. In this phase, it contains only the local host; tested with **mock multi-level topologies** for forward compatibility. |
| 0.11 | Implement the `pl.at(level=...)` dispatch path. For `L ∈ {0, 1, 2}`: **stubbed** in Phase 0 (future: delegate to `simpler` via `ChipBackend` adapter, without modifying `simpler`). For `L = 3`: handle in the Linqu runtime (host-level coordination). For `L ∈ {4, 5, 6}`: raise `NotYetSupported` at **runtime**. |
| 0.12 | Implement host-level coordination: the Linqu runtime manages host-level scope and tracks inter-chip data dependencies. Actual chip dispatch is stubbed in Phase 0; a future `ChipBackend` adapter will enable multi-chip coordination via `simpler`'s ABI. |
| 0.13 | Implement per-layer ring profiling metrics for Level 3. The metrics framework is parameterized for Levels 3–6. Level 0–2 profiling is `simpler`'s responsibility. |
| 0.14 | Write **forward-compatibility tests**: unit tests that create `TaskKey` and `LinquCoordinate` objects with non-zero L4–L6 fields, verify serialization/deserialization, ring indexing, and PeerRegistry lookup with full 7-level coordinates. These tests run on the single-host environment using mock data. |
| 0.15 | Write **end-to-end multi-level tests**: tests that exercise Level 6 → Level 5 → Level 4 → Level 3 dispatch through the multi-process verification environment. L3→L2 dispatch is stubbed. |
| 0.16 | **(Future phase) Implement `h2d_copy` / `d2h_copy` integration in `ChipBackend`.** The `ChipBackend` adapter (within `pypto_runtime_distributed`) will manage the Tier 2 memory boundary by dynamically linking to `simpler`'s `libhost_runtime.so` — it does NOT modify `simpler`. Tasks: (a) allocate device GM buffers, (b) issue `h2d_copy` to transfer input tensors from host memory to device GM, (c) issue `d2h_copy` to transfer output tensors back, (d) map opaque tensor handles to device GM addresses. |
| 0.17 | **(Future phase) Write Tier 2 boundary tests**: unit tests verifying correct `h2d_copy` / `d2h_copy` behavior — data integrity, handle-to-address mapping, and buffer lifecycle. |
| 0.18 | Add **Lingqu DFS hierarchical reduction test** (L7→L3, 1024 L3 nodes): each L3 worker reads one DFS file containing 1024 random numbers, returns local sum to parent level, and the top-level orchestrator prints the global sum aggregated from all children. Use the L3-L7 role-separated thread model (`ORCHESTRATOR`/`WORKER`) and default thread knobs (`Lx_NUM_SCHEDULER_THREADS=1`, `Lx_NUM_WORKER_THREADS=4` in the test). |

### Phase 1: Multi-Die Chips (Level 1 Added Inside `simpler`)

**Hardware target:** Host with multi-die chips.

**Guiding rule:** Level 1 is implemented **inside `simpler`**, not in the Linqu runtime. The Linqu runtime only needs minor adapter updates.

| Task | Description |
|------|-------------|
| 1.1 | Update the `ChipBackend` adapter if `simpler`'s API surface changes for multi-die dispatch (e.g. new die-index parameter). The Linqu runtime's `get_my_coordinates` populates L1 from `simpler`'s topology query. |
| 1.2 | Verify that `TaskKey` and `LinquCoordinate` correctly carry the L1 index through all Linqu runtime code paths (should already work via forward-compatible design). |

### Phase 2: Multi-Host Pod (Activate Level 4)

**Hardware target:** Multiple hosts in a pod/server, connected by high-bandwidth network.

| Task | Description |
|------|-------------|
| 2.1 | Activate Level 4 ring buffers (`task_ring[4][d]`, `buffer_ring[4][d]`). |
| 2.2 | Activate gossip-based discovery: UDP Heartbeat, deterministic seeding (probe Level-4 seed IPs), SWIM membership. |
| 2.3 | Implement `REG_CODE` / `REG_DATA` / `CALL_TASK` / `SCOPE_EXIT` message handlers on the Node Daemon (network RPC using the `LinquHeader` format defined in Phase 0). |
| 2.4 | Build `CodeCache[blob_hash]` and `DataCache[data_handle]` on remote nodes. |
| 2.5 | Implement `RETRY_WITH_CODE` lazy loading fallback. |
| 2.6 | Implement Orchestrator-side `pl.at(level=pl.Level.CLUSTER_0)` resolution: PeerRegistry query → IP list → broadcast. Each remote node runs its own `simpler` for Level 0–2 execution. |
| 2.7 | Implement distributed `SCOPE_EXIT` broadcast and per-node ring retirement (Level 3+ rings on each node). |
| 2.8 | Implement SPMD fan-out: assign `spmd_idx` per target node; broadcast `CALL_TASK`. |
| 2.9 | Implement `TASK_COMPLETE` collection and synchronization barriers. |
| 2.10 | Implement failure detection: suspect → dead transition; gossip propagation. |

### Phase 3: Supernode (Activate Level 5)

**Hardware target:** Supernode with high-bandwidth domain across pods.

| Task | Description |
|------|-------------|
| 3.1 | Activate Level 5 ring buffers. |
| 3.2 | Implement hierarchical dispatch: Orchestrator → Level-5 Leaders → Level-4 Nodes. |
| 3.3 | Implement Level-4 aggregation nodes (sub-registrars reporting up to Level 5). |
| 3.4 | Implement `optimization` parameter handling at cluster level (e.g. chunked distribution of iteration spaces across pods). |

### Phase 4: Full Cluster (Activate Level 6)

**Hardware target:** Cross-rack cluster with contracted bandwidth.

| Task | Description |
|------|-------------|
| 4.1 | Activate Level 6 ring buffers. |
| 4.2 | Implement Level-5 aggregation nodes reporting up to Level 6. |
| 4.3 | Implement bandwidth-aware scheduling: prefer local dispatch at lower levels; use compression or staged transfers at Level 6. |
| 4.4 | Implement RDMA transport optimization for Level 3–4 (intra-pod, zero-copy network). |

### Phase 5: Profiling, CI, and Production Hardening

| Task | Description |
|------|-------------|
| 5.1 | Extend ring profiling to all activated levels (3–6); full JSON and human-readable output. Level 0–2 profiling is `simpler`'s responsibility. |
| 5.2 | Define CI gating rules and baseline comparison framework (per-level, Level 3+). |
| 5.3 | Build capacity auto-tuning recommendations from profiling data. |
| 5.4 | Stress test: deep nesting, mixed `pl.free` + scope exits, node failure mid-scope, late-joining nodes, cross-level data transfer through `ChipBackend` → `simpler`. |
| 5.5 | Compatibility adapters for `simpler` profiling data (if needed for unified cross-level views). |

---

## 15. References

- **`simpler` runtime (Level 0–2, do not modify):** `pypto_workspace/simpler/` — existing chip-level and core-level runtime.
- **Machine hierarchy and function grammar:** `machine_hierarchy_and_function_hierarchy.md`
- **Multi-level runtime ring stack and `pl.free`:** `multi_level_runtime_ring_and_pypto_free_api.md`
- **ExpandMixedKernel and InCoreFunctionGroup:** `HL_new_feature_Expand_Mixed_Kernel_and_call_spmd.md`
- **TPUSH/TPOP (intra-cluster core communication):** `HL_ptoisa_newfeature20260306_TPUSH_TPOP.md`
- **Tensor valid_shape and alignment:** `tensor_valid_shape.md`
- **Gemini conversation (distributed runtime design exploration):** `Gemini_conversation.md`
