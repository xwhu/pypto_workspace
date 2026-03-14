# With Hierarchy: Hierarchical Model, Symbols, and Scope Grammar

This document describes the **Linqu system** as a conceptual hierarchical machine, the **hierarchical execution model** used in PyPTO, the **symbols** defined at each level, and the **grammar** used to specify scopes so that code is organized (and compiled) according to a chosen level.

---

## 1. The Linqu System: Conceptual Hierarchical Machine

The **Linqu system** is defined as a conceptual **hierarchical machine** : a higher-level hierarchy **encloses** several instances of the lower-level hierarchy; recursively, this forms a **logical machine**. Levels are defined **bottom-up**.


| Level | Name                   | Description                                                                                                                                                                                                                                                                                                                    |
| ----- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **0** | **Core or Core-group** | **Core:** single execution unit; types **AIV** (Vector), **AIC** (Cube). **Core-group:** a scheduling unit that groups multiple cores with local affinity and shared communication (e.g. TPUSH/TPOP). Example: **1 AIC + 2 AIV** — able to run a **function group** with 1 AIC kernel and 2 AIV kernels (InCoreFunctionGroup). |
| **1** | **Chip die**           | One die; contains multiple cores. May be **omitted** in some chip models (e.g. single-die chips).                                                                                                                                                                                                                              |
| **2** | **Chip**               | One chip; contains one or more dies (if level 1 is present) or directly multiple cores.                                                                                                                                                                                                                                        |
| **3** | **Host**               | A single OS instance; one or more chips; runs orchestration and submits work to device.                                                                                                                                                                                                                                        |
| **4** | **Cluster-level-0**    | First cluster tier; usually within a **single server or pod**; high bandwidth, tight coupling.                                                                                                                                                                                                                                 |
| **5** | **Cluster-level-1**    | Second cluster tier; usually within a **supernode**; high-bandwidth domain across nodes.                                                                                                                                                                                                                                       |
| **6** | **Cluster-level-2**    | Third cluster tier; typically connected with **contracted bandwidth**; wider-area or cross-rack.                                                                                                                                                                                                                               |
| **7** | **Global Coordinator** | Top of hierarchy; coordinates all cluster-level-2 instances. Entry point for a global program.                                                                                                                                                                                 |


```
                    ┌─────────────────────────────────────────────────────────┐
  Level 7            │  Global Coordinator (top-level entry point)               │
                    └─────────────────────────────────────────────────────────┘
                                         │ encloses several
                    ┌─────────────────────────────────────────────────────────┐
  Level 6            │  Cluster-level-2 (e.g. cross-rack, contracted bandwidth) │
                    └─────────────────────────────────────────────────────────┘
                                         │ encloses several
                    ┌─────────────────────────────────────────────────────────┐
  Level 5            │  Cluster-level-1 (e.g. supernode, high-bandwidth domain) │
                    └─────────────────────────────────────────────────────────┘
                                         │ encloses several
                    ┌─────────────────────────────────────────────────────────┐
  Level 4            │  Cluster-level-0 (e.g. single server / pod)              │
                    └─────────────────────────────────────────────────────────┘
                                         │ encloses several
                    ┌─────────────────────────────────────────────────────────┐
  Level 3            │  Host (single OS instance)                               │
                    └─────────────────────────────────────────────────────────┘
                                         │ encloses several
                    ┌─────────────────────────────────────────────────────────┐
  Level 2            │  Chip                                                │
                    └─────────────────────────────────────────────────────────┘
              (opt) │ encloses several (if level 1 present)
                    ┌─────────────────────────────────────────────────────────┐
  Level 1            │  Chip die (optional in some chip models)                │
                    └─────────────────────────────────────────────────────────┘
                                         │ encloses several
                    ┌─────────────────────────────────────────────────────────┐
  Level 0            │  Core (AIV | AIC)  or  Core-group (e.g. 1 AIC + 2 AIV)  │
                    └─────────────────────────────────────────────────────────┘
```

- **Level 0 choice:** A **core** is a single execution unit (AIV or AIC). A **core-group** is a set of cores that are co-scheduled and share local interconnect (e.g. one AIC + two AIV cores), and can run a **function group** (e.g. one AIC kernel + two AIV kernels) as a single scheduling unit.
- **Recursive enclosure:** Each level is a logical machine; it contains multiple instances of the level below (e.g. a host contains multiple chips; a chip contains multiple cores or dies).
- **Bandwidth / coupling:** As the level number increases (cluster-level-0 → 1 → 2), the domain typically grows and interconnect bandwidth is more constrained (e.g. within-pod → supernode → cross-rack).

---

## 2. PyPTO Execution Model (Mapping onto Linqu)

Execution in PyPTO is organized in a **multi-level hierarchy** that maps onto the Linqu machine. Each level has a distinct execution model, memory/lifetime rules, and compiler treatment.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Orchestration (runs on Host, Linqu level 3)                             │
│  - Sequential program counter; submits tasks; manages tensor lifetime   │
├─────────────────────────────────────────────────────────────────────────┤
│  Cluster (maps to Linqu cluster-level-0 or chip/die as appropriate)     │
│  - Unit of SPMD; e.g. one cluster = one core-group (1 AIC + 2 AIV) L0   │
│  - OutlineClusterScopes outlines code into cluster-level functions       │
├─────────────────────────────────────────────────────────────────────────┤
│  InCore (runs on a single Core, Linqu level 0)                          │
│  - Code on one core (AIC or AIV); OutlineIncoreScopes → InCore functions │
├─────────────────────────────────────────────────────────────────────────┤
│  InCore kernel (AIC / AIV) — pure kernel on one core                    │
│  - ExpandMixedKernel: pure AIC or AIV; InCoreFunctionGroup = co-scheduled│
└─────────────────────────────────────────────────────────────────────────┘
```

- **Orchestration** runs at **Host** (Linqu level 3): sequential program, creates tensors, submits tasks; tensor lifetime and ring-buffer reclamation are defined by **grammatical scope** at this level.
- **Cluster** in PyPTO is a scheduling unit that typically corresponds to a **chip or chip-die** (Linqu 1–2) or to **cluster-level-0** (Linqu 4), depending on the chip model; it is identified by an SPMD index (e.g. `spmd_idx`). Code outlined to this level becomes a **cluster-level function**.
- **InCore** runs at **Core** (Linqu level 0): one core (AIC or AIV). Outlining produces **InCore functions**, which may be **mixed** and are then split by **ExpandMixedKernel** into AIC and AIV kernels.
- **InCore kernel (AIC/AIV)** is the finest granularity: a **pure** kernel on one **core**. A **core-group** (e.g. 1 AIC + 2 AIV) runs an **InCoreFunctionGroup**: one AIC kernel + two AIV kernels, co-scheduled with TPUSH/TPOP communication.

### 2.1 Current runtime support and compiler hierarchy labels

**Current pypto runtime** implements execution for:

- **pl.Level 0** (Core / Core-group): InCore functions, AIC/AIV kernels, and InCoreFunctionGroups run on cores and core-groups.
- **pl.Level 2** (Chip): Orchestration that targets one chip (e.g. `FunctionType.Orchestration` / `pl.Level.CHIP`).

**Level 1** (Chip die) is **not** present on current chip models; the runtime does not schedule work at that level today.

**Compiler requirement:** The compiler **must** generate functions with a **hierarchy label** for each level of the hierarchy (0–6) when it outlines or assigns a function to a level. That label is attached to the function in the IR (or emitted metadata) and is **reserved for future multi-level runtime design**. When the runtime gains support for additional levels (e.g. Host, Cluster-level-0/1/2), it can dispatch and schedule using these labels without requiring a new codegen contract.

---

## 3. Symbols Defined at Each Hierarchy Level

### 3.1 Orchestration (Host)


| Symbol / Concept                                | Description                                                                                                                                              |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Scope depth** `d`                             | Integer depth of nested `pl.scope()` blocks. Used to index ring layers and task identity.                                                                |
| **Task key** `(scope_level, task_id)`           | Unique task identity. `task_id` is only unique within a scope layer.                                                                                     |
| **Tensor (logical)**                            | Named buffer produced/consumed by tasks; has producer task key, consumer list, `fanout_count`, `ref_count`.                                              |
| **Scope token**                                 | One implicit “scope-exit” reference per task: `fanout_count` includes this token so that lifetime ends when scope exits and `ref_count == fanout_count`. |
| **Ring layer** `task_ring[d]`, `buffer_ring[d]` | Per-scope-depth ring buffers for task slots and output buffers (multi-layer ring stack).                                                                 |
| `**last_task_alive[d]`**                        | Per-layer head; advancement is layer-local (inner scope retirement does not wait on outer).                                                              |
| `**pl.scope()**`                                | Grammatical scope block: entering increments `current_scope_depth`; exiting applies scope-exit semantics and decrements depth.                           |
| `**pl.free(tensor)**`                           | Early scope-lifetime end for one output buffer (applies scope token once; idempotent with `scope.exit()`).                                               |


### 3.2 Cluster


| Symbol / Concept           | Description                                                                                        |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| **Cluster**                | Physical unit of SPMD; identified by `spmd_idx` (or equivalent).                                   |
| **Cluster-level function** | Function outlined by **OutlineClusterScopes**; invoked once per cluster with cluster index.        |
| **Cluster scope**          | Lexical/IR scope that is lifted to a cluster function; loop(s) over clusters become SPMD dispatch. |


(Cluster scope is introduced by the compiler when outlining; the DSL may expose it via loop constructs that the compiler recognizes for cluster-level outlining.)

### 3.3 InCore (Core)


| Symbol / Concept          | Description                                                                                                                                                                                    |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **InCore function**       | Function with `type = InCore`; runs on a single core (AIC or AIV). May be **mixed** (orchestration + compute).                                                                                 |
| **InCore scope**          | Lexical/IR scope that is outlined into an InCore function (e.g. body of `pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer)` or body of an `@pl.function(type=InCore)`). |
| **Core type**             | `AIC` (Cube) or `AIV` (Vector); assigned after **ExpandMixedKernel** to split mixed InCore into pure kernels.                                                                                  |
| **Mixed InCore function** | InCore function containing both orchestration-style ops (e.g. `tensor.read`, `tensor.view`) and compute ops (e.g. `tensor.matmul`, `row_max`).                                                 |
| **IR coloring**           | WHITE (control/neutral), GREEN (AIV), RED (AIC); used to decide which ops go to AIC vs AIV and where TPUSH/TPOP are inserted.                                                                  |


### 3.4 InCore Kernel (AIC / AIV) and Function Group — Linqu Level 0 (Core / Core-group)


| Symbol / Concept                        | Description                                                                                                                                              |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AIC kernel**                          | InCore function that contains only Cube (RED) operations after expansion; runs on one **AIC** core (Linqu level 0).                                      |
| **AIV kernel**                          | InCore function that contains only Vector (GREEN) operations; may be parameterized by `AIV_IDX` (0 or 1); runs on one **AIV** core.                      |
| **Core-group**                          | Linqu level 0 unit: multiple cores with local affinity (e.g. **1 AIC + 2 AIV**); can run a **function group** as one scheduling unit.                    |
| **InCoreFunctionGroup**                 | IR node grouping one AIC kernel and one or two AIV kernels; co-scheduled on one **core-group** (e.g. 1 AIC + 2 AIV); share TPUSH/TPOP ring and metadata. |
| `**call_group(group, args...)`**        | Invocation of an InCoreFunctionGroup: runtime launches AIC once and AIV twice (AIV_IDX=0, 1) with shared args plus implicit `AIV_IDX`.                   |
| **TPUSH/TPOP**                          | ISA primitives for intra-cluster data movement between AIC and AIV (see TPUSH/TPOP design doc).                                                          |
| **Shared params / AIV implicit params** | Group-level: `shared_params` passed to both kernels; `aiv_implicit_params` (e.g. `AIV_IDX`) injected by runtime for AIV only.                            |


---

## 4. Grammar to Specify Scopes by Hierarchy Level

The following grammar describes how the programmer (or frontend) specifies **scopes** so that code is organized and compiled for a given level of the hierarchy.

### 4.1 Orchestration-level scope (lifetime and ring layer)

- **Syntax:** `with pl.scope():` …  
- **Semantics:**  
  - On entry: increment `current_scope_depth`; bind allocations to ring layer `d = current_scope_depth`.  
  - Tasks/tensors created inside use `(scope_level, task_id)` and ring layer `d`.  
  - On exit: apply scope-exit token to tasks in this frame; perform layer-local retirement; decrement `current_scope_depth`.
- **Optional:** `pl.free(tensor)` inside the scope to apply scope token early for that tensor’s producer task.

### 4.2 InCore-level scope (outline to InCore function)

Two ways to define code that shall be organized as an **InCore** function:

1. **Explicit InCore function**
  - **Syntax:** `@pl.function(type=pl.FunctionType.InCore)` on a function definition.  
  - **Semantics:** The function body is the InCore scope; it is compiled as a single InCore function (possibly mixed). Downstream passes (e.g. **ExpandMixedKernel**) may split it into AIC/AIV kernels and optionally wrap in an **InCoreFunctionGroup**.
2. **Automatic InCore outlining**
  - **Syntax:** `with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):` … (replaces former `pl.auto_incore()`).  
  - **Semantics:** The compiler applies chunked-loop splitting, interchange, and **OutlineIncoreScopes** to the body, producing one or more InCore functions (often mixed). Those can then be expanded by **ExpandMixedKernel**.

### 4.3 Cluster-level scope (outline to cluster function)

- Cluster scope is typically introduced by **OutlineClusterScopes** over IR that the compiler identifies as cluster-parallel (e.g. loops over cluster index or SPMD dimension).  
- The **grammar** for “this block shall be a cluster-level function” may be:  
  - An explicit decorator or context manager (e.g. `@pl.cluster` or `with pl.cluster_scope():`) if the DSL exposes it, or  
  - A structured loop pattern that the pass recognizes and outlines into a cluster function.
- **Semantics:** The outlined function is invoked per cluster (e.g. with `spmd_idx`); it may in turn call InCore functions or InCoreFunctionGroups.

### 4.4 InCore kernel level (AIC / AIV) and group

- **No direct DSL scope:** Programmers do not usually write “AIC scope” or “AIV scope” explicitly.  
- **Derived by compiler:** **ExpandMixedKernel** turns a **mixed InCore function** into:  
  - one **AIC kernel** (RED nodes),  
  - one **AIV kernel** (GREEN nodes),  
  - and an **InCoreFunctionGroup** that contains both and defines TPUSH/TPOP and shared metadata.
- **Invocation:** Orchestration or cluster code calls the **group** via `call_group(group, args...)`; the runtime schedules the AIC kernel and the AIV kernel(s) on the same cluster.

---

## 5. Unified Grammar for Function Hierarchy Declaration (Explicit and Implicit)

The PyPTO frontend today supports **two ways** of defining a function that executes at a given hierarchy level. The extension below preserves both and uses a **common grammar** applicable across **all** hierarchy levels.

### 5.1 Current state

**Explicit function hierarchy declaration:** the programmer declares the function and its execution level via a decorator and a **function type**:

- `@pl.function(type=pl.FunctionType.AIV)` — function is executed at the **AIV** (core) level.
- `@pl.function(type=pl.FunctionType.Orchestration)` — function is executed at the **Chip** level (orchestration that submits work to the chip).

Other types (e.g. `InCore`, `AIC`) may exist; the level is encoded in the **type** and is not a separate parameter.

**Implicit declaration of function boundary and hierarchy:** the programmer marks a **block**; the compiler infers a function boundary and its execution level from the construct:

- `with pl.incore:` — the block is treated as a single function to be executed at the **InCore** level (core / core-group); the compiler outlines it as one InCore function.
- `with pl.auto_incore:` — the block may be split and outlined into **one or more** InCore-level functions (e.g. after chunked-loop splitting and interchange).

So today: **explicit** = decorator + `FunctionType`; **implicit** = `pl.incore` / `pl.auto_incore` only for (In)Core level. There is no single parameter that denotes “hierarchy level” shared by both styles, and not all Linqu levels are expressible in the same way.

### 5.2 Extension goal

- **Preserve** both explicit declaration and implicit declaration.
- Introduce a **common grammar** so that the **same hierarchy level** can be specified in both styles and can be used **across all hierarchy levels** (Linqu 0–6: Core, Core-group, Chip die, Chip, Host, Cluster-level-0/1/2).

### 5.3 Common hierarchy-level parameter

Introduce a first-class notion of **hierarchy level** that aligns with the Linqu machine and can be used in both explicit and implicit forms. For example, a module-level constant or enum:


| Symbol                | Linqu level    | Meaning                                                  | Readability aliases                  |
| --------------------- | -------------- | -------------------------------------------------------- | ------------------------------------ |
| `pl.Level.AIV`        | 0 (Core)       | Single AIV core                                          | —                                    |
| `pl.Level.AIC`        | 0 (Core)       | Single AIC core                                          | —                                    |
| `pl.Level.CORE_GROUP` | 0 (Core-group) | One core-group (e.g. 1 AIC + 2 AIV); runs function group | —                                    |
| `pl.Level.CHIP_DIE`   | 1              | Chip die (optional in some models)                       | `pl.Level.L2CACHE`                   |
| `pl.Level.CHIP`       | 2              | Chip; orchestration that targets one chip                | `pl.Level.PROCESSOR`, `pl.Level.UMA` |
| `pl.Level.HOST`       | 3              | Host (one OS instance); orchestration                    | `pl.Level.NODE`                      |
| `pl.Level.CLUSTER_0`  | 4              | Cluster-level-0 (e.g. server / pod)                      | `pl.Level.POD`                       |
| `pl.Level.CLUSTER_1`  | 5              | Cluster-level-1 (e.g. supernode)                         | `pl.Level.CLOS1`                     |
| `pl.Level.CLUSTER_2`  | 6              | Cluster-level-2 (e.g. cross-rack)                        | `pl.Level.CLOS2`                     |
| `pl.Level.GLOBAL`     | 7              | Global coordinator (top of hierarchy)                     | —                                    |


**Aliases (same value as primary, for readability):** `pl.Level.L2CACHE` = `pl.Level.CHIP_DIE`; `pl.Level.PROCESSOR` = `pl.Level.UMA` = `pl.Level.CHIP`; `pl.Level.NODE` = `pl.Level.HOST`; `pl.Level.POD` = `pl.Level.CLUSTER_0`; `pl.Level.CLOS1` = `pl.Level.CLUSTER_1`; `pl.Level.CLOS2` = `pl.Level.CLUSTER_2`. Programs may use either the primary name or the alias.

(Concrete names can be chosen to match existing `FunctionType` where desired, e.g. `Orchestration` → `CHIP` or `HOST`.)

### 5.4 Explicit declaration (common grammar)

Use the **same level parameter** for all explicit function declarations:

```python
@pl.function(level=pl.Level.AIV)
def my_aiv_kernel(...): ...

@pl.function(level=pl.Level.CHIP)   # or level=pl.Level.HOST for host orchestration
def my_orchestration(...): ...

@pl.function(level=pl.Level.CLUSTER_0)
def my_cluster_kernel(...): ...
```

- **Semantics:** The decorated function is compiled and scheduled to run at the given **level**. The existing `type=pl.FunctionType.*` can be retained as a shorthand or deprecated in favor of `level=pl.Level.*` so that one grammar applies to every level.

### 5.5 Implicit declaration: `pl.at(level=..., optimization=...)`

Implicit function boundary and hierarchy use a **single** context manager, `**pl.at()`**, with optional `**optimization**` to control how the block is transformed before outlining:

```python
# One function at level L (no extra loop optimization)
with pl.at(level=pl.Level.AIV):
    ...

with pl.at(level=pl.Level.CORE_GROUP):
    ...

# Chunked-loop optimization then outline (like former pl.auto_incore)
with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
    # Compiler may split loops, interchange, then outline one or more functions at CORE_GROUP.
    ...

# Other optimizations (e.g. fully unroll static loops) then outline
with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.fully_unroll_static_loop):
    ...
```

**Parameters:**


| Parameter      | Meaning                                                                                                                                                                                                                                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `level`        | Hierarchy level (e.g. `pl.Level.CORE_GROUP`, `pl.Level.CLUSTER_0`). The block is outlined as function(s) at this level.                                                                                                                                                                                       |
| `optimization` | Optional. How to transform the block before outlining. If omitted, the block is outlined as **one** function. If set, the compiler applies the chosen strategy (e.g. chunked-loop split + interchange, or full unroll of static loops), then outlines the resulting IR into one or more functions at `level`. |


`**optimization` options (extensible):**


| Symbol                        | Meaning                                                                                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| *(none / default)*            | No extra transformation; block becomes one function at `level`.                                                                         |
| `pl.chunked_loop_optimizer`   | Apply chunked-loop splitting, interchange, and outlining (current **auto_incore** behavior); may produce multiple functions at `level`. |
| `pl.fully_unroll_static_loop` | Fully unroll loops with static bounds, then outline.                                                                                    |
| *(future)*                    | Other strategies (e.g. tile-and-outline, partial unroll) can be added.                                                                  |


**Semantics:**

- `**with pl.at(level=X):`** — one function at level X. Same as former “fixed boundary” or `with pl.incore` when `level=pl.Level.CORE_GROUP`.
- `**with pl.at(level=X, optimization=pl.chunked_loop_optimizer):**` — compiler may split and outline **one or more** functions at level X. Replaces `**pl.auto(level=X)`** and `**with pl.auto_incore**`.

**Backward compatibility:**

- `with pl.incore` → `with pl.at(level=pl.Level.CORE_GROUP)`.
- `with pl.auto_incore` → `with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer)`.

### 5.6 Summary: one level, two styles


| Style        | Syntax                                              | Meaning                                                                                                                                                                                                   |
| ------------ | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Explicit** | `@pl.function(level=pl.Level.X)`                    | One function; runs at hierarchy level X.                                                                                                                                                                  |
| **Implicit** | `with pl.at(level=pl.Level.X, optimization=...):` … | Block is transformed by optional `optimization`, then outlined as function(s) at level X. Omit `optimization` for one function; use e.g. `pl.chunked_loop_optimizer` for auto-split (former auto_incore). |


The **same** `pl.Level.*` is used for all hierarchy levels; implicit declaration is **unified** under `**pl.at(level=..., optimization=...)`** (replacing separate `pl.incore` / `pl.auto_incore` / `pl.auto`).

### 5.7 Function role: `role=pl.Role.{ORCHESTRATOR, WORKER}`

For L3–L7 CPU-side runtimes, the runtime distinguishes between two function roles within each hierarchy level:

| Role | Meaning |
|------|---------|
| `pl.Role.ORCHESTRATOR` | Builds DAG / task graph, submits child tasks (same-level workers or next-level orchestrators), manages tensor/task ring allocation. Never performs compute directly. |
| `pl.Role.WORKER` | Executes concrete compute or data tasks. Takes tensor inputs/outputs managed by the DAG scheduler. Never submits further tasks. |

Both explicit and implicit grammar forms accept the `role` parameter:

```python
# Explicit — decorator style
@pl.function(level=pl.Level.HOST, role=pl.Role.WORKER)
def dfs_reader(path: str, out: pl.Tensor):
    ...

@pl.function(level=pl.Level.POD, role=pl.Role.ORCHESTRATOR)
def pod_orchestrator(inputs: list[pl.Tensor]) -> pl.Tensor:
    ...

# Implicit — block style
with pl.at(level=pl.Level.HOST, role=pl.Role.WORKER):
    # block is outlined as a worker function at HOST level
    ...

with pl.at(level=pl.Level.POD, role=pl.Role.ORCHESTRATOR):
    # block is outlined as an orchestrator function at POD level
    ...
```

**Default behavior:** if `role` is omitted, the runtime treats the function as `ORCHESTRATOR` for backward compatibility with existing compiler outputs.

**Runtime semantics:**
- Workers are dispatched by the scheduler when all tensor-typed input dependencies are satisfied (`PENDING → READY`). Scalar arguments (coordinates, loop indices, file paths) are not tracked in the DAG.
- Orchestrators run on the orchestrator thread, build the DAG by calling `submit_worker()` and `submit_orchestrator()`, and may invoke `pl.tree_reduce()` to construct reduction DAGs.
- At each hierarchy level, the runtime maintains: 1 orchestrator thread, N scheduler threads, M worker threads — all in the same process sharing ring structures.

### 5.8 Built-in DAG patterns: `pl.tree_reduce`

The `pl.tree_reduce` utility constructs a binary tree reduction DAG from a list of leaf tensors:

```python
result = pl.tree_reduce(
    runtime,        # LevelRuntime instance
    leaves,         # list[pl.Tensor] — leaf tensors to reduce
    pair_fn,        # callable(a: pl.Tensor, b: pl.Tensor, out: pl.Tensor)
    name="reduce"   # task name for tracing
)
```

**Semantics:** submits all `(N−1)` internal-node workers upfront in a single sequential pass (no waiting between rounds). The tensor-map DAG automatically dispatches each internal-node worker when both of its inputs become ready. The orchestrator waits only once, on the root future.

This matches the tree reduction pattern described in `linqu_runtime_design.md` §7.3A-2a.

---

## 6. Summary Table: Scope Construct → Hierarchy Level


| Scope construct / IR concept                                      | PyPTO level                  | Linqu level                           | Compiler pass(es)                                                             |
| ----------------------------------------------------------------- | ---------------------------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| `pl.scope()`                                                      | Orchestration                | Host (3)                              | Runtime / frontend (scope depth, ring layer)                                  |
| `pl.free(tensor)`                                                 | Orchestration                | Host (3)                              | Frontend → `pto_rt.free(outbuf)`                                              |
| Cluster-outlined function                                         | Cluster                      | Cluster-level-0 (4) or chip/die (1–2) | OutlineClusterScopes                                                          |
| `pl.at(level=pl.Level.*)`                                         | any                          | Level (0–7)                           | One outlined function at given level                                          |
| `pl.at(level=pl.Level.*, optimization=pl.chunked_loop_optimizer)` | any                          | Level (0–7)                           | SplitChunkedLoops → … → Outline*Scopes (per level); replaces `pl.auto_incore` |
| `pl.at(level=pl.Level.*, role=pl.Role.*)`                         | any                          | Level (3–7)                           | Outlined function with explicit orchestrator/worker role                      |
| `@pl.function(type=InCore)` / `@pl.function(level=pl.Level.*)`    | InCore / any                 | Level (0–7)                           | Outline*Scopes (per level)                                                    |
| `@pl.function(level=pl.Level.*, role=pl.Role.*)`                   | any                          | Level (3–7)                           | Explicit role-annotated function at given level                               |
| `pl.tree_reduce(rt, leaves, pair_fn)`                              | DAG pattern                  | Level (3–7)                           | Binary tree reduction DAG built by orchestrator                               |
| InCoreFunctionGroup                                               | Function group on core-group | Core-group (0)                        | ExpandMixedKernel                                                             |
| AIC / AIV kernel                                                  | InCore kernel on one core    | Core (0)                              | ExpandMixedKernel (splitting + TPUSH/TPOP)                                    |


---

## 7. Example Programs

Two complete example programs demonstrate the DFS hierarchical sum algorithm using both grammar styles. Both express the same computation as `test_dfs_sum_hierarchy.cpp`:

- **`test_dfs_sum_hierarchy_pl_function.py`** — uses `@pl.function(level=..., role=...)` decorator style. Each orchestrator and worker is a named, decorated function.
- **`test_dfs_sum_hierarchy_pl_at.py`** — uses `with pl.at(level=..., role=...)` block style. Orchestrator and worker logic is defined inline within scoped blocks.

Both files are in `pypto_runtime_distributed/tests/unit/`.

---

## 8. References

- **Multi-level runtime and `pl.free`:** `multi_level_runtime_ring_and_pypto_free_api.md`
- **ExpandMixedKernel and InCoreFunctionGroup:** `HL_new_feature_Expand_Mixed_Kernel_and_call_spmd.md`
- **TPUSH/TPOP (intra-cluster):** `HL_ptoisa_newfeature20260306_TPUSH_TPOP.md`
- **Tensor shape and views:** `tensor_valid_shape.md`
- **Linqu runtime design:** `linqu_runtime_design.md`

