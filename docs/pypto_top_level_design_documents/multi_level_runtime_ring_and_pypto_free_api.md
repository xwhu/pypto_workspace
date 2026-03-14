# PyPTO Feature Design: `pl.free`

## 1. Background

In current PyPTO, a tensor defined in a grammatical scope can be:

- produced by one or multiple operations/functions
- consumed by subsequent operations/functions

Runtime tracks producer/consumer relations automatically. For each task:

- `fanout`: list of consumers
- `fanout_count`: total number of consumers **plus one initial scope-exit token**
- `ref_count`: dynamic reference count (consumer references + scope token references)

At a given scope level, `fanout_count` starts from `1` (instead of `0`).
This extra initial count is reserved for scope-exit semantics.

During orchestration execution, program counter is sequential. Runtime cannot predict future consumer submissions beyond the current execution point, so fanout may still expand while the scope is active.

Today, task/output end-of-life is determined by both:

1. the orchestration has exited the scope where tensor is defined
2. `ref_count == fanout_count` (mandatory)

Condition (2) is always required. Condition (1) can be too conservative for tensors declared in high-level scopes, causing overly long lifetime and larger-than-optimal runtime ring size / memory footprint.


## 2. Problem Statement

For tensors defined in broad scopes:

- lifetime is tied to scope exit
- even when no future meaningful use is intended by programmer, runtime still keeps the tensor alive until scope end bookkeeping happens
- memory release is delayed, increasing peak ring size

We need a way for programmer to explicitly declare "this tensor's scope lifetime should end now", without leaving the actual source scope.


## 3. Proposed DSL Feature

Add a new DSL API:

```python
pl.free(tensor)
```

Meaning:

- explicitly marks scope-lifetime end for `tensor`
- does **not** bypass fanout safety
- only replaces the "wait until scope.exit()" part of current lifetime rule


## 4. Runtime API

Add runtime API:

```text
pto_rt.free(outbuf)
```

Behavior:

- applies the same scope-exit ref-count effect to the producer task of `outbuf`
- marks producer task as already freed-by-API
- guarantees `scope.exit()` will not apply the same increment again


## 5. Runtime State Extension

For each task slot, add:

- `task_freed: bool` (default `false`)

Existing fields used:

- `fanout_count`
- `ref_count`
- tensor map entry: `outbuf -> (task_scope_level, task_id)`
- scope task list for `scope.exit()` iteration

Initialization rule:

- `fanout_count` starts at `1` at task creation time in a scope.
- this initial `1` represents the mandatory scope-exit token.


## 6. Updated End-of-Life Semantics

A task/output buffer is releasable when both are true:

1. orchestrator has applied the scope token effect (via `pl.free(outbuf)` or `scope.exit()`)
2. `ref_count == fanout_count`

Condition (2) remains mandatory.


## 7. Operational Rules

### 7.1 `pto_rt.free(outbuf)`

1. Lookup `task_scope_level` and `task_id` from `outbuf` using tensor map.
2. If `task_freed == 1`, do nothing and return.
3. Set `task_freed = 1`.
4. Increment task `ref_count += 1`.
5. Return.

If called multiple times, function is idempotent (no double increment).


### 7.2 `scope.exit()`

At scope exit, iterate all tasks in current scope:

1. If `task_freed == 0`, do `ref_count += 1`.
2. If `task_freed == 1`, skip this task.
3. Re-check release condition (`ref_count == fanout_count`) and retire if satisfied.


## 8. Correctness Invariants

Must hold at all times:

1. No double scope token increment:
   - `pl.free(outbuf)` + `scope.exit()` still applies scope token exactly once.
2. Fanout safety unchanged:
   - task/output is never released before `ref_count == fanout_count`.
3. Idempotence:
   - repeated `pl.free(outbuf)` does not corrupt ref counts.
4. Compatibility:
   - programs without `pl.free` preserve current behavior.


## 9. Frontend Requirements

Frontend lowering should emit `pto_rt.free(outbuf)` at the DSL call site.

Validation constraints:

- argument must resolve to a valid output buffer tensor
- recommended: disallow use-after-free at DSL semantic check level (optional strict mode)


## 10. Example

```python
with pl.scope():
    a = op1(...)
    b = op2(a)
    pl.free(a)            # `a` is an outbuf of its producer task
    c = op3(b)
    # scope ends later
```

Effect:

- producer task of `a` gets the scope token early via `free`.
- release occurs when `ref_count == fanout_count`.
- memory peak is reduced for long high-level scopes.


## 11. Test Plan

1. Baseline compatibility:
   - existing models without `pl.free` produce identical behavior.
2. Single call:
   - one tensor with explicit `free`, verify earlier release opportunity.
3. Multiple calls on same tensor:
   - no double increment, no crash.
4. Fanout pending:
   - call `free` before all consumers execute; verify no early free.
5. Scope-exit interaction:
   - task marked by `free` must not be incremented again at `scope.exit`.
6. Memory footprint:
   - compare peak ring size before/after on representative orchestration workloads.


## 12. Rollout Notes

- Feature gate can be added for staged rollout.
- Runtime debug log should print:
  - tensor id
  - `task_freed` transition
  - release decision (blocked by fanout vs released)


## 13. Runtime Major Upgrade: Multi-Layer Ring Stack

### 13.1 Current Limitation of Single Global Ring

Current `pto_rt` uses one global ring buffer to manage:

- task slots
- output buffer slots

`last_task_alive` only advances when:

1. task is out of orchestration scope
2. all output tensors satisfy `ref_count == fanout_count`

This is safe but can be suboptimal. A tensor/task in top-level scope can hold
the global head and block retirement of many inner-scope tensors/tasks that are
already end-of-life.


### 13.2 Target Architecture

Replace single global ring with a **multi-layer ring stack** indexed by scope depth.

When orchestration enters a deeper scope:

- runtime switches allocation target to next ring layer
- task slots and output buffers for that scope are allocated in that layer

When orchestration exits that scope:

- retirement and reclamation happen at that layer first
- outer layer no longer blocks inner-layer reuse


### 13.3 Core Model

Define for each scope depth `d`:

- `task_ring[d]`
- `buffer_ring[d]`
- `last_task_alive[d]`
- `head/tail` pointers per ring

Global runtime state adds:

- `current_scope_depth`
- scope stack with frame metadata:
  - owned tasks
  - owned tensors
  - ring layer id

Task identity upgrade:

- current runtime uses a single `task_id`
- new runtime must use dual tag: `(scope_level, task_id)`
- this dual tag is the canonical identifier for task lookup and references


### 13.4 Allocation Rules

1. On `scope.enter`:
   - `current_scope_depth += 1`
   - bind current allocations to ring layer `d = current_scope_depth`
2. On task creation in layer `d`:
   - allocate task slot from `task_ring[d]`
   - allocate outputs from `buffer_ring[d]`
   - assign task key as `(d, task_id_in_layer)`
   - register tensor producer/consumer metadata as today
3. On `scope.exit`:
   - mark tasks/tensors in this frame as out-of-scope
   - trigger layer-local retire scan for `d`
   - then `current_scope_depth -= 1`


### 13.5 Retirement Rules

For each layer `d`, task/output is reclaimable only if both remain true:

1. scope token has been applied (`scope.exit` or `pl.free`)
2. `ref_count == fanout_count`

If satisfied, advance `last_task_alive[d]` and reclaim corresponding ring slots
in `task_ring[d]` and `buffer_ring[d]`.

Important:

- reclaim decision is **layer-local**
- inner layer progress does not wait on outer layer head movement
- task matching for retire scan uses `(scope_level, task_id)`; never `task_id` alone


### 13.5.1 Producer/Consumer Metadata Upgrade

All references that currently store only `task_id` must migrate to dual tag.

Required metadata changes:

1. Tensor producer record:
   - from: `producer_task_id`
   - to: `producer_task_key = (producer_scope_level, producer_task_id)`
2. Tensor consumer record list:
   - from: `consumer_task_id[]`
   - to: `consumer_task_key[] = (consumer_scope_level, consumer_task_id)[]`
3. Tensor map:
   - `outbuf -> (task_scope_level, task_id)` (already required by `free(outbuf)`)
4. Runtime debug/profile event fields:
   - include `scope_level` and `task_id` together


### 13.6 Interaction with `pl.free`

`pl.free` remains a per-output-buffer early scope-token tool.

With ring stack:

- `pl.free` can unlock reclamation earlier **inside current layer**
- scope exit still performs final sweep
- `task_freed` rule prevents double increment

So the two optimizations are complementary:

- `pl.free` reduces logical lifetime
- ring stack removes cross-scope structural blocking


### 13.7 Compatibility and Migration

Suggested rollout:

1. Feature flag:
   - `runtime.multi_layer_ring = false` (default, legacy behavior)
   - `true` enables ring-stack path
2. Keep producer/consumer semantics, with explicit `fanout_count` initial `+1` token rule.
3. Introduce `TaskKey(scope_level, task_id)` in runtime interfaces; keep legacy
   `task_id` only as a derived/debug field (non-unique globally).
4. Add migration adapters in serialization/log parsing if old traces only carry
   `task_id`.
5. Reuse existing correctness checks; extend them with `TaskKey` equality.


### 13.8 Invariants for Ring Stack

Must hold:

1. No early free:
   - never reclaim unless `ref_count == fanout_count`
2. No double scope-token effect:
   - `task_freed` + `scope.exit` remains exactly-once
3. Layer isolation:
   - `last_task_alive[d]` only controls layer `d`
4. Deterministic behavior:
   - same program order yields same reclamation decisions (given same inputs)
5. Task identity uniqueness:
   - `TaskKey(scope_level, task_id)` is unique at runtime; `task_id` alone is not
     used for correctness decisions.


### 13.9 Test Additions

Add tests beyond Section 11:

1. Nested-scope stress:
   - deep scope nesting with heavy inner allocations
   - verify inner layer can reclaim without outer-layer stall
2. Mixed free + scope exit:
   - explicit `pl.free` in inner scopes and natural scope exits
3. Peak memory comparison:
   - single-ring vs ring-stack on representative models
4. Slot reuse safety:
   - ensure reclaimed slots are not reused before dependency completion
5. Cross-scope same-task-id collision:
   - force same `task_id` values in different scope levels and verify no
     producer/consumer mismatch.


### 13.10 Runtime Profiling and Ring Capacity Tuning

To make multi-layer ring runtime effective in production, runtime must provide
strong profiling to measure peak usage of each ring level. Without this, ring
sizes are either over-provisioned (waste memory) or under-provisioned (stall
performance).

Required per-layer metrics (`d` = scope depth):

- `task_ring_capacity[d]`
- `task_ring_peak_used[d]`
- `task_ring_peak_occupancy_pct[d]`
- `task_ring_block_count[d]` (allocation blocked due to full ring)
- `task_ring_block_time_us[d]`
- `buffer_ring_capacity_bytes[d]`
- `buffer_ring_peak_used_bytes[d]`
- `buffer_ring_peak_occupancy_pct[d]`
- `buffer_ring_block_count[d]`
- `buffer_ring_block_time_us[d]`
- `retire_scan_calls[d]`
- `retire_scan_reclaimed_tasks[d]`
- `retire_scan_reclaimed_bytes[d]`

Global rollout metrics:

- total blocked time across all layers
- max concurrent active scope depth
- program-level peak memory (sum over layers)
- per-op / per-scope attribution of blocking hotspots

Profiling interface recommendations:

1. Runtime flags:
   - `runtime.profile_ring=true`
   - `runtime.profile_ring_detail_level={basic|verbose}`
2. Output format:
   - machine-readable JSON (for CI and auto-tuning)
   - human-readable summary table
3. Time windows:
   - full-run aggregate
   - optional per-iteration/per-step snapshots

Suggested JSON sketch:

```json
{
  "run_id": "xxx",
  "layers": [
    {
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

Capacity tuning workflow:

1. Run representative workload set with profiling enabled.
2. For each layer `d`, collect p95/p99 of peak usage.
3. Set deployment ring capacity with safety margin:
   - `task_capacity[d] = ceil(p99_task_peak[d] * margin)`
   - `buffer_capacity[d] = ceil(p99_buffer_peak[d] * margin)`
   - recommended `margin`: 1.1 ~ 1.3 depending on workload variance.
4. Re-validate blocking counters:
   - target `block_count == 0` for latency-sensitive paths.
5. Repeat for different model shapes / batch sizes and store profiles as
   deployment presets.

Operational note:

- If `block_count` remains non-zero after capacity increase, inspect whether
  logical lifetime is too long (missing `pl.free`) or retire ordering is
  suboptimal, instead of only increasing ring sizes.


### 13.11 CI Gating and Regression Policy

To make profiling actionable, integrate ring metrics into CI gates.

Recommended gating rules (configurable per workload class):

1. Hard fail (must pass):
   - `buffer_ring_block_count_total == 0` for latency-critical pipelines
   - `task_ring_block_count_total == 0`
2. Soft fail / warning:
   - any layer `peak_occupancy_pct > 95%`
   - global `total_block_time_us` regresses by > X% from baseline
3. Trend guard:
   - p99 `peak_total_buffer_bytes` regression > Y% across 7-day moving window

Suggested baseline strategy:

- Store a versioned profiling baseline per model shape + batch profile.
- Compare PR results against nearest baseline preset.
- Require explicit approval tag for intentional capacity increase.

Example CI verdict schema:

```json
{
  "workload": "decode_bs8_seq4k",
  "status": "fail",
  "checks": [
    {
      "name": "buffer_block_count_zero",
      "actual": 2,
      "expected": 0,
      "severity": "hard_fail"
    },
    {
      "name": "layer2_peak_occupancy_pct",
      "actual": 97.4,
      "threshold": 95.0,
      "severity": "warn"
    }
  ]
}
```

Regression triage guidance:

1. If block count increases:
   - inspect missing/late `pl.free` opportunities
   - inspect ring layer mapping for changed scope depth behavior
2. If occupancy increases without blocks:
   - evaluate whether shape mix changed
   - adjust preset capacities only after confirming no logic regressions
3. If only one layer regresses:
   - tune that layer first; avoid global over-provisioning.

