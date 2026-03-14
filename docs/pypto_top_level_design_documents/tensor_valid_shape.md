# Tensor `valid_shape`: Decoupling Storage Layout from Logical Data Extent

## Overview

This document specifies a new `valid_shape` member for pypto's internal tensor type, which allows sub-tensors to carry storage layouts that satisfy the ISA's 512-byte alignment requirement while accurately tracking which portion of that storage contains valid data.

## Background

### Tensor Storage and Sub-Tensor Extraction

In pypto, a program's input tensors can have arbitrary shapes that define their storage layout in global memory (DDR). Sub-tensors (views into the original tensor) are created using operations such as:

```python
chunk = pl.view(data, [1, 64], [i, 0])       # extract a [1, 64] view at offset [i, 0]
row   = pl.view(matrix, [1, 128], [pos, 0])  # extract a single row
```

These sub-tensors reference a contiguous region of the original tensor's storage, and all subsequent `pl.*` operations operate on them as independent tensor operands.

### ISA 512-Byte Alignment Constraint

The target hardware ISA (`pto-isa`) imposes a fundamental constraint:

> **Every tensor operand of a `pl.*` operation must have a total storage size that is a multiple of 512 bytes.**

For a tensor of shape `[R, C]` with element type `dtype`, the total storage size is:

```
storage_size = R × C × sizeof(dtype)
```

This value **must** satisfy:

```
storage_size % 512 == 0
```

This constraint applies to all tensor-level operations (`pl.add`, `pl.mul`, `pl.matmul`, `pl.view`, etc.) and propagates to the block-level operations they lower to (`block.load`, `block.store`, `block.add`, etc.). It originates from the hardware DMA engine and on-chip memory controllers, which transfer data in 512-byte blocks.

### The Tiling Problem

When implementing loop tiling (splitting a large computation into smaller chunks), the programmer or compiler divides a tensor along one or more axes. For example, tiling a `[4096, 5120]` tensor along the row axis with tile size 4:

```python
for p0 in pl.range(0, 4096, TOK_TILE):    # TOK_TILE = 4
    x_chunk = pl.view(hidden_states, [TOK_TILE, K_CHUNK], [b, p0, k0])
    # x_chunk shape: [4, 256], dtype: BF16
    # storage = 4 × 256 × 2 = 2048 bytes ✓ (multiple of 512)
```

This works when the tile dimensions evenly divide the full dimension. However, when the total extent is not a multiple of the tile size, the **last (tail) tile** contains fewer valid elements:

```python
# Example: tiling 100 elements with tile size 32
# Tiles: [0:32], [32:64], [64:96], [96:100]
#                                    ↑ tail tile: only 4 valid elements

for i in pl.range(0, 100, 32):
    chunk = pl.view(data, [32, 128], [i, 0])
    # Last iteration: i=96, but only 4 rows are valid (96..99)
    # If we use shape [4, 128] with BF16:
    #   storage = 4 × 128 × 2 = 1024 bytes ✓ (OK in this case)
    # But consider shape [4, 100] with BF16:
    #   storage = 4 × 100 × 2 = 800 bytes ✗ (NOT a multiple of 512)
```

The tail block often has a storage size that is **not** a multiple of 512 bytes, violating the ISA requirement. The programmer cannot simply shrink the shape to match the valid data extent without risking an alignment violation.

### Current Workaround

Today, pypto programs handle this by always using shapes that are 512-byte-aligned, even for tail tiles, and accepting that some padding rows/columns are "wasted." From the Qwen3 prefill example:

```python
# all pl.view / pl.slice of GM tensors use 512-B-aligned shapes
# (full TOK_TILE rows even on the tail tile; padding rows are harmless)
```

This works but has limitations:

1. **Correctness risk** — Operations may read or write garbage in padding regions. For accumulators this is harmless, but for operations like KV-cache writes, processing padding data corrupts state.
2. **Manual discipline** — Every programmer must manually ensure 512B alignment and mentally track which portion is valid.
3. **Compiler opacity** — The compiler has no way to distinguish valid data from padding, hindering optimizations (e.g., masking, predicated stores, dead-data elimination).

## Design: The `valid_shape` Member

### Concept

We introduce a new member `valid_shape` to the tensor internal type alongside the existing `shape`:

| Member | Meaning |
| ------ | ------- |
| `shape` | **Storage layout shape** — defines the physical storage extent. Must satisfy the 512B alignment constraint: `product(shape) × sizeof(dtype) % 512 == 0`. |
| `valid_shape` | **Logical data extent** — the portion of the storage that contains valid data. `valid_shape[i] <= shape[i]` for every axis `i`. |

The invariant is:

```
∀ i:  valid_shape[i] <= shape[i]
product(shape) × sizeof(dtype) % 512 == 0
```

When `valid_shape` is not explicitly set, it defaults to `shape` (i.e., the entire storage contains valid data — the common case).

### Visual Representation

```
                 shape[1] (columns)
        ┌──────────────────────────────┐
        │  valid data                  │ ← valid_shape[0] rows
        │  ┌────────────────────┐      │
        │  │                    │      │
        │  │   valid_shape      │ pad  │
        │  │   [R_v, C_v]       │      │
        │  │                    │      │
        │  └────────────────────┘      │
        │  padding rows                │ ← shape[0] - valid_shape[0]
        │                              │
        └──────────────────────────────┘
              storage layout: shape [R, C]
              storage_size = R × C × sizeof(dtype)   (must be 512B-aligned)
```

### Example

Consider tiling a `[100, 64]` FP32 tensor with tile size 32 along the row axis:

```python
for i in pl.range(0, 100, 32):
    valid_rows = pl.min(32, 100 - i)

    # Storage shape: always [32, 64] → 32 × 64 × 4 = 8192 bytes (✓ 512B-aligned)
    # Valid shape:   [valid_rows, 64]
    chunk = pl.view(data, shape=[32, 64], offset=[i, 0], valid_shape=[valid_rows, 64])
```

| Iteration | `i` | `valid_rows` | `shape` | `valid_shape` | Storage (bytes) | 512B? |
| --------- | --- | ------------ | ------- | ------------- | --------------- | ----- |
| 0 | 0 | 32 | [32, 64] | [32, 64] | 8192 | ✓ |
| 1 | 32 | 32 | [32, 64] | [32, 64] | 8192 | ✓ |
| 2 | 64 | 32 | [32, 64] | [32, 64] | 8192 | ✓ |
| 3 | 96 | 4 | [32, 64] | [4, 64] | 8192 | ✓ |

In iteration 3, the tensor has 32 rows of storage but only 4 rows of valid data. Operations can use `valid_shape` to:
- Only accumulate over valid rows
- Mask out padding in store operations
- Avoid polluting the KV cache with garbage

## API Design: Automatic Deduction with Manual Override

The central principle is: **`valid_shape` is automatically deduced by default, but can be explicitly overridden by the programmer.** If the programmer does not specify `valid_shape`, the compiler computes it from the source tensor's `valid_shape` and the operation's semantics. If the programmer provides an explicit `valid_shape`, it takes precedence.

This applies uniformly to all shape-manipulation operations: `pl.view`, `pl.slice`, `pl.reshape`, `pl.transpose`, and their `deep_` variants (`pl.deep_view`, `pl.deep_slice`, `pl.deep_reshape`, `pl.deep_transpose`).

### Unified API Pattern

Every shape-manipulation operation follows the same optional-override pattern:

```python
result = pl.op(tensor, ..., valid_shape=None)
#                            ^^^^^^^^^^^^^^
#   None  → automatic deduction from source tensor
#   [...]  → programmer-specified override
```

### Querying `valid_shape`

```python
vs = pl.valid_shape(tensor)   # returns the valid_shape as a tuple of Expr
s  = pl.shape(tensor)         # returns the storage shape (unchanged)
```

## Automatic Deduction Rules

### Notation

For any tensor `T`:
- `T.shape` = storage layout shape (512B-aligned)
- `T.vs` = valid_shape (logical data extent, `T.vs[i] <= T.shape[i]`)
- When `T.vs` has not been set, `T.vs = T.shape` (fully valid)

### Rule 1: `pl.view` / `pl.deep_view`

```python
result = pl.view(src, shape, offset)
result = pl.view(src, shape, offset, valid_shape=[...])  # manual override
```

`pl.view` extracts a sub-tensor starting at `offset` with storage layout `shape`. The output rank can differ from the input rank — the offset tuple always matches the *source* rank, and the output dimensions map to the trailing axes of the source.

**Deduction:** The valid extent within the view window is the intersection of the view region with the source's valid region, clamped per axis:

```
input_rank  = len(src.shape)
output_rank = len(shape)
axis_base   = input_rank - output_rank

for i in range(output_rank):
    src_axis  = axis_base + i
    remaining = src.vs[src_axis] - offset[src_axis]
    result.vs[i] = clamp(remaining, 0, shape[i])
```

**Example:**

```python
# src: shape=[BATCH, SEQ, HIDDEN], vs=[BATCH, 100, HIDDEN]
# (100 valid tokens out of SEQ padded rows)

chunk = pl.view(src, [4, 256], [b, 96, k0])
# axis_base = 3 - 2 = 1
# result.vs[0] = clamp(100 - 96, 0, 4) = 4     ← all 4 rows valid
# result.vs[1] = clamp(HIDDEN - k0, 0, 256) = 256
# → valid_shape = [4, 256]

chunk = pl.view(src, [4, 256], [b, 98, k0])
# result.vs[0] = clamp(100 - 98, 0, 4) = 2     ← only 2 of 4 rows valid
# result.vs[1] = clamp(HIDDEN - k0, 0, 256) = 256
# → valid_shape = [2, 256]
```

### Rule 2: `pl.slice` / `pl.deep_slice`

```python
result = pl.slice(src, starts, ends)
result = pl.slice(src, starts, ends, valid_shape=[...])  # manual override
```

`pl.slice` extracts the region `[starts[i] : ends[i]]` on each axis. The storage `shape` is chosen by the compiler to be 512B-aligned (by padding `ends - starts` up to the nearest 512B multiple). The valid extent is the *actual* slice range, which may be further limited by the source's valid extent.

**Deduction:**

```
for i in range(rank):
    slice_extent = ends[i] - starts[i]
    remaining    = src.vs[i] - starts[i]
    result.vs[i] = clamp(min(slice_extent, remaining), 0, result.shape[i])
```

**Example:**

```python
# src: shape=[128, 64], vs=[100, 64]

chunk = pl.slice(src, [96, 0], [128, 64])
# storage shape (512B-padded): [32, 64]
# slice_extent = [32, 64]
# remaining    = [100-96, 64-0] = [4, 64]
# result.vs    = [min(32,4), min(64,64)] = [4, 64]
```

### Rule 3: `pl.reshape` / `pl.deep_reshape`

```python
result = pl.reshape(src, new_shape)
result = pl.reshape(src, new_shape, valid_shape=[...])  # manual override
```

Reshape reinterprets the storage layout without changing the total element count. Valid data deduction under reshape is non-trivial because padding elements may be redistributed across the new axes.

**Deduction (two cases):**

**Case A — Source is fully valid** (`src.vs == src.shape`): Output is fully valid.

```
result.vs = new_shape
```

**Case B — Source has padding** (`src.vs != src.shape`): The deduction depends on whether the reshape only affects *fully-valid* axes.

An axis `i` is "fully valid" if `src.vs[i] == src.shape[i]`. If the reshape only merges/splits fully-valid axes while leaving the padded axis untouched, the valid_shape can be deduced:

```
# Example: src shape=[32, 64], vs=[4, 64]
#   axis 0 is padded, axis 1 is fully valid

# Allowed: reshape to [32, 8, 8] (splitting axis 1 only)
#   → result.vs = [4, 8, 8]

# Allowed: reshape to [2, 16, 64] (splitting axis 0 only, if 4 is divisible)
#   → result.vs = [2, 2, 64]  (preserving 4 valid out of 32 = 2 valid out of 16 per group of 2)

# Disallowed (ambiguous): reshape to [2048] (flattening padded + valid axes)
#   → compiler error: must provide explicit valid_shape
```

In the general case where padded and valid axes are merged, the compiler **requires an explicit `valid_shape`**:

```python
# Compiler cannot deduce: flattening a padded [32,64] into [2048]
flat = pl.reshape(src, [2048])                     # ERROR: ambiguous valid_shape
flat = pl.reshape(src, [2048], valid_shape=[256])   # OK: programmer provides it
```

**Special case for `deep_reshape` in chain splitting:** `deep_reshape` is copy-semantic and often used to reshape a reduction output like `[8, 1]` → `[1, 8]` (reshaping a fully-valid tensor). In this case, the deduction is trivially `result.vs = new_shape`.

### Rule 4: `pl.transpose` / `pl.deep_transpose`

```python
result = pl.transpose(src, axis1, axis2)
result = pl.transpose(src, axis1, axis2, valid_shape=[...])  # manual override
```

Transpose swaps two axes. The valid_shape follows the same permutation.

**Deduction:**

```
result.vs = copy(src.vs)
swap(result.vs[axis1], result.vs[axis2])
```

**Example:**

```python
# src: shape=[32, 64], vs=[4, 64]
result = pl.transpose(src, 0, 1)
# result.shape = [64, 32]
# result.vs    = [64, 4]
```

This rule is exact and always applies — no ambiguity.

### Rule 5: `pl.create_tensor`

```python
t = pl.create_tensor(shape, dtype=pl.FP32, valid_shape=None)
```

For newly created tensors, `valid_shape` defaults to `shape` (fully valid). The programmer can specify a smaller `valid_shape` if the tensor is intended as a padded buffer:

```python
buf = pl.create_tensor([32, 64], dtype=pl.FP32, valid_shape=[4, 64])
```

### Deduction Summary Table

| Operation | Automatic Deduction | Ambiguous Case |
| --------- | ------------------- | -------------- |
| `view` / `deep_view` | `clamp(src.vs[axis] - offset[axis], 0, shape[axis])` per axis | Never — always deducible |
| `slice` / `deep_slice` | `clamp(min(extent, src.vs[axis] - start[axis]), 0, shape[axis])` per axis | Never — always deducible |
| `transpose` / `deep_transpose` | Swap `vs` axes matching the transposition | Never — always deducible |
| `reshape` / `deep_reshape` | Copy `vs` when source is fully valid; split/merge fully-valid axes only | Merging padded + valid axes → requires explicit `valid_shape` |
| `create_tensor` | `valid_shape = shape` (fully valid) | N/A |

### Propagation Through Compute Operations

Once `valid_shape` is established on the inputs, it propagates through compute operations automatically:

| Compute Op | valid_shape Rule |
| ---------- | ---------------- |
| Element-wise (`add`, `mul`, `sub`, `div`, `exp`, ...) | `result.vs[i] = min(lhs.vs[i], rhs.vs[i])` |
| Scalar broadcast (`adds`, `muls`, ...) | `result.vs = src.vs` |
| `matmul(A, B)` | `result.vs = [A.vs[0], B.vs[1]]` (valid rows × valid cols) |
| `row_sum` / `row_max` | `result.vs = [src.vs[0], 1]` (only valid rows contribute) |
| `assemble(target, source, offset)` | `result.vs = target.vs` (target extent unchanged) |
| `cast` | `result.vs = src.vs` |

## Worked Example: Simplified Tiling Loop

With automatic deduction, the programmer writes simpler code — no manual `valid_shape` tracking:

**Before (current — manual discipline):**

```python
for i in pl.range(0, 100, 32):
    valid_rows = pl.min(32, 100 - i)
    chunk = pl.view(data, [32, 64], [i, 0], valid_shape=[valid_rows, 64])
```

**After (automatic deduction):**

```python
# data: shape=[128, 64], vs=[100, 64]  (100 valid rows in 128-row storage)

for i in pl.range(0, 100, 32):
    chunk = pl.view(data, [32, 64], [i, 0])
    # valid_shape automatically deduced:
    #   clamp(100 - i, 0, 32) → 32 for i=0,32,64; 4 for i=96
    # No manual tracking needed!
```

The key insight: the valid_shape of the *source tensor* already encodes the data extent. Each operation that manipulates shape propagates this information down the chain. Programmers only need to set `valid_shape` once at the boundaries (e.g., on program input tensors or after receiving data with a known valid extent), and all downstream operations inherit it automatically.

**Override when needed:**

```python
chunk = pl.view(data, [32, 64], [i, 0], valid_shape=[custom_rows, 64])
# Programmer overrides the automatic deduction — e.g., when they know
# only a subset of the mathematically-valid rows should be processed
```

## IR Representation

### TensorType Extension

The `TensorType` internal class gains a new optional field:

```
TensorType:
    shape_: list[Expr]           # storage layout shape (existing)
    dtype_: DataType             # element type (existing)
    memref_: MemRef | None       # memory reference (existing)
    tensor_view_: TensorView     # layout info (existing)
    valid_shape_: list[Expr]     # logical data extent (NEW, optional)
```

When `valid_shape_` is empty or absent, it is semantically equivalent to `shape_`.

### IR Printing

Tensors with a `valid_shape` different from `shape` are printed with an additional annotation:

```python
# valid_shape == shape (default, no annotation)
x: Tensor[[32, 64], FP32]

# valid_shape != shape
x: Tensor[[32, 64], FP32, valid_shape=[4, 64]]
```

### Shape-Manipulation IR Operations

All shape-manipulation operations gain an optional `valid_shapes` positional argument. When absent, the type-inference function applies the automatic deduction rules described above.

```python
# tensor.view — optional 4th argument
tensor.view(data, (i, 0), (32, 64))                            # auto-deduced
tensor.view(data, (i, 0), (32, 64), valid_shapes=(vr, 64))     # explicit override

# tensor.slice — optional 4th argument
tensor.slice(data, (0, 0), (32, 64))                           # auto-deduced
tensor.slice(data, (0, 0), (32, 64), valid_shapes=(vr, 64))    # explicit override

# tensor.reshape — optional 3rd argument
tensor.reshape(data, (2048,))                                   # auto-deduced (if unambiguous)
tensor.reshape(data, (2048,), valid_shapes=(256,))              # explicit override

# tensor.transpose — optional 4th argument
tensor.transpose(data, 0, 1)                                    # auto-deduced (axis swap)
tensor.transpose(data, 0, 1, valid_shapes=(64, 4))             # explicit override

# deep_* variants follow the same pattern
tensor.deep_view(data, (i, 0), (32, 64))                       # auto-deduced
tensor.deep_reshape(data, (1, 8))                               # auto-deduced
```

## Compiler Pass Impact

### Existing Passes

| Pass | Impact |
| ---- | ------ |
| **ConvertToSSA** | No change — `valid_shape` is part of the type, SSA conversion is type-transparent. |
| **FlattenCallExpr** | No change — operates on expression structure, not type internals. |
| **RunVerifier** | **Update** — add verification that `valid_shape[i] <= shape[i]` for all axes. Verify 512B alignment applies to `shape`, not `valid_shape`. |
| **InitMemRef** | **Update** — memory allocation size is based on `shape` (storage), not `valid_shape`. |
| **MemoryReuse** | No change — reuse decisions are based on MemRef sizes, which derive from `shape`. |
| **InsertSync** | No change — synchronization is per-operation, independent of valid extent. |
| **AllocateMemoryAddr** | No change — addresses are computed from MemRef sizes derived from `shape`. |

### ConvertTensorToBlockOps

This pass converts tensor-level operations to block-level (tile) operations. It is the primary consumer of `valid_shape`:

```python
# Input (tensor level) — valid_shape was auto-deduced during type inference:
chunk = tensor.view(data, (i, 0), (32, 64))
# chunk.type: Tensor[[32, 64], BF16, valid_shape=[4, 64]]   (auto-deduced)
result = tensor.add(chunk, other)

# Output (block level):
tile_chunk = block.load(data, (i, 0), (32, 64), valid_shapes=(4, 64))
tile_result = block.add(tile_chunk, tile_other)
# tile_result inherits valid_shape from operands
```

The `valid_shapes` argument is propagated through `block.load` to the hardware TLOAD instruction, enabling the DMA to load only the valid portion while keeping the storage footprint 512B-aligned.

### ExpandMixedKernel

This pass splits mixed InCore functions into AIC and AIV kernels. It converts `tensor.reshape` → `tensor.deep_reshape` and `tensor.view` → `tensor.deep_view` in the AIV kernel to break dependency chains. The `deep_*` variants carry the same `valid_shape` semantics and apply the same deduction rules as their shallow counterparts, ensuring `valid_shape` is preserved across chain boundaries.

## Codegen Impact

### PTO Codegen

The `valid_shape` maps to the `v_row` / `v_col` fields in the `pto.tile_buf` type:

```
%0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32,
    rows=32, cols=64,
    v_row=4, v_col=64,          ← valid_shape
    blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

### CCE Codegen

The `valid_shape` maps to the `valid_shape` parameter in TileView, which is already supported in the existing TileView structure:

```python
tile_view = ir.TileView()
tile_view.valid_shape = [ir.ConstInt(4, DataType.INT64, span),
                          ir.ConstInt(64, DataType.INT64, span)]
```

This existing TileView `valid_shape` field (used today for TileType) is now also available for TensorType through the new `valid_shape_` member.

## Usage Examples

### Example 1: Tiling with Automatic Deduction (No Manual Tracking)

The source tensor's `valid_shape` flows through all downstream operations without any programmer intervention:

```python
@pl.function
def tiled_add(
    # Input tensors: 128 storage rows, only 100 contain valid data
    a: pl.Tensor[[128, 64], pl.FP32, valid_shape=[100, 64]],
    b: pl.Tensor[[128, 64], pl.FP32, valid_shape=[100, 64]],
    out: pl.Out[pl.Tensor[[128, 64], pl.FP32, valid_shape=[100, 64]]],
) -> pl.Tensor[[128, 64], pl.FP32]:
    TILE_ROWS = 32

    for ti in pl.range(0, 128, TILE_ROWS):
        # No valid_shape argument — automatically deduced from a.vs and b.vs
        a_chunk = pl.view(a, [TILE_ROWS, 64], [ti, 0])
        b_chunk = pl.view(b, [TILE_ROWS, 64], [ti, 0])
        # ti=0:  a_chunk.vs = [clamp(100-0,  0, 32), 64] = [32, 64]  (full)
        # ti=32: a_chunk.vs = [clamp(100-32, 0, 32), 64] = [32, 64]  (full)
        # ti=64: a_chunk.vs = [clamp(100-64, 0, 32), 64] = [32, 64]  (full)
        # ti=96: a_chunk.vs = [clamp(100-96, 0, 32), 64] = [4, 64]   (tail!)

        result = pl.add(a_chunk, b_chunk)
        # result.vs = [min(a_chunk.vs[0], b_chunk.vs[0]), 64] — auto-propagated

        out = pl.assemble(out, result, [ti, 0])

    return out
```

### Example 2: Chained Operations — `view` → `reshape` → `transpose`

Deduction chains across multiple shape manipulations:

```python
# src: shape=[BATCH, SEQ, HIDDEN], vs=[BATCH, 100, HIDDEN]
# (100 valid tokens padded to SEQ)

# Step 1: view — extracts a token tile
tile = pl.view(src, [4, HIDDEN], [b, 96, 0])
# tile.vs = [clamp(100-96, 0, 4), HIDDEN] = [4, HIDDEN]  ← auto

# Step 2: reshape — split HIDDEN into heads (fully-valid axis only)
tile_heads = pl.reshape(tile, [4, NUM_HEADS, HEAD_DIM])
# axis 0 has padding (vs=4, shape=4 here, but in general could differ)
# axis 1,2 are a split of HIDDEN (fully valid) → deducible
# tile_heads.vs = [4, NUM_HEADS, HEAD_DIM]  ← auto

# Step 3: transpose — swap head and token axes
tile_t = pl.transpose(tile_heads, 0, 1)
# tile_t.vs = [NUM_HEADS, 4, HEAD_DIM]  ← auto (axis swap)
```

### Example 3: Manual Override for Ambiguous Reshape

When the compiler cannot deduce `valid_shape`, the programmer provides it:

```python
# src: shape=[32, 64], vs=[4, 64]
# Flatten into 1D — merges padded axis 0 with valid axis 1
flat = pl.reshape(src, [2048], valid_shape=[256])
# Programmer knows: 4 valid rows × 64 cols = 256 valid elements
# Without override, this would be a compile error
```

### Example 4: Deep Operations in Kernel Splitting

`deep_reshape` and `deep_view` are used by the ExpandMixedKernel pass to break chains. They follow the same deduction rules:

```python
# In AIV kernel after chain splitting:
# mi_0: shape=[8, 1], vs=[8, 1] (fully valid reduction output)

mi_flat = pl.deep_reshape(mi_0, [1, 8])
# mi_flat.vs = [1, 8]  ← auto (source fully valid → result fully valid)

global_max = pl.row_max(mi_flat)
# global_max.vs = [1, 1]  ← auto (row reduction preserves valid row count)
```

### Example 5: Attention with Auto-Deduced Score Masking

```python
# scores: shape=[1, SEQ_TILE], vs=[1, SEQ_TILE] (from matmul, fully valid storage)
# But we only want the first valid_len columns:

scores_valid = pl.view(scores, [1, SEQ_TILE], [0, 0])
# If scores.vs = [1, SEQ_TILE], then scores_valid.vs = [1, SEQ_TILE]

# To get a narrower valid extent, the programmer can override:
scores_valid = pl.view(scores, [1, SEQ_TILE], [0, 0], valid_shape=[1, valid_len])
# Only valid_len scores are considered meaningful
```

## Relationship to TileView `valid_shape`

The existing `TileView.valid_shape` field on `TileType` serves the same conceptual purpose at the tile (on-chip) level. The new `TensorType.valid_shape_` extends this concept to the tensor (DDR) level, creating a consistent valid-extent tracking throughout the memory hierarchy:

```
DDR Tensor (shape=[32,64], valid_shape=[4,64])
    ↓  block.load
On-chip Tile (shape=[32,64], tile_view.valid_shape=[4,64])
    ↓  block.add
On-chip Tile (shape=[32,64], tile_view.valid_shape=[4,64])
    ↓  block.store
DDR Tensor (shape=[32,64], valid_shape=[4,64])
```

## Summary

| Aspect | Before | After |
| ------ | ------ | ----- |
| Storage alignment | Manual: programmer picks 512B-aligned shapes | Same — `shape` must be 512B-aligned |
| Valid data tracking | Manual: programmer mentally tracks tail sizes | Automatic: deduced through shape-manipulation ops |
| Programmer burden | Must compute `valid_shape` at every view/slice/reshape | Set `valid_shape` once on source tensors; downstream ops auto-deduce |
| Tail tile correctness | Risk of processing/writing garbage | Compiler-aware masking via `valid_shape` |
| Compiler optimizations | Limited — no knowledge of valid extent | Enabled — predicated stores, dead-data elimination |
| API | `pl.view(t, shape, offset)` | `pl.view(t, shape, offset)` — same call, but now carries deduced `valid_shape` |
| Manual override | N/A | `pl.view(t, shape, offset, valid_shape=[...])` — opt-in when needed |
| `deep_*` variants | No valid_shape tracking | Same deduction rules as shallow counterparts |
| Ambiguous cases | Silent — garbage processed | Compile error: programmer must provide explicit `valid_shape` |
