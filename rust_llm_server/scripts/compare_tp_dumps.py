#!/usr/bin/env python3
"""Compare TP=1 vs TP=2 debug tensor dumps.

Reads binary tensor files written by DebugDumper (Rust) and compares them
element-by-element. Handles TP shard concatenation for column-sharded tensors
(Q/K/V/gate/up) and partial-sum reduction for row-sharded tensors (O-Proj/down).

Usage:
    # Compare TP=1 (single dir) vs TP=2 (two rank dirs):
    python scripts/compare_tp_dumps.py /tmp/tp1_dump /tmp/tp2_rank0 /tmp/tp2_rank1

    # Compare any two single dumps:
    python scripts/compare_tp_dumps.py /tmp/dump_a /tmp/dump_b
"""

import sys
import json
import struct
import os
from pathlib import Path

import numpy as np


# ── BF16 / FP16 helpers ────────────────────────────────────────────────

def bf16_to_fp32(raw: bytes) -> np.ndarray:
    """Convert raw BF16 bytes to float32 numpy array."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    # BF16 → FP32: shift left 16 bits to occupy upper 16 bits of float32
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def fp16_to_fp32(raw: bytes) -> np.ndarray:
    """Convert raw FP16 bytes to float32 numpy array."""
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


def load_tensor(bin_path: Path, dtype_str: str) -> np.ndarray:
    """Load a binary dump file and return as float32 array."""
    raw = bin_path.read_bytes()
    if dtype_str == "bfloat16":
        return bf16_to_fp32(raw)
    elif dtype_str == "float16":
        return fp16_to_fp32(raw)
    elif dtype_str == "float32":
        return np.frombuffer(raw, dtype=np.float32)
    elif dtype_str == "int32":
        return np.frombuffer(raw, dtype=np.int32).astype(np.float32)
    else:
        # Fallback: try BF16
        return bf16_to_fp32(raw)


def load_meta(dump_dir: Path) -> dict:
    """Load meta.json from a dump directory."""
    meta_path = dump_dir / "meta.json"
    if not meta_path.exists():
        print(f"WARNING: {meta_path} not found, using dtype=bfloat16 fallback")
        return {}
    with open(meta_path) as f:
        return json.load(f)


# ── Comparison metrics ─────────────────────────────────────────────────

def compare_tensors(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute comparison metrics between two float32 arrays."""
    # Handle shape mismatch
    if a.shape != b.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
        }

    diff = np.abs(a - b)

    # NaN / Inf detection
    a_nan = np.sum(np.isnan(a))
    b_nan = np.sum(np.isnan(b))
    a_inf = np.sum(np.isinf(a))
    b_inf = np.sum(np.isinf(b))

    # Mask out NaN/Inf for numeric comparisons
    valid = np.isfinite(a) & np.isfinite(b)
    n_valid = np.sum(valid)

    if n_valid == 0:
        return {
            "status": "ALL_NAN_INF",
            "a_nan": int(a_nan), "b_nan": int(b_nan),
            "a_inf": int(a_inf), "b_inf": int(b_inf),
        }

    valid_diff = diff[valid]
    valid_a = np.abs(a[valid])

    max_abs = float(np.max(valid_diff))
    mean_abs = float(np.mean(valid_diff))

    # Relative error (avoid div by zero)
    denom = np.maximum(valid_a, 1e-8)
    max_rel = float(np.max(valid_diff / denom))
    mean_rel = float(np.mean(valid_diff / denom))

    # Cosine similarity
    a_flat = a[valid]
    b_flat = b[valid]
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    cosine = float(dot / (norm_a * norm_b + 1e-12))

    return {
        "status": "OK",
        "numel": int(a.size),
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "max_rel_err": max_rel,
        "mean_rel_err": mean_rel,
        "cosine_sim": cosine,
        "a_nan": int(a_nan), "b_nan": int(b_nan),
        "a_inf": int(a_inf), "b_inf": int(b_inf),
    }


# ── Shard-aware tensor loading ─────────────────────────────────────────

# Steps that are column-sharded (concat along last dim to reconstruct full tensor)
COLUMN_SHARDED = {"02_q_proj", "03_k_proj", "04_v_proj",
                  "05_q_norm", "06_k_norm",
                  "07_q_rope", "08_k_rope", "09_attn_out",
                  "14_gate_proj", "15_up_proj", "16_silu_mul"}

# Steps that are row-sharded partial sums (element-wise sum to reconstruct)
ROW_SHARDED_SUM = {"10_o_proj", "17_down_proj"}

# Steps that should match exactly (replicated or post-AllReduce)
EXACT_MATCH = {"00_embedding", "01_input_ln", "11_o_proj_allreduce",
               "12_residual_attn", "13_post_attn_ln",
               "18_down_proj_allreduce", "19_residual_ffn"}


def extract_step_name(name: str) -> str:
    """Extract step suffix from dump name: 'layer0_02_q_proj' → '02_q_proj'"""
    parts = name.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return name


def reconstruct_tp2_tensor(
    step_name: str,
    r0_tensor: np.ndarray,
    r1_tensor: np.ndarray,
    r0_shape: list,
    r1_shape: list,
) -> np.ndarray:
    """Reconstruct the full TP=1 equivalent from two TP=2 rank tensors."""
    if step_name in COLUMN_SHARDED:
        # Concat along last dimension (each rank has half the heads/features)
        r0 = r0_tensor.reshape(r0_shape)
        r1 = r1_tensor.reshape(r1_shape)
        return np.concatenate([r0, r1], axis=-1).flatten()
    elif step_name in ROW_SHARDED_SUM:
        # Element-wise sum (each rank has partial matmul result)
        return r0_tensor + r1_tensor
    else:
        # Post-AllReduce or replicated — both ranks should have same data.
        # Return rank 0's data (comparison script will also check rank 1).
        return r0_tensor


# ── Main ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    tp1_dir = Path(sys.argv[1])
    tp2_rank0_dir = Path(sys.argv[2])
    tp2_rank1_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    is_two_rank = tp2_rank1_dir is not None

    tp1_meta = load_meta(tp1_dir)
    tp2r0_meta = load_meta(tp2_rank0_dir)
    tp2r1_meta = load_meta(tp2_rank1_dir) if is_two_rank else {}

    # Collect all checkpoint names from TP=1
    if tp1_meta:
        tp1_names = sorted(tp1_meta.keys())
    else:
        tp1_names = sorted([
            p.stem for p in tp1_dir.glob("*.bin")
        ])

    # Print header
    print()
    print(f"{'Checkpoint':<35} {'Status':<14} {'MaxAbsErr':>12} {'MeanAbsErr':>12} "
          f"{'MaxRelErr':>12} {'CosineSim':>12} {'A_NaN':>6} {'B_NaN':>6} {'A_Inf':>6} {'B_Inf':>6}")
    print("─" * 160)

    for name in tp1_names:
        step_name = extract_step_name(name)

        # Load TP=1 tensor
        tp1_bin = tp1_dir / f"{name}.bin"
        if not tp1_bin.exists():
            print(f"{name:<35} {'MISSING_TP1':<14}")
            continue

        tp1_dtype = "bfloat16"
        tp1_shape = None
        if name in tp1_meta:
            tp1_dtype = tp1_meta[name].get("dtype", "bfloat16")
            tp1_shape = tp1_meta[name].get("shape", None)
        tp1_tensor = load_tensor(tp1_bin, tp1_dtype)

        # Load TP=2 tensor(s)
        tp2r0_bin = tp2_rank0_dir / f"{name}.bin"
        if not tp2r0_bin.exists():
            # Try AllReduce-specific names that only exist in TP=2
            print(f"{name:<35} {'MISSING_TP2':<14}")
            continue

        tp2r0_dtype = "bfloat16"
        tp2r0_shape = None
        if name in tp2r0_meta:
            tp2r0_dtype = tp2r0_meta[name].get("dtype", "bfloat16")
            tp2r0_shape = tp2r0_meta[name].get("shape", None)
        tp2r0_tensor = load_tensor(tp2r0_bin, tp2r0_dtype)

        if is_two_rank:
            tp2r1_bin = tp2_rank1_dir / f"{name}.bin"
            if tp2r1_bin.exists():
                tp2r1_dtype = "bfloat16"
                tp2r1_shape = None
                if name in tp2r1_meta:
                    tp2r1_dtype = tp2r1_meta[name].get("dtype", "bfloat16")
                    tp2r1_shape = tp2r1_meta[name].get("shape", None)
                tp2r1_tensor = load_tensor(tp2r1_bin, tp2r1_dtype)

                # Reconstruct full tensor from two shards
                tp2_tensor = reconstruct_tp2_tensor(
                    step_name, tp2r0_tensor, tp2r1_tensor,
                    tp2r0_shape or [tp2r0_tensor.size],
                    tp2r1_shape or [tp2r1_tensor.size],
                )
            else:
                tp2_tensor = tp2r0_tensor
        else:
            tp2_tensor = tp2r0_tensor

        # Compare
        result = compare_tensors(tp1_tensor, tp2_tensor)
        status = result["status"]

        if status == "OK":
            max_abs = result["max_abs_err"]
            mean_abs = result["mean_abs_err"]
            max_rel = result["max_rel_err"]
            cosine = result["cosine_sim"]
            a_nan = result["a_nan"]
            b_nan = result["b_nan"]
            a_inf = result["a_inf"]
            b_inf = result["b_inf"]

            # Color coding
            flag = ""
            if max_abs > 1.0 or cosine < 0.99:
                flag = " ⚠️ LARGE"
            elif max_abs > 0.01:
                flag = " ⚡"

            print(f"{name:<35} {'OK' + flag:<14} {max_abs:>12.6f} {mean_abs:>12.6f} "
                  f"{max_rel:>12.6f} {cosine:>12.8f} {a_nan:>6} {b_nan:>6} {a_inf:>6} {b_inf:>6}")
        elif status == "SHAPE_MISMATCH":
            print(f"{name:<35} {'SHAPE_DIFF':<14} a={result['a_shape']} b={result['b_shape']}")
        elif status == "ALL_NAN_INF":
            print(f"{name:<35} {'ALL_NaN/Inf':<14} a_nan={result['a_nan']} b_nan={result['b_nan']} "
                  f"a_inf={result['a_inf']} b_inf={result['b_inf']}")

    # Also check for TP=2-only files (AllReduce results)
    if is_two_rank:
        print()
        print("── TP=2 AllReduce results (rank0 vs rank1 consistency) ──")
        print(f"{'Checkpoint':<35} {'Status':<14} {'MaxAbsErr':>12} {'CosineSim':>12}")
        print("─" * 85)

        for name in sorted(tp2r0_meta.keys()):
            step_name = extract_step_name(name)
            if step_name not in EXACT_MATCH:
                continue
            r0_bin = tp2_rank0_dir / f"{name}.bin"
            r1_bin = tp2_rank1_dir / f"{name}.bin"
            if not r0_bin.exists() or not r1_bin.exists():
                continue
            r0 = load_tensor(r0_bin, tp2r0_meta[name].get("dtype", "bfloat16"))
            r1 = load_tensor(r1_bin, tp2r1_meta.get(name, {}).get("dtype", "bfloat16"))
            result = compare_tensors(r0, r1)
            if result["status"] == "OK":
                flag = ""
                if result["max_abs_err"] > 1e-6:
                    flag = " ⚠️"
                print(f"{name:<35} {'OK' + flag:<14} {result['max_abs_err']:>12.6f} {result['cosine_sim']:>12.8f}")
            else:
                print(f"{name:<35} {result['status']:<14}")

    print()


if __name__ == "__main__":
    main()
