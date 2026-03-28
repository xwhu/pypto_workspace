#!/usr/bin/env python3
"""Scan a single TP dump directory for NaN/Inf values.

Usage:
    python scripts/scan_overflow.py /tmp/tp2_hp_rank0

Reads meta.json for dtype info, loads each .bin file, and reports
which checkpoints contain NaN or Inf values and their counts.
"""

import sys, json
from pathlib import Path
import numpy as np


def bf16_to_fp32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)

def fp16_to_fp32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)

def load(path: Path, dtype: str) -> np.ndarray:
    raw = path.read_bytes()
    if dtype == "bfloat16": return bf16_to_fp32(raw)
    if dtype == "float16": return fp16_to_fp32(raw)
    if dtype == "float32": return np.frombuffer(raw, dtype=np.float32)
    return bf16_to_fp32(raw)


def main():
    dump_dir = Path(sys.argv[1])
    meta_path = dump_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    print(f"\nScanning {dump_dir} for NaN/Inf...\n")
    print(f"{'Checkpoint':<40} {'NaN':>8} {'Inf':>8} {'Max':>14} {'Min':>14} {'Shape'}")
    print("─" * 110)

    found_any = False
    for name in sorted(meta.keys()) if meta else sorted(p.stem for p in dump_dir.glob("*.bin")):
        bin_path = dump_dir / f"{name}.bin"
        if not bin_path.exists():
            continue
        dtype = meta.get(name, {}).get("dtype", "bfloat16")
        shape = meta.get(name, {}).get("shape", [])
        arr = load(bin_path, dtype)

        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        finite = arr[np.isfinite(arr)]
        vmax = float(np.max(finite)) if len(finite) > 0 else float('nan')
        vmin = float(np.min(finite)) if len(finite) > 0 else float('nan')

        flag = ""
        if n_nan > 0 or n_inf > 0:
            flag = " ⚠️ OVERFLOW"
            found_any = True
        elif abs(vmax) > 1e4 or abs(vmin) > 1e4:
            flag = " ⚡ LARGE"

        print(f"{name:<40} {n_nan:>8} {n_inf:>8} {vmax:>14.4f} {vmin:>14.4f} {shape}{flag}")

    print()
    if not found_any:
        print("✅ No NaN/Inf found in any checkpoint.")
    else:
        print("❌ Overflow detected! See ⚠️ lines above.")


if __name__ == "__main__":
    main()
