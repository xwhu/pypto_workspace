#!/usr/bin/env python3
"""
Generate reference data for Ascend operator tests.

Each operator gets random FP16 inputs and PyTorch computed reference outputs,
saved as raw .bin files. The Rust integration tests load these files, run the
same operation on the Ascend NPU, and compare results.

Usage:
    source ~/.venv/bin/activate
    python tests/gen_reference.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "reference_data")
os.makedirs(OUT, exist_ok=True)


def save_f16(name: str, t: torch.Tensor):
    """Save tensor as contiguous FP16 raw bytes."""
    t.cpu().contiguous().half().numpy().tofile(os.path.join(OUT, f"{name}.bin"))


def save_i64(name: str, t: torch.Tensor):
    """Save tensor as contiguous Int64 raw bytes."""
    t.cpu().contiguous().long().numpy().tofile(os.path.join(OUT, f"{name}.bin"))


def save_i32(name: str, t: torch.Tensor):
    """Save tensor as contiguous Int32 raw bytes."""
    t.cpu().contiguous().int().numpy().tofile(os.path.join(OUT, f"{name}.bin"))


def save_meta(name: str, **kwargs):
    """Save shape/dtype metadata as a simple text file."""
    with open(os.path.join(OUT, f"{name}.meta"), "w") as f:
        for k, v in kwargs.items():
            f.write(f"{k}={v}\n")


# ═══════════════════════════════════════════════════════════════════
# 1. RmsNorm
# ═══════════════════════════════════════════════════════════════════
print("Generating: RmsNorm")
x = torch.randn(1, 4, 1024, dtype=torch.float16)
w = torch.randn(1024, dtype=torch.float16)
eps = 1e-6

# Manual RmsNorm (matches Qwen3 implementation)
x_f32 = x.float()
rms = torch.sqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
y_ref = (x_f32 / rms * w.float()).half()

save_f16("rmsnorm_x", x)
save_f16("rmsnorm_w", w)
save_f16("rmsnorm_y", y_ref)
save_meta("rmsnorm", x_shape="1,4,1024", w_shape="1024", eps=str(eps))

# ═══════════════════════════════════════════════════════════════════
# 2. MatMul (Linear)
# ═══════════════════════════════════════════════════════════════════
print("Generating: MatMul (Linear)")
a = torch.randn(1, 4, 1024, dtype=torch.float16)
weight = torch.randn(2048, 1024, dtype=torch.float16)  # [out_features, in_features]

# F.linear computes: a @ weight^T
out_ref = F.linear(a.float(), weight.float()).half()

save_f16("matmul_a", a)
save_f16("matmul_w", weight)
save_f16("matmul_out", out_ref)
save_meta("matmul", a_shape="1,4,1024", w_shape="2048,1024", out_shape="1,4,2048")

# ═══════════════════════════════════════════════════════════════════
# 3. Embedding
# ═══════════════════════════════════════════════════════════════════
print("Generating: Embedding")
ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
table = torch.randn(100, 64, dtype=torch.float16)
emb_ref = F.embedding(ids, table.float()).half()

save_i64("embedding_ids", ids)
save_f16("embedding_table", table)
save_f16("embedding_out", emb_ref)
save_meta("embedding", ids_shape="1,4", table_shape="100,64", out_shape="1,4,64")

# ═══════════════════════════════════════════════════════════════════
# 4. Add (residual connection)
# ═══════════════════════════════════════════════════════════════════
print("Generating: Add")
add_a = torch.randn(1, 4, 1024, dtype=torch.float16)
add_b = torch.randn(1, 4, 1024, dtype=torch.float16)
add_out = (add_a.float() + add_b.float()).half()

save_f16("add_a", add_a)
save_f16("add_b", add_b)
save_f16("add_out", add_out)
save_meta("add", shape="1,4,1024")

# ═══════════════════════════════════════════════════════════════════
# 5. SiluMul (SwiGLU activation)
# ═══════════════════════════════════════════════════════════════════
print("Generating: SiluMul")
gate = torch.randn(1, 4, 2048, dtype=torch.float16)
up = torch.randn(1, 4, 2048, dtype=torch.float16)
silu_out = (F.silu(gate.float()) * up.float()).half()

save_f16("silumul_gate", gate)
save_f16("silumul_up", up)
save_f16("silumul_out", silu_out)
save_meta("silumul", shape="1,4,2048")

# ═══════════════════════════════════════════════════════════════════
# 6. ArgMax (sampling)
# ═══════════════════════════════════════════════════════════════════
print("Generating: ArgMax")
logits = torch.randn(1, 4, 1000, dtype=torch.float16)
# Take argmax of the last token's logits (index 3)
argmax_ref = logits[0, -1, :].float().argmax().item()

save_f16("argmax_logits", logits)
# Save as single int32
np.array([argmax_ref], dtype=np.int32).tofile(os.path.join(OUT, "argmax_out.bin"))
save_meta("argmax", logits_shape="1,4,1000", argmax_value=str(argmax_ref))

# ═══════════════════════════════════════════════════════════════════
# 7. Softmax
# ═══════════════════════════════════════════════════════════════════
print("Generating: Softmax")
sf_input = torch.randn(1, 4, 1000, dtype=torch.float16)
sf_out = F.softmax(sf_input.float(), dim=-1).half()

save_f16("softmax_input", sf_input)
save_f16("softmax_out", sf_out)
save_meta("softmax", shape="1,4,1000")

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
files = [f for f in os.listdir(OUT) if f.endswith('.bin')]
total_bytes = sum(os.path.getsize(os.path.join(OUT, f)) for f in files)
print(f"\n✓ Reference data saved to {OUT}/")
print(f"  {len(files)} .bin files, {total_bytes / 1024:.1f} KB total")
for f in sorted(files):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"    {f:40s} {size:>8,} bytes")
