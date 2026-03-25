"""
Corrected test based on actual schema:
npu::npu_incre_flash_attention(
    Tensor query, Tensor key, Tensor value,
    *, Tensor? padding_mask, Tensor? atten_mask, Tensor? pse_shift,
    SymInt[]? actual_seq_lengths, Tensor? antiquant_scale, Tensor? antiquant_offset,
    Tensor? block_table, Tensor? dequant_scale1, Tensor? quant_scale1,
    Tensor? dequant_scale2, Tensor? quant_scale2, Tensor? quant_offset2,
    Tensor? kv_padding_size,
    int num_heads=1, float scale_value=1.,
    str input_layout="BSH", int num_key_value_heads=0,
    int block_size=0, int inner_precise=1
) -> Tensor

Key insight: key/value are TENSORS (the full KV cache pool), NOT lists.
block_table maps logical blocks to physical blocks within these tensors.
"""
import torch
import torch_npu

B = 1; H = 16; KV_H = 8; D = 128; S = 25
block_size = 16; pool_blocks = 256
device = "npu:2"
scale = 1.0 / (D ** 0.5)

def run_test(name, **kwargs):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
    try:
        out = torch_npu.npu_incre_flash_attention(**kwargs)
        print(f"  ✅ SUCCESS! output shape={out.shape}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

# Test 1: BSH layout, key as full cache pool [pool_blocks, block_size, kv_heads, head_dim]
# This is 4D... but BSH expects 3D for key. So reshape to [pool_blocks * block_size, KV_H * D]
# Actually the key should be shaped such that for BSH:
#   query: [B, 1, H * D]        (B=1, S_q=1)
#   key:   [pool_blocks * block_size, KV_H * D]  (treated as total tokens)
# Wait, that doesn't match. For non-paged BSH:
#   key: [B, S_kv, KV_H * D]
# For paged, the "S_kv" dimension is the total pool token slots.

# Test 1: BSH, key=[1, pool*bs, KVH*D]
run_test("T1: BSH key=[1, pool*bs, KVH*D]",
    query=torch.randn(B, 1, H * D, dtype=torch.float16, device=device),
    key=torch.randn(1, pool_blocks * block_size, KV_H * D, dtype=torch.float16, device=device),
    value=torch.randn(1, pool_blocks * block_size, KV_H * D, dtype=torch.float16, device=device),
    block_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
    actual_seq_lengths=[S],
    num_heads=H, num_key_value_heads=KV_H,
    input_layout="BSH", scale_value=scale,
    block_size=block_size, inner_precise=0,
)

# Test 2: BNSD, key=[1, KVH, pool*bs, D]
run_test("T2: BNSD key=[1, KVH, pool*bs, D]",
    query=torch.randn(B, H, 1, D, dtype=torch.float16, device=device),
    key=torch.randn(1, KV_H, pool_blocks * block_size, D, dtype=torch.float16, device=device),
    value=torch.randn(1, KV_H, pool_blocks * block_size, D, dtype=torch.float16, device=device),
    block_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
    actual_seq_lengths=[S],
    num_heads=H, num_key_value_heads=KV_H,
    input_layout="BNSD", scale_value=scale,
    block_size=block_size, inner_precise=0,
)

# Test 3: BSH, key=[pool_blocks, block_size, KVH * D] (3D with pool_blocks as batch)
run_test("T3: BSH key=[pool_blocks, bs, KVH*D]",
    query=torch.randn(B, 1, H * D, dtype=torch.float16, device=device),
    key=torch.randn(pool_blocks, block_size, KV_H * D, dtype=torch.float16, device=device),
    value=torch.randn(pool_blocks, block_size, KV_H * D, dtype=torch.float16, device=device),
    block_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
    actual_seq_lengths=[S],
    num_heads=H, num_key_value_heads=KV_H,
    input_layout="BSH", scale_value=scale,
    block_size=block_size, inner_precise=0,
)

# Test 4: What if in paged mode, key shape doesn't matter much as long as it's contiguous
# and block_table indexes into it? Let's try just the full pool as flat 2D
run_test("T4: BSH key=[pool*bs, KVH*D] 2D",
    query=torch.randn(B, 1, H * D, dtype=torch.float16, device=device),
    key=torch.randn(pool_blocks * block_size, KV_H * D, dtype=torch.float16, device=device),
    value=torch.randn(pool_blocks * block_size, KV_H * D, dtype=torch.float16, device=device),
    block_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
    actual_seq_lengths=[S],
    num_heads=H, num_key_value_heads=KV_H,
    input_layout="BSH", scale_value=scale,
    block_size=block_size, inner_precise=0,
)

# Test 5: Try actual_seq_lengths as Tensor instead of list
run_test("T5: same as T1 but actual_seq_lengths=Tensor",
    query=torch.randn(B, 1, H * D, dtype=torch.float16, device=device),
    key=torch.randn(1, pool_blocks * block_size, KV_H * D, dtype=torch.float16, device=device),
    value=torch.randn(1, pool_blocks * block_size, KV_H * D, dtype=torch.float16, device=device),
    block_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
    actual_seq_lengths=torch.tensor([S], dtype=torch.int64, device=device),
    num_heads=H, num_key_value_heads=KV_H,
    input_layout="BSH", scale_value=scale,
    block_size=block_size, inner_precise=0,
)

# Test 6: Non-paged (no block_table, no block_size) to verify API works at all
run_test("T6: non-paged BSH (no block_table)",
    query=torch.randn(B, 1, H * D, dtype=torch.float16, device=device),
    key=torch.randn(B, S, KV_H * D, dtype=torch.float16, device=device),
    value=torch.randn(B, S, KV_H * D, dtype=torch.float16, device=device),
    actual_seq_lengths=[S],
    num_heads=H, num_key_value_heads=KV_H,
    input_layout="BSH", scale_value=scale,
)

# Test 7: Non-paged BNSD
run_test("T7: non-paged BNSD (no block_table)",
    query=torch.randn(B, H, 1, D, dtype=torch.float16, device=device),
    key=torch.randn(B, KV_H, S, D, dtype=torch.float16, device=device),
    value=torch.randn(B, KV_H, S, D, dtype=torch.float16, device=device),
    actual_seq_lengths=[S],
    num_heads=H, num_key_value_heads=KV_H,
    input_layout="BNSD", scale_value=scale,
)

print("\n" + "=" * 60)
print("DONE.")
