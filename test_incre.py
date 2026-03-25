import torch
import torch_npu
import sys

# use device 7 to avoid device 0 since it is abnormal
torch.npu.set_device("npu:7")

B = 1
H = 16
KV_H = 8
D = 128
S = 25
block_size = 16
blocks_per_chunk = 256

q = torch.randn(B, 1, H * D, dtype=torch.float16).npu()

# [blocks_per_chunk, block_size, kv_heads, head_dim]
k_pool = torch.randn(blocks_per_chunk, block_size, KV_H, D, dtype=torch.float16).npu()
v_pool = torch.randn(blocks_per_chunk, block_size, KV_H, D, dtype=torch.float16).npu()

block_table = torch.tensor([[0, 1]], dtype=torch.int32).npu()
actual_seq_lengths = torch.tensor([S], dtype=torch.int64).npu()

# Attempt 1: As a single continuous pool
try:
    print("Test 1: Normal continuous pool...", flush=True)
    out = torch_npu.npu_incre_flash_attention_v4(
        query=q,
        key=[k_pool],
        value=[v_pool],
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths,
        num_heads=H,
        num_kv_heads=KV_H,
        input_layout="BSH",
        scale_value=1.0 / (D ** 0.5),
        block_size=block_size,
        inner_precise=0,
    )
    print("Test 1 Success!", out.shape, flush=True)
except Exception as e:
    print("Test 1 Failed:", str(e), flush=True)

# Attempt 2: BNSD layout
q_bnsd = torch.randn(B, H, 1, D, dtype=torch.float16).npu()
k_pool_bnsd = torch.randn(blocks_per_chunk, KV_H, block_size, D, dtype=torch.float16).npu()
v_pool_bnsd = torch.randn(blocks_per_chunk, KV_H, block_size, D, dtype=torch.float16).npu()
try:
    print("\nTest 2: BNSD layout...", flush=True)
    out = torch_npu.npu_incre_flash_attention_v4(
        query=q_bnsd,
        key=[k_pool_bnsd],
        value=[v_pool_bnsd],
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths,
        num_heads=H,
        num_kv_heads=KV_H,
        input_layout="BNSD",
        scale_value=1.0 / (D ** 0.5),
        block_size=block_size,
        inner_precise=0,
    )
    print("Test 2 Success!", out.shape, flush=True)
except Exception as e:
    print("Test 2 Failed:", str(e), flush=True)

# Attempt 3: List of blocks instead of list of pools
k_blocks = [torch.randn(block_size, KV_H, D, dtype=torch.float16).npu() for _ in range(2)]
v_blocks = [torch.randn(block_size, KV_H, D, dtype=torch.float16).npu() for _ in range(2)]
try:
    print("\nTest 3: List of individual blocks...", flush=True)
    out = torch_npu.npu_incre_flash_attention_v4(
        query=q,
        key=k_blocks,
        value=v_blocks,
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths,
        num_heads=H,
        num_kv_heads=KV_H,
        input_layout="BSH",
        scale_value=1.0 / (D ** 0.5),
        block_size=block_size,
        inner_precise=0,
    )
    print("Test 3 Success!", out.shape, flush=True)
except Exception as e:
    print("Test 3 Failed:", str(e), flush=True)

# Attempt 4: Wait, check PyPTO shape parameters!
# What if key cache is essentially BSH layout but [total_pool_blocks, block_size, head_dim]??
try:
    print("\nTest 4: PyPTO shape Flattened...", flush=True)
    k_flat = torch.randn(blocks_per_chunk * block_size, H * D, dtype=torch.float16).npu()
    v_flat = torch.randn(blocks_per_chunk * block_size, H * D, dtype=torch.float16).npu()
    out = torch_npu.npu_incre_flash_attention_v4(
        query=q,
        key=[k_flat],
        value=[v_flat],
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths,
        num_heads=H,
        num_kv_heads=KV_H,
        input_layout="BSH",
        scale_value=1.0 / (D ** 0.5),
        block_size=block_size,
        inner_precise=0,
    )
    print("Test 4 Success!", out.shape, flush=True)
except Exception as e:
    print("Test 4 Failed:", str(e), flush=True)

