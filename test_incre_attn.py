import torch
import torch_npu

B = 1
H = 16
KV_H = 8
D = 128
S = 25
block_size = 16

q = torch.randn(B, 1, H * D, dtype=torch.float16).npu()
k = torch.randn(10, block_size, KV_H, D, dtype=torch.float16).npu()
v = torch.randn(10, block_size, KV_H, D, dtype=torch.float16).npu()
block_table = torch.tensor([[0, 1]], dtype=torch.int32).npu()

actual_seq_lengths = torch.tensor([S], dtype=torch.int64).npu()

try:
    print("Testing npu_incre_flash_attention_v4 with list containing 1 tensor...", flush=True)
    out = torch_npu.npu_incre_flash_attention_v4(
        query=q,
        key=[k],
        value=[v],
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths,
        num_heads=H,
        input_layout="BSH",
        num_kv_heads=KV_H,
        scale_value=1.0 / (D ** 0.5),
        block_size=block_size,
        inner_precise=0,
    )
    print("Success! Output shape:", out.shape, flush=True)
except Exception as e:
    print("Failed shape 1:", str(e), flush=True)

k_list = [torch.randn(block_size, KV_H, D, dtype=torch.float16).npu() for _ in range(2)]
v_list = [torch.randn(block_size, KV_H, D, dtype=torch.float16).npu() for _ in range(2)]

try:
    print("Testing npu_incre_flash_attention_v4 with list containing block tensors...", flush=True)
    out = torch_npu.npu_incre_flash_attention_v4(
        query=q,
        key=k_list,
        value=v_list,
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths,
        num_heads=H,
        input_layout="BSH",
        num_kv_heads=KV_H,
        scale_value=1.0 / (D ** 0.5),
        block_size=block_size,
        inner_precise=0,
    )
    print("Success 2! Output shape:", out.shape, flush=True)
except Exception as e:
    print("Failed shape 2:", str(e), flush=True)

