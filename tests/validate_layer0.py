#!/usr/bin/env python3
"""
Full single-layer Qwen3 validation using safetensors + numpy + tokenizers.
Computes through QK norm, RoPE (identity at pos 0), attention, O proj, residual.
"""
import numpy as np
import json
import os

MODEL_DIR = os.path.expanduser("~/pypto_workspace/Qwen3-0.6B")

def load_safetensors_numpy(directory):
    from safetensors import safe_open
    weights = {}
    for fname in sorted(os.listdir(directory)):
        if fname.endswith('.safetensors'):
            path = os.path.join(directory, fname)
            with safe_open(path, framework="numpy") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
    return weights

def rms_norm(x, weight, eps=1e-6):
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    return ((x_f32 / rms) * weight.astype(np.float32)).astype(np.float16)

def silu(x):
    x_f32 = x.astype(np.float32)
    return (x_f32 / (1.0 + np.exp(-x_f32))).astype(np.float16)

def main():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)
    
    num_heads = cfg['num_attention_heads']
    num_kv_heads = cfg['num_key_value_heads']
    head_dim = cfg.get('head_dim', cfg['hidden_size'] // num_heads)
    rms_eps = cfg.get('rms_norm_eps', 1e-6)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    token_ids = tokenizer.encode("hi").ids
    print(f"Token IDs: {token_ids}")

    weights = load_safetensors_numpy(MODEL_DIR)
    
    # === Embedding ===
    embed_table = weights["model.embed_tokens.weight"]
    hidden = embed_table[token_ids[0]].reshape(1, 1, -1)
    print(f"Embedding: {hidden[0,0,:5].astype(np.float32).tolist()}")

    # === Layer 0 ===
    prefix = "model.layers.0"
    
    # Input LayerNorm
    normed = rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"], rms_eps)
    print(f"RmsNorm: {normed[0,0,:5].astype(np.float32).tolist()}")
    
    # Q/K/V projections
    q = np.matmul(normed.astype(np.float32), weights[f"{prefix}.self_attn.q_proj.weight"].astype(np.float32).T).astype(np.float16)
    k = np.matmul(normed.astype(np.float32), weights[f"{prefix}.self_attn.k_proj.weight"].astype(np.float32).T).astype(np.float16)
    v = np.matmul(normed.astype(np.float32), weights[f"{prefix}.self_attn.v_proj.weight"].astype(np.float32).T).astype(np.float16)
    print(f"Q proj: {q[0,0,:5].astype(np.float32).tolist()}")
    print(f"K proj: {k[0,0,:5].astype(np.float32).tolist()}")
    print(f"V proj: {v[0,0,:5].astype(np.float32).tolist()}")

    # QK Norm: reshape to [B*S*num_heads, head_dim], apply rms_norm, reshape back
    q_reshaped = q.reshape(-1, head_dim)  # [16, 128]
    q_normed = rms_norm(q_reshaped, weights[f"{prefix}.self_attn.q_norm.weight"], rms_eps)
    q = q_normed.reshape(1, 1, -1)
    print(f"Q after qk_norm: {q[0,0,:5].astype(np.float32).tolist()}")

    k_reshaped = k.reshape(-1, head_dim)  # [8, 128]
    k_normed = rms_norm(k_reshaped, weights[f"{prefix}.self_attn.k_norm.weight"], rms_eps)
    k = k_normed.reshape(1, 1, -1)
    print(f"K after qk_norm: {k[0,0,:5].astype(np.float32).tolist()}")

    # RoPE at position 0: identity (cos=1, sin=0)
    print(f"RoPE at pos 0: identity (skip)")

    # === Attention (single token: output = V broadcast to all heads) ===
    # Reshape for attention
    q_4d = q.reshape(1, 1, num_heads, head_dim)      # [1, 1, 16, 128]
    k_4d = k.reshape(1, 1, num_kv_heads, head_dim)    # [1, 1, 8, 128]
    v_4d = v.reshape(1, 1, num_kv_heads, head_dim)    # [1, 1, 8, 128]

    # For single token: score = Q @ K^T / sqrt(d) = scalar per head
    # softmax(scalar) = 1.0
    # output = 1.0 * V  (with GQA broadcasting)
    scale = 1.0 / np.sqrt(head_dim)
    gqa_ratio = num_heads // num_kv_heads
    
    attn_out_heads = []
    for h in range(num_heads):
        kv_h = h // gqa_ratio
        score = np.sum(q_4d[0, 0, h].astype(np.float32) * k_4d[0, 0, kv_h].astype(np.float32)) * scale
        # softmax of single value = 1.0
        attn_out_heads.append(v_4d[0, 0, kv_h].astype(np.float32))
    
    attn_out = np.stack(attn_out_heads).reshape(1, 1, num_heads * head_dim).astype(np.float16)
    print(f"Attention out: {attn_out[0,0,:5].astype(np.float32).tolist()}")
    print(f"V direct: {v[0,0,:5].astype(np.float32).tolist()}")
    print(f"(For single token, attn out first head should equal V first kv-head)")

    # O projection
    o_proj = np.matmul(attn_out.astype(np.float32), weights[f"{prefix}.self_attn.o_proj.weight"].astype(np.float32).T).astype(np.float16)
    print(f"O proj out: {o_proj[0,0,:5].astype(np.float32).tolist()}")

    # Residual
    hidden = (hidden.astype(np.float32) + o_proj.astype(np.float32)).astype(np.float16)
    print(f"After attn residual: {hidden[0,0,:5].astype(np.float32).tolist()}")

    # Post-attention LayerNorm
    normed2 = rms_norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], rms_eps)
    print(f"Post-attn norm: {normed2[0,0,:5].astype(np.float32).tolist()}")

    # MLP
    gate = np.matmul(normed2.astype(np.float32), weights[f"{prefix}.mlp.gate_proj.weight"].astype(np.float32).T).astype(np.float16)
    up = np.matmul(normed2.astype(np.float32), weights[f"{prefix}.mlp.up_proj.weight"].astype(np.float32).T).astype(np.float16)
    mlp_out = silu(gate) * up.astype(np.float32)
    mlp_out = mlp_out.astype(np.float16)
    down = np.matmul(mlp_out.astype(np.float32), weights[f"{prefix}.mlp.down_proj.weight"].astype(np.float32).T).astype(np.float16)
    print(f"MLP down proj: {down[0,0,:5].astype(np.float32).tolist()}")

    # Residual
    hidden = (hidden.astype(np.float32) + down.astype(np.float32)).astype(np.float16)
    print(f"After MLP residual (layer 0 done): {hidden[0,0,:5].astype(np.float32).tolist()}")

    print("\n=== Compare these with Rust debug output for layer 0 ===")

if __name__ == "__main__":
    main()
