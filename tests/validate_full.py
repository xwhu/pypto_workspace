#!/usr/bin/env python3
"""
Full 28-layer Qwen3-0.6B forward pass using safetensors + numpy.
Computes the expected output token for "hi" to compare with Rust server.
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
    num_layers = cfg['num_hidden_layers']
    hidden_size = cfg['hidden_size']

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    token_ids = tokenizer.encode("hi").ids
    print(f"Token IDs: {token_ids}, num_layers: {num_layers}")

    weights = load_safetensors_numpy(MODEL_DIR)
    
    # Embedding
    hidden = weights["model.embed_tokens.weight"][token_ids[0]].reshape(1, 1, -1)
    
    gqa_ratio = num_heads // num_kv_heads

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        
        # Input LayerNorm
        normed = rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"], rms_eps)
        
        # Q/K/V
        q = np.matmul(normed.astype(np.float32), weights[f"{prefix}.self_attn.q_proj.weight"].astype(np.float32).T).astype(np.float16)
        k = np.matmul(normed.astype(np.float32), weights[f"{prefix}.self_attn.k_proj.weight"].astype(np.float32).T).astype(np.float16)
        v = np.matmul(normed.astype(np.float32), weights[f"{prefix}.self_attn.v_proj.weight"].astype(np.float32).T).astype(np.float16)
        
        # QK Norm
        q = rms_norm(q.reshape(-1, head_dim), weights[f"{prefix}.self_attn.q_norm.weight"], rms_eps).reshape(1, 1, -1)
        k = rms_norm(k.reshape(-1, head_dim), weights[f"{prefix}.self_attn.k_norm.weight"], rms_eps).reshape(1, 1, -1)
        
        # RoPE at position 0: identity
        # Attention: single token → output = V with GQA broadcast
        v_4d = v.reshape(1, 1, num_kv_heads, head_dim)
        attn_heads = []
        for h in range(num_heads):
            kv_h = h // gqa_ratio
            attn_heads.append(v_4d[0, 0, kv_h].astype(np.float32))
        attn_out = np.stack(attn_heads).reshape(1, 1, num_heads * head_dim).astype(np.float16)
        
        # O projection
        o_proj = np.matmul(attn_out.astype(np.float32), weights[f"{prefix}.self_attn.o_proj.weight"].astype(np.float32).T).astype(np.float16)
        
        # Residual
        hidden = (hidden.astype(np.float32) + o_proj.astype(np.float32)).astype(np.float16)
        
        # Post-attention norm
        normed2 = rms_norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], rms_eps)
        
        # MLP
        gate = np.matmul(normed2.astype(np.float32), weights[f"{prefix}.mlp.gate_proj.weight"].astype(np.float32).T).astype(np.float16)
        up = np.matmul(normed2.astype(np.float32), weights[f"{prefix}.mlp.up_proj.weight"].astype(np.float32).T).astype(np.float16)
        mlp_out = (silu(gate).astype(np.float32) * up.astype(np.float32)).astype(np.float16)
        down = np.matmul(mlp_out.astype(np.float32), weights[f"{prefix}.mlp.down_proj.weight"].astype(np.float32).T).astype(np.float16)
        
        # Residual
        hidden = (hidden.astype(np.float32) + down.astype(np.float32)).astype(np.float16)
        
        if layer_idx < 3 or layer_idx == num_layers - 1:
            print(f"Layer {layer_idx}: hidden[:5] = {hidden[0,0,:5].astype(np.float32).tolist()}")

    # Final norm
    final_normed = rms_norm(hidden, weights["model.norm.weight"], rms_eps)
    print(f"Final norm: {final_normed[0,0,:5].astype(np.float32).tolist()}")

    # LM Head
    lm_head_weight = weights.get("lm_head.weight", weights.get("model.embed_tokens.weight"))
    logits = np.matmul(final_normed.astype(np.float32), lm_head_weight.astype(np.float32).T).astype(np.float16)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits[:10]: {logits[0,0,:10].astype(np.float32).tolist()}")

    # Argmax
    top_token = np.argmax(logits[0, 0].astype(np.float32))
    top_logit = logits[0, 0, top_token].astype(np.float32)
    
    decoded = tokenizer.decode([int(top_token)])
    print(f"\nPredicted token: {top_token} ({decoded!r}), logit={top_logit:.4f}")
    
    # Top 5
    top5_indices = np.argsort(logits[0, 0].astype(np.float32))[-5:][::-1]
    print("Top 5:")
    for idx in top5_indices:
        val = logits[0, 0, idx].astype(np.float32)
        text = tokenizer.decode([int(idx)])
        print(f"  {idx:6d} ({text!r:20s}) logit={val:.4f}")

if __name__ == "__main__":
    main()
