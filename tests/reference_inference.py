#!/usr/bin/env python3
"""
Quick Qwen3-0.6B inference reference.

Run on the NPU server to get reference outputs for comparison.
Usage: python3 tests/reference_inference.py
"""
import torch
import os

# Use CPU for reference (avoids NPU-specific issues)
DEVICE = "cpu"
MODEL_DIR = os.path.expanduser("~/pypto_workspace/models/Qwen3-0.6B")

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Prompt: {prompt!r}")
    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Number of tokens: {input_ids.shape[1]}")

    print(f"\nLoading model from {MODEL_DIR} (FP32 CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()

    # Single forward pass to get logits
    with torch.no_grad():
        outputs = model(input_ids.to(DEVICE))
        logits = outputs.logits  # [batch, seq, vocab]
        print(f"\nLogits shape: {logits.shape}")
        print(f"Last token logits top-10:")
        last_logits = logits[0, -1]
        top10_vals, top10_ids = torch.topk(last_logits, 10)
        for v, i in zip(top10_vals, top10_ids):
            decoded = tokenizer.decode([i.item()])
            print(f"  token {i.item():6d} ({decoded!r:20s}) logit={v.item():.4f}")

    # Full generation
    print("\n--- Generation (greedy, 10 tokens) ---")
    with torch.no_grad():
        gen = model.generate(
            input_ids.to(DEVICE),
            max_new_tokens=10,
            do_sample=False,
            temperature=1.0,
        )
    gen_tokens = gen[0].tolist()
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    print(f"Generated IDs: {gen_tokens}")
    print(f"Generated text: {gen_text!r}")

    # Also test intermediate: dump first layer Q after RoPE
    print("\n--- Layer 0 intermediate check ---")
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids.to(DEVICE))
        print(f"Embedding out [0,0,:5]: {hidden[0,0,:5].tolist()}")

        normed = model.model.layers[0].input_layernorm(hidden)
        print(f"RmsNorm out [0,0,:5]: {normed[0,0,:5].tolist()}")

        q = model.model.layers[0].self_attn.q_proj(normed)
        print(f"Q proj out [0,0,:5]: {q[0,0,:5].tolist()}")
        print(f"Q proj shape: {q.shape}")

if __name__ == "__main__":
    main()
