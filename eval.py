"""
Nimbus-1 Evaluation Script

Evaluates a trained checkpoint on:
  - Validation perplexity
  - HellaSwag (zero-shot accuracy)
  - Sample generation

Usage:
    python eval.py --checkpoint out/ckpt.pt
    python eval.py --checkpoint out/ckpt.pt --hellaswag
    python eval.py --checkpoint out/ckpt.pt --generate --prompt "The future of AI is"
"""

import os
import json
import math
import argparse
import contextlib
import urllib.request

import numpy as np
import torch
import tiktoken

from model.transformer import Nimbus, NimbusConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--hellaswag", action="store_true")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--prompt", type=str, default="The future of artificial intelligence is")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = NimbusConfig(**ckpt["model_config"])
    model = Nimbus(model_cfg)
    state_dict = ckpt["model"]
    # Strip _orig_mod prefix if model was compiled
    unwanted = "_orig_mod."
    state_dict = {
        (k[len(unwanted):] if k.startswith(unwanted) else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, model_cfg, ckpt.get("iter_num", 0), ckpt.get("best_val_loss", None)


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(model, data_path: str, block_size: int, device: str,
                    max_batches: int = 200):
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    losses = []
    device_type = "cuda" if "cuda" in device else "cpu"

    for i in range(max_batches):
        start = i * block_size
        if start + block_size + 1 > len(data):
            break
        chunk = torch.from_numpy(data[start : start + block_size + 1].astype(np.int64))
        x = chunk[:-1].unsqueeze(0).to(device)
        y = chunk[1:].unsqueeze(0).to(device)
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())

    mean_loss = float(np.mean(losses))
    ppl = math.exp(mean_loss)
    return mean_loss, ppl


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------

HELLASWAG_URL = (
    "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
)


def download_hellaswag(cache_path="data/hellaswag_val.jsonl"):
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print("Downloading HellaSwag validation set...")
        urllib.request.urlretrieve(HELLASWAG_URL, cache_path)
    return cache_path


def render_example(example: dict, enc):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_tokens = enc.encode_ordinary(ctx)
    token_rows, mask_rows = [], []

    for end in endings:
        end_tokens = enc.encode_ordinary(" " + end)
        tokens = ctx_tokens + end_tokens
        mask   = [0] * len(ctx_tokens) + [1] * len(end_tokens)
        token_rows.append(tokens)
        mask_rows.append(mask)

    max_len = max(len(t) for t in token_rows)
    tokens_arr = torch.zeros(4, max_len, dtype=torch.long)
    mask_arr   = torch.zeros(4, max_len, dtype=torch.long)
    for i, (t, m) in enumerate(zip(token_rows, mask_rows)):
        tokens_arr[i, :len(t)] = torch.tensor(t)
        mask_arr[i, :len(m)]   = torch.tensor(m)

    return tokens_arr, mask_arr, int(label)


@torch.no_grad()
def eval_hellaswag(model, device: str, block_size: int) -> float:
    enc = tiktoken.get_encoding("gpt2")
    cache_path = download_hellaswag()
    device_type = "cuda" if "cuda" in device else "cpu"

    num_correct = num_total = 0

    with open(cache_path) as f:
        for line in f:
            example = json.loads(line)
            tokens, mask, label = render_example(example, enc)

            tokens = tokens[:, :block_size].to(device)
            mask   = mask[:, :block_size].to(device)

            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(tokens)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_tokens = tokens[..., 1:].contiguous()
            shift_mask   = mask[..., 1:].contiguous()

            logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_logprobs = logprobs.gather(
                dim=-1, index=shift_tokens.unsqueeze(-1)
            ).squeeze(-1)

            masked_logprobs = (token_logprobs * shift_mask).sum(dim=-1)
            counts = shift_mask.sum(dim=-1).clamp(min=1)
            avg_logprobs = masked_logprobs / counts

            pred = avg_logprobs.argmax().item()
            num_correct += int(pred == label)
            num_total   += 1

            if num_total % 1000 == 0:
                print(f"  {num_correct}/{num_total} = {num_correct/num_total*100:.2f}%")

    return num_correct / num_total


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sample(model, prompt: str, max_new_tokens: int, device: str) -> str:
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=200)
    return enc.decode(out[0].tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_cfg, iter_num, best_val_loss = load_model(args.checkpoint, args.device)
    print(f"Model: {model.get_num_params()/1e6:.1f}M params | "
          f"iter {iter_num} | best val loss {best_val_loss:.4f}")

    val_path = os.path.join(args.data_dir, "val.bin")
    if os.path.exists(val_path):
        print("\nPerplexity...")
        val_loss, ppl = eval_perplexity(model, val_path, model_cfg.block_size, args.device)
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")

    if args.hellaswag:
        print("\nHellaSwag...")
        acc = eval_hellaswag(model, args.device, model_cfg.block_size)
        print(f"  Accuracy: {acc*100:.2f}%  (GPT-2 124M baseline: ~29.4%)")

    if args.generate:
        print(f'\nGenerating from: "{args.prompt}"')
        text = generate_sample(model, args.prompt, args.max_new_tokens, args.device)
        print("-" * 60)
        print(text)
        print("-" * 60)


if __name__ == "__main__":
    main()
