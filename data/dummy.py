"""
Generate small dummy train.bin / val.bin for local testing.
Uses random token IDs — the model won't learn anything real,
but it lets you verify the training loop runs without errors.

Usage:
    python data/dummy.py
    python data/dummy.py --tokens 5000000   # 5M tokens
"""

import os
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_tokens", type=int, default=2_000_000, help="Training tokens")
    p.add_argument("--val_tokens",   type=int, default=100_000,   help="Validation tokens")
    p.add_argument("--vocab_size",   type=int, default=50257)
    p.add_argument("--out_dir",      type=str, default="data")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split, n in [("train", args.train_tokens), ("val", args.val_tokens)]:
        path = os.path.join(args.out_dir, f"{split}.bin")
        data = np.random.randint(0, args.vocab_size, size=n, dtype=np.uint16)
        data.tofile(path)
        print(f"Wrote {n:,} tokens → {path}")

    print("\nDone. Run: python train.py")
    print("(This is random data — for real training run: python data/prepare.py)")


if __name__ == "__main__":
    main()
