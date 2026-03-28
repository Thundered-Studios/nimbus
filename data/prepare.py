"""
Data preparation for Nimbus-1.

Downloads a 10B token sample of FineWeb from Hugging Face,
tokenizes it with the GPT-2 BPE tokenizer (tiktoken),
and saves as memory-mapped binary files for fast training.

Usage:
    python data/prepare.py --tokens 10BT --split 0.0005

Output:
    data/train.bin  - training tokens (~9.995B)
    data/val.bin    - validation tokens (~5M)
"""

import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=str, default="10BT",
                   help="FineWeb sample size: 10BT or 100BT")
    p.add_argument("--val_split", type=float, default=0.0005,
                   help="Fraction of data to use for validation")
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--num_proc", type=int, default=8)
    return p.parse_args()


def tokenize(example, enc):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}


def write_bin(out_path, dataset):
    total_tokens = sum(dataset["len"])
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    idx = 0
    for batch in tqdm(dataset.iter(batch_size=1024), desc=f"Writing {out_path}"):
        for ids in batch["ids"]:
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)
    arr.flush()
    print(f"Wrote {total_tokens:,} tokens to {out_path}")
    return total_tokens


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    print(f"Vocab size: {enc.n_vocab}")

    print(f"Loading FineWeb sample: {args.tokens}")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=f"sample-{args.tokens}",
        split="train",
        num_proc=args.num_proc,
    )

    split = dataset.train_test_split(
        test_size=args.val_split,
        seed=42,
        shuffle=True,
    )
    train_data = split["train"]
    val_data   = split["test"]

    print("Tokenizing...")
    train_tok = train_data.map(
        lambda ex: tokenize(ex, enc),
        remove_columns=["text"],
        num_proc=args.num_proc,
        desc="Tokenizing train",
    )
    val_tok = val_data.map(
        lambda ex: tokenize(ex, enc),
        remove_columns=["text"],
        num_proc=args.num_proc,
        desc="Tokenizing val",
    )

    n_train = write_bin(os.path.join(args.out_dir, "train.bin"), train_tok)
    n_val   = write_bin(os.path.join(args.out_dir, "val.bin"),   val_tok)

    print(f"\nDone.")
    print(f"  Train: {n_train / 1e9:.2f}B tokens")
    print(f"  Val:   {n_val  / 1e6:.2f}M tokens")


if __name__ == "__main__":
    main()
