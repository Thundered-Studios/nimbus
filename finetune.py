"""
Nimbus fine-tuning script (LoRA / QLoRA).

Fine-tunes Nimbus on a JSONL dataset of chat conversations.

Dataset format (data/finetune.jsonl):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    {"messages": [...]}

Usage:
    pip install peft trl
    python finetune.py --data data/finetune.jsonl --variant 1.5b --output out/nimbus-ft
    python finetune.py --data data/finetune.jsonl --variant 7b --4bit --output out/nimbus-ft
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments
from trl import SFTTrainer

from nimbus import Nimbus, NimbusConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str,   required=True)
    p.add_argument("--variant",    type=str,   default="4b")
    p.add_argument("--output",     type=str,   default="out/nimbus-ft")
    p.add_argument("--4bit",       dest="load_4bit", action="store_true")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--batch",      type=int,   default=4)
    p.add_argument("--grad-accum", type=int,   default=4)
    p.add_argument("--max-seq",    type=int,   default=2048)
    p.add_argument("--lora-r",     type=int,   default=16)
    p.add_argument("--lora-alpha", type=int,   default=32)
    return p.parse_args()


def load_dataset_from_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def main():
    args = parse_args()

    cfg = NimbusConfig(load_in_4bit=args.load_4bit)
    nimbus = Nimbus.load(variant=args.variant, config=cfg)
    model = nimbus.enable_training(gradient_checkpointing=True)
    tokenizer = nimbus._tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = load_dataset_from_jsonl(args.data)
    print(f"Loaded {len(dataset)} examples from {args.data}")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=args.max_seq,
    )

    print("Starting fine-tune...")
    trainer.train()

    nimbus._model = model
    nimbus.save(args.output)
    print(f"Done. Model saved to {args.output}")


if __name__ == "__main__":
    main()
