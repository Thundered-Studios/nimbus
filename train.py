"""
Nimbus-1 Training Script

Runs on ANY hardware: multi-GPU (DDP), single GPU (CUDA/MPS), or CPU.
Auto-detects hardware and picks the right settings.

Single GPU / CPU:
    python train.py
    python train.py --config configs/single_gpu_8gb.yaml

Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=4 train.py --config configs/multi_gpu.yaml

Resume:
    python train.py --resume out/ckpt.pt
"""

import os
import sys
import time
import math
import yaml
import argparse
import contextlib
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.transformer import Nimbus, NimbusConfig
from data.dataset import get_dataloader


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_device():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Silicon"
    return "cpu", "CPU"


def get_dtype(device_type: str, requested: str) -> str:
    """Pick the best supported dtype for this device."""
    if device_type == "cpu":
        return "float32"
    if device_type == "mps":
        # MPS bfloat16 support is limited; float32 is safer
        return "float32"
    # CUDA: honour the requested dtype
    if requested == "bfloat16" and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return requested


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data"
    out_dir: str = "out"

    # Model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.0
    bias: bool = True
    gradient_checkpointing: bool = False
    use_flash_attn: bool = True
    norm_type: str = "layernorm"

    # Training
    max_iters: int = 19_073
    batch_size: int = 8
    grad_accum_steps: int = 64
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10
    save_interval: int = 1000

    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule
    warmup_iters: int = 715
    lr_decay_iters: int = 19_073
    min_lr: float = 6e-5

    # System
    device: str = "auto"   # "auto" | "cuda" | "mps" | "cpu" | "cuda:0" etc.
    dtype: str = "bfloat16"
    compile: bool = True
    compile_mode: str = "max-autotune"
    seed: int = 1337
    num_workers: int = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def is_ddp() -> bool:
    return int(os.environ.get("RANK", -1)) != -1

def ddp_rank() -> int:
    return int(os.environ.get("RANK", 0))

def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))

def is_master() -> bool:
    return ddp_rank() == 0

def log(msg: str):
    if is_master():
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(cfg: TrainConfig, it: int) -> float:
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / max(1, cfg.warmup_iters)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, cfg: TrainConfig, ctx, device):
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()
    results = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        it = iter(loader)
        for _ in range(cfg.eval_iters):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with ctx:
                _, loss = raw(x, y)
            losses.append(loss.item())
        results[split] = float(np.mean(losses))
    raw.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = TrainConfig()

    if args.config:
        with open(args.config) as f:
            for k, v in yaml.safe_load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    if args.device:
        cfg.device = args.device

    # ---- DDP setup ---------------------------------------------------------
    using_ddp = is_ddp()
    if using_ddp:
        dist.init_process_group("nccl")
        local_rank = ddp_local_rank()
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        if cfg.device == "auto":
            device, hw_name = detect_device()
            log(f"Auto-detected: {hw_name} → {device}")
        else:
            device = cfg.device
            hw_name = device

    device_type = device.split(":")[0]  # "cuda", "mps", "cpu"

    # ---- dtype -------------------------------------------------------------
    cfg.dtype = get_dtype(device_type, cfg.dtype)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[cfg.dtype]

    # Autocast context: only on cuda/mps
    if device_type in ("cuda", "mps"):
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    else:
        ctx = contextlib.nullcontext()

    # ---- Reproducibility ---------------------------------------------------
    torch.manual_seed(cfg.seed + ddp_rank())
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ---- CPU thread count --------------------------------------------------
    if device_type == "cpu":
        torch.set_num_threads(os.cpu_count())
        cfg.compile = False  # compile rarely helps on CPU
        log(f"CPU mode: using {os.cpu_count()} threads, compile disabled")

    # ---- MPS ---------------------------------------------------------------
    if device_type == "mps":
        cfg.compile = False  # torch.compile on MPS is still unstable
        cfg.use_flash_attn = False
        log("MPS mode: compile disabled, using PyTorch SDPA")

    os.makedirs(cfg.out_dir, exist_ok=True)

    # ---- Data --------------------------------------------------------------
    num_workers = 0 if device_type == "cpu" else cfg.num_workers
    train_loader = get_dataloader(
        os.path.join(cfg.data_dir, "train.bin"),
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = get_dataloader(
        os.path.join(cfg.data_dir, "val.bin"),
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    # ---- Model -------------------------------------------------------------
    model_cfg = NimbusConfig(
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
        bias=cfg.bias,
        gradient_checkpointing=cfg.gradient_checkpointing,
        use_flash_attn=cfg.use_flash_attn,
        norm_type=cfg.norm_type,
    )

    iter_num = 0
    best_val_loss = float("inf")

    if args.resume:
        log(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model_cfg = NimbusConfig(**ckpt["model_config"])
        model = Nimbus(model_cfg)
        state = ckpt["model"]
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        iter_num = ckpt["iter_num"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
    else:
        model = Nimbus(model_cfg)

    model.to(device)

    # GradScaler: only useful for float16
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "float16" and device_type == "cuda"))

    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        device_type=device_type,
    )
    if args.resume and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    # ---- torch.compile -----------------------------------------------------
    if cfg.compile:
        log(f"Compiling model (mode={cfg.compile_mode})...")
        model = torch.compile(model, mode=cfg.compile_mode)
        log("Compilation done.")

    # ---- DDP wrap ----------------------------------------------------------
    if using_ddp:
        model = DDP(model, device_ids=[ddp_local_rank()])

    raw_model = model.module if using_ddp else model

    # ---- Training ----------------------------------------------------------
    world_size = ddp_world_size()
    tokens_per_iter = cfg.grad_accum_steps * cfg.batch_size * cfg.block_size * world_size
    log(f"Tokens per iter: {tokens_per_iter:,} | dtype: {cfg.dtype} | world: {world_size}x")

    train_iter = iter(train_loader)
    t0 = time.perf_counter()
    model.train()

    while iter_num < cfg.max_iters:
        lr = get_lr(cfg, iter_num)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval + checkpoint
        if iter_num % cfg.eval_interval == 0 and is_master():
            losses = estimate_loss(model, train_loader, val_loader, cfg, ctx, device)
            log(f"step {iter_num:6d}: train {losses['train']:.4f}  val {losses['val']:.4f}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": asdict(model_cfg),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": asdict(cfg),
                }
                path = os.path.join(cfg.out_dir, "ckpt.pt")
                torch.save(ckpt, path)
                log(f"  ✓ Saved {path} (val {best_val_loss:.4f})")

        # Forward + backward with gradient accumulation
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(cfg.grad_accum_steps):
            # Only sync gradients on the last micro-step (DDP efficiency)
            if using_ddp:
                model.require_backward_grad_sync = (micro_step == cfg.grad_accum_steps - 1)

            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

        if cfg.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        t1 = time.perf_counter()
        if iter_num % cfg.log_interval == 0:
            lossf = loss.item() * cfg.grad_accum_steps
            tok_s = tokens_per_iter / (t1 - t0)
            log(f"iter {iter_num:6d} | loss {lossf:.4f} | lr {lr:.2e} | {tok_s/1e3:.1f}k tok/s")
        t0 = t1

        iter_num += 1

    if using_ddp:
        dist.destroy_process_group()

    log("Training complete.")


if __name__ == "__main__":
    main()
