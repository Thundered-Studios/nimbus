"""
Nimbus-1 Auto-Launcher

Detects your hardware and launches training with the optimal config.
Just run:  python launch.py

For multi-GPU:  python launch.py --gpus 4
To override:    python launch.py --config configs/single_gpu_8gb.yaml
"""

import os
import sys
import argparse
import subprocess

import torch


def detect_hardware():
    info = {}

    # CUDA
    if torch.cuda.is_available():
        info["type"] = "cuda"
        info["count"] = torch.cuda.device_count()
        info["name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = props.total_memory / 1e9
        return info

    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["type"] = "mps"
        info["count"] = 1
        info["name"] = "Apple Silicon (MPS)"
        info["vram_gb"] = None
        return info

    # CPU
    import os as _os
    info["type"] = "cpu"
    info["count"] = 1
    info["name"] = "CPU"
    info["vram_gb"] = None
    info["threads"] = _os.cpu_count()
    return info


def pick_config(hw: dict, gpu_override: int | None) -> str:
    t = hw["type"]

    if t == "cpu":
        return "configs/cpu.yaml"

    if t == "mps":
        return "configs/apple_silicon.yaml"

    if t == "cuda":
        n_gpu = gpu_override or hw["count"]
        if n_gpu > 1:
            return "configs/multi_gpu.yaml"
        vram = hw["vram_gb"]
        if vram < 16:
            return "configs/single_gpu_8gb.yaml"
        return "configs/single_gpu_40gb.yaml"

    return "configs/cpu.yaml"


def build_command(cfg_path: str, hw: dict, args) -> list[str]:
    n_gpu = args.gpus or hw.get("count", 1)

    if hw["type"] == "cuda" and n_gpu > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpu}",
            "--nnodes=1",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:29500",
            "train.py",
            "--config", cfg_path,
        ]
    else:
        cmd = [sys.executable, "train.py", "--config", cfg_path]

    if args.resume:
        cmd += ["--resume", args.resume]

    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Override config file")
    p.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = p.parse_args()

    hw = detect_hardware()

    print("=" * 50)
    print("  Nimbus-1 Launcher")
    print("=" * 50)
    print(f"  Hardware : {hw['name']}")
    if hw["type"] == "cuda":
        print(f"  GPUs     : {hw['count']}x ({hw['vram_gb']:.1f} GB VRAM each)")
    print()

    cfg_path = args.config or pick_config(hw, args.gpus)
    print(f"  Config   : {cfg_path}")

    cmd = build_command(cfg_path, hw, args)
    print(f"  Command  : {' '.join(cmd)}")
    print("=" * 50)
    print()

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
