"""
Nimbus configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NimbusConfig:
    # Model identity
    model_name: str = "Nimbus-1"
    version: str = "1.0.0"

    # Generation defaults
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1

    # System
    torch_dtype: str = "auto"           # "auto" | "float16" | "bfloat16" | "float32"
    device_map: str = "auto"
    load_in_4bit: bool = False          # 4-bit quantization (runs on 6GB+ VRAM / CPU)
    load_in_8bit: bool = False          # 8-bit quantization

    # Context
    max_context_length: int = 32768

    # Chat template
    system_prompt: str = (
        "You are Nimbus, a helpful and intelligent AI assistant "
        "created by Thundered Studios."
    )
