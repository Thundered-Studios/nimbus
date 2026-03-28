"""
Nimbus — model loader.

Uses Nimbus's own architecture (sourced from Qwen3, renamed throughout).
Loads pretrained weights directly into NimbusForCausalLM.
"""

from __future__ import annotations

from typing import Iterator, Optional
from threading import Thread

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from .modeling import NimbusForCausalLM
from .configuration import NimbusConfig as _NimbusConfig


VARIANTS = {
    "0.6b": ("Qwen/Qwen3-0.6B",   _NimbusConfig),
    "1.7b": ("Qwen/Qwen3-1.7B",   _NimbusConfig),
    "4b":   ("Qwen/Qwen3-4B",     _NimbusConfig),
    "8b":   ("Qwen/Qwen3-8B",     _NimbusConfig),
    "14b":  ("Qwen/Qwen3-14B",    _NimbusConfig),
    "32b":  ("Qwen/Qwen3-32B",    _NimbusConfig),
}


class NimbusConfig:
    def __init__(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        system_prompt: str = "You are Nimbus, a helpful and intelligent AI assistant created by Thundered Studios.",
    ):
        self.max_new_tokens     = max_new_tokens
        self.temperature        = temperature
        self.top_p              = top_p
        self.top_k              = top_k
        self.repetition_penalty = repetition_penalty
        self.torch_dtype        = torch_dtype
        self.device_map         = device_map
        self.load_in_4bit       = load_in_4bit
        self.load_in_8bit       = load_in_8bit
        self.system_prompt      = system_prompt


class Nimbus:
    def __init__(self, model, tokenizer, config: NimbusConfig):
        self._model     = model
        self._tokenizer = tokenizer
        self.config     = config

    @classmethod
    def load(
        cls,
        variant: str = "4b",
        config: Optional[NimbusConfig] = None,
        local_path: Optional[str] = None,
    ) -> "Nimbus":
        cfg  = config or NimbusConfig()
        repo, _ = VARIANTS.get(variant, VARIANTS["4b"])
        src  = local_path or repo

        print(f"Loading Nimbus {variant.upper()}...")

        # Quantization config
        quant_cfg = None
        if cfg.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif cfg.load_in_8bit:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        dtype = "auto" if cfg.torch_dtype == "auto" else getattr(torch, cfg.torch_dtype, torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=False)

        # Load the pretrained weights into NimbusForCausalLM directly
        model = NimbusForCausalLM.from_pretrained(
            src,
            torch_dtype=dtype,
            device_map=cfg.device_map,
            quantization_config=quant_cfg,
            # Map Qwen3 config keys → NimbusConfig
            config=_NimbusConfig.from_pretrained(src),
        )
        model.eval()

        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"Nimbus ready — {n_params:.1f}B parameters")

        return cls(model, tokenizer, cfg)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: str, history: list[dict] | None = None) -> list[dict]:
        messages = [{"role": "system", "content": self.config.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def _tokenize(self, messages: list[dict]) -> torch.Tensor:
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._tokenizer(text, return_tensors="pt").input_ids.to(self._model.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def chat(
        self,
        prompt: str,
        history: list[dict] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        input_ids = self._tokenize(self._build_messages(prompt, history))
        output = self._model.generate(
            input_ids,
            max_new_tokens    = max_new_tokens or self.config.max_new_tokens,
            temperature       = temperature    or self.config.temperature,
            top_p             = top_p          or self.config.top_p,
            top_k             = self.config.top_k,
            repetition_penalty= self.config.repetition_penalty,
            do_sample         = True,
            pad_token_id      = self._tokenizer.eos_token_id,
        )
        new_tokens = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(
        self,
        prompt: str,
        history: list[dict] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> Iterator[str]:
        input_ids = self._tokenize(self._build_messages(prompt, history))
        streamer  = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            input_ids         = input_ids,
            max_new_tokens    = max_new_tokens or self.config.max_new_tokens,
            temperature       = temperature    or self.config.temperature,
            top_p             = top_p          or self.config.top_p,
            top_k             = self.config.top_k,
            repetition_penalty= self.config.repetition_penalty,
            do_sample         = True,
            streamer          = streamer,
            pad_token_id      = self._tokenizer.eos_token_id,
        )
        Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True).start()
        yield from streamer

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def enable_training(self, gradient_checkpointing: bool = True):
        self._model.train()
        if gradient_checkpointing:
            self._model.gradient_checkpointing_enable()
        return self._model

    def save(self, path: str):
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        print(f"Saved to {path}")

    def __repr__(self):
        n = sum(p.numel() for p in self._model.parameters()) / 1e9
        return f"Nimbus({n:.1f}B params | {next(self._model.parameters()).dtype} | {self._model.device})"
