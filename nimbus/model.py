"""
Nimbus — core model class.

Wraps a pretrained transformer backbone with Nimbus identity,
chat formatting, streaming generation, and hardware-aware loading.
"""

from __future__ import annotations

import os
import sys
from typing import Iterator, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from threading import Thread

from .config import NimbusConfig


# HuggingFace repo for the pretrained backbone
_BACKBONE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Variant map — choose by size
VARIANTS = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7b":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "8b":   "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "14b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "32b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "70b":  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: list[int]):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class Nimbus:
    """
    Nimbus language model.

    Usage:
        nimbus = Nimbus.load()
        print(nimbus.chat("What is a transformer?"))
    """

    def __init__(self, model, tokenizer, config: NimbusConfig):
        self._model = model
        self._tokenizer = tokenizer
        self.config = config

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        variant: str = "1.5b",
        config: Optional[NimbusConfig] = None,
        local_path: Optional[str] = None,
    ) -> "Nimbus":
        """
        Load Nimbus. Downloads weights automatically on first run.

        Args:
            variant: Model size — "1.5b", "7b", "8b", "14b", "32b", "70b"
            config:  NimbusConfig (uses defaults if None)
            local_path: Load from a local directory instead of HuggingFace
        """
        cfg = config or NimbusConfig()
        repo = local_path or VARIANTS.get(variant, VARIANTS["1.5b"])

        print(f"Loading Nimbus ({variant.upper()})...")

        # Quantization
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

        # dtype
        if cfg.torch_dtype == "auto":
            dtype = "auto"
        else:
            dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(
            repo,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=dtype,
            device_map=cfg.device_map,
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )
        model.eval()

        print(f"Nimbus ready. ({_count_params(model):.1f}B parameters)")
        return cls(model, tokenizer, cfg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: str, history: list[dict] | None = None) -> list[dict]:
        messages = [{"role": "system", "content": self.config.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def _tokenize(self, messages: list[dict]) -> torch.Tensor:
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        return inputs.input_ids.to(self._model.device)

    def _stop_criteria(self) -> StoppingCriteriaList:
        stop_tokens = []
        for tok in ["<|end▁of▁sentence|>", "<|im_end|>", "</s>", "<|eot_id|>"]:
            ids = self._tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                stop_tokens.extend(ids)
        return StoppingCriteriaList([StopOnTokens(stop_tokens)])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def chat(
        self,
        prompt: str,
        history: list[dict] | None = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Single-turn or multi-turn chat. Returns the full response string."""
        messages = self._build_messages(prompt, history)
        input_ids = self._tokenize(messages)

        output = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=True,
            stopping_criteria=self._stop_criteria(),
            pad_token_id=self._tokenizer.eos_token_id,
        )

        new_tokens = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(
        self,
        prompt: str,
        history: list[dict] | None = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        """Streaming chat — yields text chunks as they are generated."""
        messages = self._build_messages(prompt, history)
        input_ids = self._tokenize(messages)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=True,
            streamer=streamer,
            pad_token_id=self._tokenizer.eos_token_id,
            stopping_criteria=self._stop_criteria(),
        )

        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()

    # ------------------------------------------------------------------
    # Fine-tuning helpers
    # ------------------------------------------------------------------

    def enable_training(self, gradient_checkpointing: bool = True):
        """Switch the model to training mode with optional gradient checkpointing."""
        self._model.train()
        if gradient_checkpointing:
            self._model.gradient_checkpointing_enable()
        return self._model

    def save(self, path: str):
        """Save fine-tuned weights and tokenizer."""
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        print(f"Saved to {path}")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Nimbus(params={_count_params(self._model):.1f}B, "
            f"device={self._model.device}, "
            f"dtype={next(self._model.parameters()).dtype})"
        )


def _count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e9
