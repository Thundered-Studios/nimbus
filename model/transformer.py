"""
Nimbus-1 Transformer Architecture
Decoder-only transformer, GPT-style. ~124M parameters.

Speed features:
  - Flash Attention 2 (flash-attn package) → PyTorch SDPA → manual fallback
  - Fused LayerNorm (apex) → PyTorch fallback
  - Gradient checkpointing for low-VRAM hardware
  - Weight tying (saves 38M params)
  - Residual projection scaling (GPT-2 paper)
"""

import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Optional fast kernels
# ---------------------------------------------------------------------------

try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import pad_input, unpad_input
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from apex.normalization import FusedLayerNorm as ApexLayerNorm
    HAS_APEX = False  # disabled by default; set True if apex is installed
except ImportError:
    HAS_APEX = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NimbusConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = True
    # Speed / memory knobs
    gradient_checkpointing: bool = False
    use_flash_attn: bool = True      # use flash-attn package if available
    norm_type: str = "layernorm"     # "layernorm" | "rmsnorm"


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm — faster than LayerNorm, no mean subtraction."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def build_norm(norm_type: str, dim: int, bias: bool) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if HAS_APEX:
        return ApexLayerNorm(dim)
    return nn.LayerNorm(dim, bias=bias)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: NimbusConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.use_flash_attn = config.use_flash_attn and HAS_FLASH_ATTN
        self.has_sdpa = hasattr(F, "scaled_dot_product_attention")

        if not self.use_flash_attn and not self.has_sdpa:
            # Manual causal mask fallback
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                    .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        def reshape(t):
            return t.view(B, T, self.n_head, self.head_dim)

        if self.use_flash_attn:
            # flash-attn expects (B, T, H, D)
            q, k, v = reshape(q), reshape(k), reshape(v)
            y = flash_attn_func(
                q.half(), k.half(), v.half(),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            ).to(x.dtype)
            y = y.contiguous().view(B, T, C)
        elif self.has_sdpa:
            # PyTorch 2.0 SDPA (fused, picks best kernel automatically)
            q = reshape(q).transpose(1, 2)  # (B, H, T, D)
            k = reshape(k).transpose(1, 2)
            v = reshape(v).transpose(1, 2)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)
        else:
            # Manual fallback
            q = reshape(q).transpose(1, 2)
            k = reshape(k).transpose(1, 2)
            v = reshape(v).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_drop(self.c_proj(y))


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: NimbusConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config: NimbusConfig):
        super().__init__()
        self.ln_1 = build_norm(config.norm_type, config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = build_norm(config.norm_type, config.n_embd, config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Nimbus Model
# ---------------------------------------------------------------------------

class Nimbus(nn.Module):
    def __init__(self, config: NimbusConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = build_norm(config.norm_type, config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Nimbus-1: {self.get_num_params()/1e6:.1f}M params | "
              f"flash={'pkg' if HAS_FLASH_ATTN else ('sdpa' if hasattr(F,'scaled_dot_product_attention') else 'manual')} | "
              f"grad_ckpt={config.gradient_checkpointing}")

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        for block in self.transformer.h:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay_params   = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_ok and device_type == "cuda"
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas,
            fused=use_fused,
        )

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx
