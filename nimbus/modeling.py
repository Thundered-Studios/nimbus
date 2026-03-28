# Nimbus model architecture
# Source: Qwen3 (Apache 2.0) — renamed and adapted for Nimbus

from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils.generic import maybe_autocast

from .configuration import NimbusConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class NimbusRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# MLP (SwiGLU)
# ---------------------------------------------------------------------------

class NimbusMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Rotary Embeddings
# ---------------------------------------------------------------------------

class NimbusRotaryEmbedding(nn.Module):
    def __init__(self, config: NimbusConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        rope_params = config.rope_parameters or {}
        rope_type = rope_params.get("rope_type", "default")
        self.rope_type = rope_type

        if rope_type == "default" or rope_type not in ROPE_INIT_FUNCTIONS:
            inv_freq, self.attention_scaling = self._default_rope(config, device)
        else:
            inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def _default_rope(config, device=None):
        rope_params = config.rope_parameters or {}
        base = rope_params.get("rope_theta", 10000.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states   = repeat_kv(key,   module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output  = torch.matmul(attn_weights, value_states)
    attn_output  = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class NimbusAttention(nn.Module):
    def __init__(self, config: NimbusConfig, layer_idx: int):
        super().__init__()
        self.config    = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if config.layer_types else "full_attention"
        self.head_dim  = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling   = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = NimbusRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = NimbusRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape  = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Use Flash Attention / SDPA if available, else eager
        attn_fn = eager_attention_forward
        if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
            attn_fn = ALL_ATTENTION_FUNCTIONS.get_interface(
                getattr(self.config, "_attn_implementation", "eager"),
                eager_attention_forward,
            )

        attn_output, _ = attn_fn(
            self, query_states, key_states, value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class NimbusDecoderLayer(nn.Module):
    def __init__(self, config: NimbusConfig, layer_idx: int):
        super().__init__()
        self.self_attn              = NimbusAttention(config=config, layer_idx=layer_idx)
        self.mlp                    = NimbusMLP(config)
        self.input_layernorm        = NimbusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NimbusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        residual     = hidden_states
        hidden_states, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual      = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class NimbusPreTrainedModel(PreTrainedModel):
    config_class              = NimbusConfig
    base_model_prefix         = "model"
    supports_gradient_checkpointing = True
    _no_split_modules         = ["NimbusDecoderLayer"]
    _supports_flash_attn      = True
    _supports_sdpa            = True


class NimbusModel(NimbusPreTrainedModel):
    def __init__(self, config: NimbusConfig):
        super().__init__(config)
        self.embed_tokens  = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers        = nn.ModuleList([NimbusDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm          = NimbusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb    = NimbusRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in config.layer_types
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if position_ids is None:
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0) + past_len

        # Build causal masks
        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states      = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    causal_mask_mapping[layer_type],
                    position_ids,
                    past_key_values,
                    use_cache,
                    position_embeddings,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[layer_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------

class NimbusForCausalLM(NimbusPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model      = NimbusModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head    = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, logits_to_keep=0, **kwargs):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slc = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slc, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


__all__ = ["NimbusForCausalLM", "NimbusModel", "NimbusPreTrainedModel", "NimbusConfig"]
