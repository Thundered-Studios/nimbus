# Nimbus model configuration
# Source: Qwen3 architecture (Apache 2.0)

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import RopeParameters


class NimbusConfig(PretrainedConfig):
    model_type = "nimbus"

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: dict = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    layer_types: list = None
    attention_dropout: float = 0.0
    pad_token_id: int = None
    bos_token_id: int = None
    eos_token_id: int = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
