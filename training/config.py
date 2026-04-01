"""Model configuration for Bangla-LLM 300M.

Defines the canonical model hyperparameters as a dataclass and provides
conversion to HuggingFace LlamaConfig for use with Transformers.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import LlamaConfig


@dataclass
class BanglaLLMConfig:
    """Canonical configuration for the Bangla-LLM 300M model.

    Architecture: LLaMA-style decoder-only transformer with GQA, SwiGLU FFN,
    RMSNorm, and RoPE positional encoding.

    Default hyperparameters yield ~306M parameters with weight tying.
    """

    d_model: int = 1024
    n_layers: int = 18
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ffn: int = 4096
    vocab_size: int = 32000
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    weight_tying: bool = True
    dropout: float = 0.0

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    def to_hf_config(self) -> LlamaConfig:
        """Convert to a HuggingFace LlamaConfig with matching parameters.

        Returns:
            LlamaConfig ready for LlamaForCausalLM instantiation.
        """
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.d_model,
            intermediate_size=self.d_ffn,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            num_key_value_heads=self.n_kv_heads,
            hidden_act="silu",
            max_position_embeddings=self.max_seq_len,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=self.weight_tying,
            rope_parameters={
                "rope_theta": self.rope_theta,
                "rope_type": "default",
            },
            attention_bias=False,
            attention_dropout=self.dropout,
            mlp_bias=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=None,
        )
