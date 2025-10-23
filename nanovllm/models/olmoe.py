# FILE: ./nanovllm/models/olmoe.py
import torch
from torch import nn
import torch.nn.functional as F
from transformers import OlmoeConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.moe import SparseMoeBlock
from nanovllm.layers.linear import QKVParallelLinear, Linear, CPULinear, MergedCPULinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context

class OlmoeAttention(nn.Module):
    def __init__(
        self,
        config: OlmoeConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        self.scaling = self.head_dim ** -0.5

        # In Olmoe, q_proj, k_proj, v_proj are separate. We merge them for efficiency
        # and handle the loading via packed_modules_mapping.
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
        )
        self.o_proj = Linear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Olmoe has specific q_norm and k_norm
        self.q_norm = RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.num_kv_heads * self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Apply Q/K norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

class OlmoeMLP(nn.Module):
    """A standard MLP for one expert. This will run on CPU."""
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        # Use MergedCPULinear for gate_proj and up_proj
        self.gate_up_proj = MergedCPULinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = CPULinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor):
        # This function expects a CPU tensor and returns a CPU tensor.
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class OlmoeDecoderLayer(nn.Module):
    def __init__(self, config: OlmoeConfig) -> None:
        super().__init__()
        self.self_attn = OlmoeAttention(config)
        self.mlp = SparseMoeBlock(
            layer_idx=-1,
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.intermediate_size, # Olmoe uses intermediate_size for experts
            norm_top_k_prob=getattr(config, "norm_top_k_prob", False)
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # --- Self-Attention Block ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        # --- MoE MLP Block ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Note: self.mlp is the SparseMoeBlock which returns only the hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class OlmoeModel(nn.Module):
    def __init__(self, config: OlmoeConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([OlmoeDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            # The layer now handles its own residual connections internally
            hidden_states = layer(positions, hidden_states)
        # Final LayerNorm after all layers
        hidden_states = self.norm(hidden_states)
        return hidden_states


class OlmoeForCausalLM(nn.Module):
    # This mapping helps the weight loader to correctly place weights from separate
    # files (e.g., q_proj, k_proj) into our merged layers (e.g., qkv_proj).
    packed_modules_mapping = {
        # Attention layers
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config: OlmoeConfig) -> None:
        super().__init__()
        self.model = OlmoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)