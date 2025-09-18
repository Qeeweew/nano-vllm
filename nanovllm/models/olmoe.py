# FILE: ./nanovllm/models/olmoe.py
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import OlmoeConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear, CPULinear, MergedCPULinear, ReplicatedLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class OlmoeAttention(nn.Module):
    def __init__(
        self,
        config: OlmoeConfig,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
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
        self.o_proj = RowParallelLinear(
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

import nanovllm_ext

class OlmoeSparseMoeBlock(nn.Module):
    """
    Sparse MoE block with CPU offloading for experts.
    The router (gate) runs on GPU. After routing, all expert computations
    are performed on the CPU in float32 using a high-performance C++ kernel.
    Assumes that expert weights have been quantized and stacked offline by the
    `quantize_and_replace_moe_mlp` function.
    """
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.norm_top_k_prob = getattr(config, "norm_top_k_prob", False)

        # Gate runs on GPU to select experts
        self.gate = ReplicatedLinear(config.hidden_size, self.num_experts, bias=False)

        # This list will be populated, used for quantization, and then removed.
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])

        # Buffers for stacked quantized weights. These will be populated by the quantization utility.
        self.register_buffer("gate_up_qs_stacked", torch.empty(0, dtype=torch.int8))
        self.register_buffer("gate_up_d_stacked", torch.empty(0, dtype=torch.half))
        self.register_buffer("down_proj_qs_stacked", torch.empty(0, dtype=torch.int8))
        self.register_buffer("down_proj_d_stacked", torch.empty(0, dtype=torch.half))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Input hidden_states is expected to be 2D: [num_tokens, hidden_dim]
        if hidden_states.dim() != 2:
            raise ValueError(f"OlmoeSparseMoeBlock expects a 2D input, but got shape {hidden_states.shape}")

        orig_device = hidden_states.device
        orig_dtype = hidden_states.dtype
        
        # === 1. GPU Part: Routing ===
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_top_k_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # === 2. Transfer Data to CPU ===
        hidden_states_cpu = hidden_states.to(device="cpu")
        routing_weights_cpu = routing_weights.to(device="cpu", dtype=orig_dtype)
        selected_experts_cpu = selected_experts.to(device="cpu", dtype=torch.int32)

        # === 3. CPU Part: Expert Computation via C++ Kernel ===
        final_hidden_states_cpu = nanovllm_ext.moe_q8_forward(
            hidden_states_cpu,
            routing_weights_cpu,
            selected_experts_cpu,
            self.gate_up_qs_stacked,
            self.gate_up_d_stacked,
            self.down_proj_qs_stacked,
            self.down_proj_d_stacked
        )
        
        # === 4. Transfer Result back to GPU ===
        return final_hidden_states_cpu.to(device=orig_device, non_blocking=True)

class OlmoeDecoderLayer(nn.Module):
    def __init__(self, config: OlmoeConfig) -> None:
        super().__init__()
        self.self_attn = OlmoeAttention(config)
        self.mlp = OlmoeSparseMoeBlock(config)
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
        # Note: self.mlp is the OlmoeSparseMoeBlock which returns only the hidden_states
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
        # Expert MLPs
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
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