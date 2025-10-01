import torch
from torch import nn

from nanovllm.layers.linear import Linear
from nanovllm.utils.context import get_context

# Import the C++ extension for quantized MoE computation
import nanovllm_ext

class SparseMoeBlock(nn.Module):
    """
    Sparse MoE block with CPU offloading for experts.
    The router (gate) runs on GPU. After routing, all expert computations
    are performed on the CPU in float32 using a high-performance C++ kernel.
    Assumes that expert weights have been quantized and stacked offline.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        norm_top_k_prob: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.norm_top_k_prob = norm_top_k_prob

        # Gate runs on NPU to select experts
        self.gate = Linear(self.hidden_size, self.num_experts, bias=False)
        
        # --- Quantized weights stored on CPU ---
        self.gate_up_qs_stacked = nn.Parameter(torch.empty(
            self.num_experts, self.intermediate_size * 2, self.hidden_size, dtype=torch.int8, device="cpu"
        ), requires_grad=False)
        self.gate_up_d_stacked = nn.Parameter(torch.empty(
            self.num_experts, self.intermediate_size * 2, self.hidden_size // 32, dtype=torch.half, device="cpu"
        ), requires_grad=False)
        
        self.down_proj_qs_stacked = nn.Parameter(torch.empty(
            self.num_experts, self.hidden_size, self.intermediate_size, dtype=torch.int8, device="cpu"
        ), requires_grad=False)
        self.down_proj_d_stacked = nn.Parameter(torch.empty(
            self.num_experts, self.hidden_size, self.intermediate_size // 32, dtype=torch.half, device="cpu"
        ), requires_grad=False)

    def forward_pre_expert(
        self,
        hidden_states: torch.Tensor,
        pinned_hidden: torch.Tensor,
        pinned_logits: torch.Tensor,
    ):
        """
        Phase 1 (NPU, Graphable): Routing and dispatching data to CPU pinned memory.
        """
        # === 1. NPU Part: Calculate router logits ===
        router_logits = self.gate(hidden_states)
        
        # === 2. Transfer Data to CPU (Graphable Operation) ===
        # Initiate non-blocking copies of hidden states and logits to pinned CPU memory.
        pinned_hidden.copy_(hidden_states, non_blocking=True)
        pinned_logits.copy_(router_logits, non_blocking=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Eager execution forward method demonstrating the two-phase computation.
        """
        orig_device = hidden_states.device
        num_tokens, H = hidden_states.shape

        # Get pinned memory buffers for NPU -> CPU communication.
        ctx = get_context()
        pinned_hidden = ctx.get_pinned_buffer("moe_hidden", (num_tokens, H), hidden_states.dtype)
        pinned_logits = ctx.get_pinned_buffer("moe_logits", (num_tokens, self.num_experts), hidden_states.dtype)
        
        # --- Phase 1: NPU Computation ---
        self.forward_pre_expert(hidden_states, pinned_hidden, pinned_logits)
        
        # --- Synchronization Point ---
        torch.npu.synchronize()

        # --- Phase 2: CPU Computation ---

        # === 1. CPU Gating: Call C++ kernel which returns new CPU tensors ===
        routing_weights, selected_experts = nanovllm_ext.gating_top_k_softmax(
            pinned_logits,
            self.top_k,
            self.norm_top_k_prob
        )
        
        # === 2. CPU Expert Computation ===
        # The result is written in-place into pinned_hidden, which is then returned.
        nanovllm_ext.moe_q8_forward(
            pinned_hidden,
            routing_weights,
            selected_experts,
            self.gate_up_qs_stacked,
            self.gate_up_d_stacked,
            self.down_proj_qs_stacked,
            self.down_proj_d_stacked
        )

        # Asynchronously copy the final result back to the NPU.
        return pinned_hidden.to(orig_device, non_blocking=True)

    def expert_weight_loader(self, loaded_weight: torch.Tensor, expert_idx: int, proj_name: str):
        """
        Receives a single expert's weight, quantizes it, and places it into the correct
        slice of the stacked parameter tensors.
        """
        loaded_weight = loaded_weight.contiguous().to(device="cpu", dtype=torch.float32)

        qs, d = nanovllm_ext.quantize_repack_weight(loaded_weight)

        if proj_name == "gate_proj":
            self.gate_up_qs_stacked.data[expert_idx, :self.intermediate_size, :] = qs
            self.gate_up_d_stacked.data[expert_idx, :self.intermediate_size, :] = d
        elif proj_name == "up_proj":
            self.gate_up_qs_stacked.data[expert_idx, self.intermediate_size:, :] = qs
            self.gate_up_d_stacked.data[expert_idx, self.intermediate_size:, :] = d
        elif proj_name == "down_proj":
            self.down_proj_qs_stacked.data[expert_idx] = qs
            self.down_proj_d_stacked.data[expert_idx] = d
        else:
            raise ValueError(f"Unknown expert projection name: {proj_name}")