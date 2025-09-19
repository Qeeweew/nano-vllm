import torch
from torch import nn

from nanovllm.layers.linear import Linear
from nanovllm.utils.context import get_context

# Import the C++ extension for quantized MoE computation
import nanovllm_ext
import torch_npu

class SparseMoeBlock(nn.Module):
    """
    Sparse MoE block with CPU offloading for experts.
    The router (gate) runs on GPU. After routing, all expert computations
    are performed on the CPU in float32 using a high-performance C++ kernel.
    Assumes that expert weights have been quantized and stacked offline by the
    `quantize_and_replace_moe_mlp` function.
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
        # Stacking all experts' weights into single tensors on the CPU.
        # This structure must match what the moe_q8_forward C++ kernel expects.
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Input hidden_states is expected to be 2D: [num_tokens, hidden_dim]
        if hidden_states.dim() != 2:
            raise ValueError(f"OlmoeSparseMoeBlock expects a 2D input, but got shape {hidden_states.shape}")

        orig_device = hidden_states.device
        B, H = hidden_states.shape

        
        # === 1. GPU Part: Routing ===
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts, _ = torch_npu.npu_moe_gating_top_k_softmax(router_logits, None, self.top_k)

        if self.norm_top_k_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # === 2. Transfer Data to CPU ===
        ctx = get_context()

        pinned_hidden = ctx.get_pinned_buffer("moe_hidden", (B, H), hidden_states.dtype)
        pinned_routing = ctx.get_pinned_buffer("moe_routing", (B, self.top_k), routing_weights.dtype)
        pinned_experts = ctx.get_pinned_buffer("moe_experts", (B, self.top_k), selected_experts.dtype)

        # 异步拷贝
        pinned_hidden.copy_(hidden_states, non_blocking=True)
        pinned_routing.copy_(routing_weights, non_blocking=True)
        pinned_experts.copy_(selected_experts, non_blocking=True)

        # === 3. CPU Part: Expert Computation via C++ Kernel ===
        # inplace operation to save memory
        pinned_hidden = nanovllm_ext.moe_q8_forward(
            pinned_hidden,
            pinned_routing,
            pinned_experts,
            self.gate_up_qs_stacked,
            self.gate_up_d_stacked,
            self.down_proj_qs_stacked,
            self.down_proj_d_stacked
        )
        
        # === 4. Transfer Result back to GPU ===
        return pinned_hidden.to(device=orig_device, non_blocking=True)

    def expert_weight_loader(self, loaded_weight: torch.Tensor, expert_idx: int, proj_name: str):
        """
        Receives a single expert's weight, quantizes it, and places it into the correct
        slice of the stacked parameter tensors.
        """
        # Ensure weight is on CPU and contiguous for the C++ extension
        loaded_weight = loaded_weight.contiguous().to(device="cpu", dtype=torch.float32)

        # Quantize the weight
        qs, d = nanovllm_ext.quantize_repack_weight(loaded_weight)

        # Place the quantized tensors into the correct slice of the stacked parameters
        if proj_name == "gate_proj":
            # This is the first half of the merged gate_up tensor
            self.gate_up_qs_stacked.data[expert_idx, :self.intermediate_size, :] = qs
            self.gate_up_d_stacked.data[expert_idx, :self.intermediate_size, :] = d
        elif proj_name == "up_proj":
            # This is the second half of the merged gate_up tensor
            self.gate_up_qs_stacked.data[expert_idx, self.intermediate_size:, :] = qs
            self.gate_up_d_stacked.data[expert_idx, self.intermediate_size:, :] = d
        elif proj_name == "down_proj":
            self.down_proj_qs_stacked.data[expert_idx] = qs
            self.down_proj_d_stacked.data[expert_idx] = d
        else:
            raise ValueError(f"Unknown expert projection name: {proj_name}")