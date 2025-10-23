import torch
from torch import nn

from nanovllm.layers.linear import Linear
from nanovllm.utils.context import get_context
import nanovllm_ext

class SparseMoeBlock(nn.Module):
    """
    Sparse MoE block that interfaces with a C++ MoEInfer object for expert
    computation on the CPU. The C++ object manages all expert weights.
    """
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        norm_top_k_prob: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.norm_top_k_prob = norm_top_k_prob

        # Gate runs on GPU to select experts
        self.gate = Linear(self.hidden_size, self.num_experts, bias=False)

        # Create and hold the handle to the C++ MoEInfer object
        self.moe_infer_handle = nanovllm_ext.create_moe_infer_handle(
            self.num_experts, self.hidden_size, self.intermediate_size
        )

    def forward_pre_expert(
        self,
        hidden_states: torch.Tensor,
        pinned_hidden: torch.Tensor,
        pinned_logits: torch.Tensor,
    ):
        """Phase 1 (GPU, Graphable): Routing and dispatching data to CPU pinned memory."""
        router_logits = self.gate(hidden_states)

        ctx = get_context()
        if ctx.moe_tracker and not ctx.is_prefill:
            with torch.no_grad():
                _, top_indices = torch.topk(router_logits, self.top_k, dim=-1)
                num_sequences = hidden_states.shape[0]
                sequence_indices = torch.arange(num_sequences, device=hidden_states.device)
                ctx.moe_tracker.log_activations(self.layer_idx, top_indices, sequence_indices)

        pinned_hidden.copy_(hidden_states, non_blocking=True)
        pinned_logits.copy_(router_logits, non_blocking=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Eager execution forward method demonstrating the two-phase computation."""
        orig_device = hidden_states.device
        num_tokens, H = hidden_states.shape

        ctx = get_context()
        pinned_hidden = ctx.get_pinned_buffer("moe_hidden", (num_tokens, H), hidden_states.dtype)
        pinned_logits = ctx.get_pinned_buffer("moe_logits", (num_tokens, self.num_experts), hidden_states.dtype)

        self.forward_pre_expert(hidden_states, pinned_hidden, pinned_logits)

        stream = torch.cuda.current_stream().cuda_stream
        keep_args = not ctx.is_prefill and ctx.is_graph_captured

        nanovllm_ext.launch_moe_cpu_task(
            pinned_hidden,
            pinned_logits,
            self.moe_infer_handle, # <-- Pass the handle instead of weights
            self.top_k,
            self.norm_top_k_prob,
            stream,
            keep_args
        )

        return pinned_hidden.to(orig_device, non_blocking=True)

    def expert_weight_loader(self, loaded_weight: torch.Tensor, expert_idx: int, proj_name: str):
        """
        Receives a single expert's FP32 weight, and passes it to the C++ object
        for online quantization and storage.
        """
        loaded_weight_cpu = loaded_weight.contiguous().to(device="cpu", dtype=torch.float32)
        nanovllm_ext.moe_infer_quantize_and_store(
            self.moe_infer_handle, expert_idx, proj_name, loaded_weight_cpu
        )

    def load_quantized_weights(self, gate_up_qs, gate_up_d, down_proj_qs, down_proj_d):
        """
        Receives pre-quantized stacked weights and passes them to the C++ object
        for direct storage.
        """
        nanovllm_ext.moe_infer_store_quantized(
            self.moe_infer_handle,
            gate_up_qs.contiguous().cpu(),
            gate_up_d.contiguous().cpu(),
            down_proj_qs.contiguous().cpu(),
            down_proj_d.contiguous().cpu()
        )