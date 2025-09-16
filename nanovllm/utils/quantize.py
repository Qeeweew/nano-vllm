import torch
from nanovllm.layers.quantized_linear import QuantizedLinear
from nanovllm.layers.linear import CPULinear, MergedCPULinear
from nanovllm.models.qwen3 import Qwen3MLP
from nanovllm.models.olmoe import OlmoeSparseMoeBlock, OlmoeMLP

@torch.no_grad()
def quantize_and_replace_mlp(model):
    """
    Finds all Qwen3MLP modules in the model, quantizes their weights,
    and replaces the original linear layers with QuantizedLinear layers.
    """
    print("Quantizing and replacing MLP layers for CPU acceleration...")
    for layer in model.model.layers:
        mlp = layer.mlp
        assert isinstance(mlp, Qwen3MLP), "This quantization is specific to Qwen3MLP"

        old_gate_up = mlp.gate_up_proj
        assert isinstance(old_gate_up, (CPULinear, MergedCPULinear))
        
        q_gate_up = QuantizedLinear(
            old_gate_up.weight.shape[1], 
            old_gate_up.weight.shape[0], 
            bias=(old_gate_up.bias is not None)
        )
        q_gate_up.quantize(old_gate_up.weight.data)
        if old_gate_up.bias is not None:
            q_gate_up.bias.data.copy_(old_gate_up.bias.data)
        
        # Replace the old layer with the new quantized one
        mlp.gate_up_proj = q_gate_up
        
        # --- Quantize and replace down_proj ---
        old_down = mlp.down_proj
        assert isinstance(old_down, (CPULinear, MergedCPULinear))
        
        q_down = QuantizedLinear(
            old_down.weight.shape[1], 
            old_down.weight.shape[0], 
            bias=(old_down.bias is not None)
        )
        q_down.quantize(old_down.weight.data)
        if old_down.bias is not None:
            q_down.bias.data.copy_(old_down.bias.data)
            
        # Replace the old layer with the new quantized one
        mlp.down_proj = q_down
    print("MLP quantization complete.")

@torch.no_grad()
def quantize_and_replace_moe_mlp(model):
    """
    Finds all OlmoeSparseMoeBlock modules, quantizes their expert MLPs' weights,
    replaces linear layers with QuantizedLinear, and then stacks the expert weights
    into single tensors for the high-performance C++ kernel.
    """
    print("Quantizing and replacing Olmoe expert MLP layers for CPU acceleration...")
    
    # Step 1: Quantize and replace linear layers within each expert
    for layer in model.model.layers:
        moe_block = layer.mlp
        if not isinstance(moe_block, OlmoeSparseMoeBlock):
            continue

        for i, expert_mlp in enumerate(moe_block.experts):
            assert isinstance(expert_mlp, OlmoeMLP), f"Expert {i} is not an OlmoeMLP"

            # --- Quantize and replace gate_up_proj ---
            old_gate_up = expert_mlp.gate_up_proj
            assert isinstance(old_gate_up, MergedCPULinear)
            
            q_gate_up = QuantizedLinear(
                old_gate_up.weight.shape[1], 
                old_gate_up.weight.shape[0], 
                bias=(old_gate_up.bias is not None)
            )
            q_gate_up.quantize(old_gate_up.weight.data)
            if old_gate_up.bias is not None:
                q_gate_up.bias.data.copy_(old_gate_up.bias.data)
            expert_mlp.gate_up_proj = q_gate_up
            
            # --- Quantize and replace down_proj ---
            old_down = expert_mlp.down_proj
            assert isinstance(old_down, CPULinear)
            
            q_down = QuantizedLinear(
                old_down.weight.shape[1], 
                old_down.weight.shape[0], 
                bias=(old_down.bias is not None)
            )
            q_down.quantize(old_down.weight.data)
            if old_down.bias is not None:
                q_down.bias.data.copy_(old_down.bias.data)
            expert_mlp.down_proj = q_down

    print("Olmoe expert MLP quantization complete.")
    print("Stacking expert weights for C++ kernel...")

    # Step 2: Stack the quantized weights from all experts and clean up
    for layer in model.model.layers:
        moe_block = layer.mlp
        if not isinstance(moe_block, OlmoeSparseMoeBlock):
            continue
        
        gate_up_qs, gate_up_d = [], []
        down_proj_qs, down_proj_d = [], []

        for expert_mlp in moe_block.experts:
            gate_up_qs.append(expert_mlp.gate_up_proj.weight_qs)
            gate_up_d.append(expert_mlp.gate_up_proj.weight_d)
            down_proj_qs.append(expert_mlp.down_proj.weight_qs)
            down_proj_d.append(expert_mlp.down_proj.weight_d)
        
        # Stack along a new dimension (dim=0)
        moe_block.gate_up_qs_stacked = torch.stack(gate_up_qs, dim=0)
        moe_block.gate_up_d_stacked = torch.stack(gate_up_d, dim=0)
        moe_block.down_proj_qs_stacked = torch.stack(down_proj_qs, dim=0)
        moe_block.down_proj_d_stacked = torch.stack(down_proj_d, dim=0)

        # Free the memory of individual expert modules as they are no longer needed
        del moe_block.experts
        moe_block.experts = None

    print("Expert weight stacking complete. Redundant expert modules have been removed.")

