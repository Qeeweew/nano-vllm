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
    and replaces the original CPU linear layers with QuantizedLinear layers.
    """
    print("Quantizing and replacing Olmoe expert MLP layers for CPU acceleration...")
    
    # Iterate through each decoder layer in the model
    for layer in model.model.layers:
        moe_block = layer.mlp
        if not isinstance(moe_block, OlmoeSparseMoeBlock):
            continue

        # Iterate through each expert in the MoE block
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
            
            # Replace the old layer with the new quantized one
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
                
            # Replace the old layer with the new quantized one
            expert_mlp.down_proj = q_down

    print("Olmoe expert MLP quantization complete.")
