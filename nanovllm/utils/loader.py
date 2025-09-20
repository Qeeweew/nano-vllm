import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from tqdm.auto import tqdm


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # Regex to parse expert weight names, e.g., 'model.layers.0.mlp.experts.1.gate_proj.weight'
    expert_pattern = re.compile(r'^(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')

    files = glob(os.path.join(path, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    # Step 1: Pre-scan all files to get the total number of tensors for the progress bar
    total_tensors = 0
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            total_tensors += len(f.keys())

    # Step 2: Load weights with a progress bar
    with tqdm(total=total_tensors, desc=f"Loading weights from {os.path.basename(path)}", unit="tensor") as pbar:
        for file in files:
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    # --- Custom logic for MoE expert weights ---
                    match = expert_pattern.match(weight_name)
                    if match:
                        mlp_prefix, expert_idx_str, proj_name = match.groups()
                        expert_idx = int(expert_idx_str)
                        
                        # Get the OlmoeSparseMoeBlock module
                        moe_block = model.get_submodule(mlp_prefix)
                        
                        # Load the tensor from the file
                        loaded_weight = f.get_tensor(weight_name)
                        
                        # Call the custom loader which will quantize and place the weight
                        moe_block.expert_weight_loader(loaded_weight, expert_idx, proj_name)
                        pbar.update(1)
                        continue

                    # --- Original logic for other weights ---
                    is_packed = False
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            is_packed = True
                            break
                    
                    if not is_packed:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                    
                    pbar.update(1)