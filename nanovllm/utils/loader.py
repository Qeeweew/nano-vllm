import os
import re
from glob import glob
from collections import defaultdict
import torch
from torch import nn
from safetensors import safe_open
from tqdm.auto import tqdm

from nanovllm.layers.moe import SparseMoeBlock # Import for type checking


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # Regex for FP32 expert weights (online quantization)
    fp32_expert_pattern = re.compile(r'^(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')
    # Regex for pre-quantized expert weights (offline loading)
    quant_expert_pattern = re.compile(r'^(.*\.mlp)\.(gate_up_qs_stacked|gate_up_d_stacked|down_proj_qs_stacked|down_proj_d_stacked)$')

    files = glob(os.path.join(path, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    total_tensors = sum(len(safe_open(f, "pt", "cpu").keys()) for f in files)
    
    # --- State for loading pre-quantized weights ---
    # This dict will store tensors for each MoE layer until all 4 parts are loaded.
    # Key: mlp_prefix (e.g., 'model.layers.0.mlp'), Value: dict of tensors
    quantized_moe_layer_buffers = defaultdict(dict)

    with tqdm(total=total_tensors, desc=f"Loading weights from {os.path.basename(path)}", unit="tensor") as pbar:
        for file in files:
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    fp32_match = fp32_expert_pattern.match(weight_name)
                    quant_match = quant_expert_pattern.match(weight_name)

                    if fp32_match:
                        # --- SCENARIO 1: Online Quantization (FP32 expert weights) ---
                        mlp_prefix, expert_idx_str, proj_name = fp32_match.groups()
                        expert_idx = int(expert_idx_str)
                        moe_block = model.get_submodule(mlp_prefix)
                        assert isinstance(moe_block, SparseMoeBlock)
                        loaded_weight = f.get_tensor(weight_name)
                        moe_block.expert_weight_loader(loaded_weight, expert_idx, proj_name)

                    elif quant_match:
                        # --- SCENARIO 2: Offline Loading (Pre-quantized weights) ---
                        mlp_prefix, tensor_key = quant_match.groups()
                        buffer = quantized_moe_layer_buffers[mlp_prefix]
                        buffer[tensor_key] = f.get_tensor(weight_name)

                        # Check if all 4 tensors for this layer have been collected
                        if len(buffer) == 4:
                            moe_block = model.get_submodule(mlp_prefix)
                            assert isinstance(moe_block, SparseMoeBlock)
                            moe_block.load_quantized_weights(
                                buffer['gate_up_qs_stacked'],
                                buffer['gate_up_d_stacked'],
                                buffer['down_proj_qs_stacked'],
                                buffer['down_proj_d_stacked']
                            )
                            # Clear buffer for this layer to save memory
                            del quantized_moe_layer_buffers[mlp_prefix]

                    else:
                        # --- SCENARIO 3: All other weights (non-expert) ---
                        is_packed = False
                        for k, (v, shard_id) in packed_modules_mapping.items():
                            if k in weight_name:
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

    if quantized_moe_layer_buffers:
        # This should not happen if the quantized files are correct
        raise RuntimeError(f"Incomplete set of quantized MoE weights found for layers: {list(quantized_moe_layer_buffers.keys())}")