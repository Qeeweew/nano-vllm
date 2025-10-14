# ./scripts/quantize_moe.py

import os
import re
import json
import shutil
import argparse
from glob import glob
from collections import defaultdict
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# We need the C++ extension for the quantization function.
try:
    import nanovllm_ext
except ImportError:
    print("Error: Could not import 'nanovllm_ext'.")
    print("Please make sure you have built the C++ extension by running:")
    print("  cd csrc && python setup.py install")
    exit(1)


def quantize_moe_robust(model_path: str, output_path: str):
    """
    Robustly quantizes MoE expert weights for any model architecture.
    It dynamically identifies MoE layers, saves each layer's quantized weights
    into a separate file, and consolidates all non-expert weights into one file.
    This approach is fast, memory-efficient, and model-agnostic.
    """
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"Input model path is not a directory: {model_path}")

    if os.path.abspath(model_path) == os.path.abspath(output_path):
        raise ValueError("Output path cannot be the same as the input model path.")

    os.makedirs(output_path, exist_ok=True)
    print(f"Quantizing MoE model from '{model_path}' to '{output_path}' (Robust Mode)...")

    # 1. Load model configuration for dimensions
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    num_experts = config.get("num_experts")
    intermediate_size = config.get("moe_intermediate_size") or config.get("intermediate_size")
    
    if num_experts is None or intermediate_size is None:
        print("Warning: 'num_experts' or intermediate size not found in config.json. Assuming this is not an MoE model.")
        # Fallback: Just copy all files
        shutil.copytree(model_path, output_path, dirs_exist_ok=True)
        print("Finished: Copied original model files as no MoE configuration was detected.")
        return

    print(f"Model config loaded: {num_experts} experts, hidden_size={hidden_size}, intermediate_size={intermediate_size}")

    # 2. Scan all tensor keys to dynamically identify MoE layers and categorize all weights
    files = glob(os.path.join(model_path, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    expert_pattern = re.compile(r'model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight')
    
    expert_weights_by_layer = defaultdict(list)
    non_expert_weights_info = []
    moe_layer_indices = set()

    print("Scanning all tensor keys...")
    for file_path in files:
        with safe_open(file_path, "pt", "cpu") as f:
            for key in f.keys():
                match = expert_pattern.match(key)
                if match:
                    layer_idx = int(match.groups()[0])
                    moe_layer_indices.add(layer_idx)
                    expert_weights_by_layer[layer_idx].append((key, file_path))
                else:
                    non_expert_weights_info.append((key, file_path))
    
    if not moe_layer_indices:
        print("Warning: No expert weights found during scan. Assuming this is not an MoE model.")
        shutil.copytree(model_path, output_path, dirs_exist_ok=True)
        print("Finished: Copied original model files as no expert weights were found.")
        return

    print(f"Dynamically identified MoE blocks at layer indices: {sorted(list(moe_layer_indices))}")

    # 3. Process and save non-expert weights into a single consolidated file
    print("\n--- Consolidating non-expert weights ---")
    non_expert_tensors = {}
    with tqdm(total=len(non_expert_weights_info), desc="Loading non-expert weights") as pbar:
        for key, file_path in non_expert_weights_info:
            with safe_open(file_path, "pt", "cpu") as f:
                non_expert_tensors[key] = f.get_tensor(key)
            pbar.update(1)

    non_expert_output_file = os.path.join(output_path, "model-non-expert-weights.safetensors")
    print(f"Saving {len(non_expert_tensors)} non-expert tensors to {non_expert_output_file}...")
    save_file(non_expert_tensors, non_expert_output_file, metadata={"format": "pt"})
    del non_expert_tensors # Free memory

    # 4. Process MoE layers one by one and save to separate files
    for layer_idx in sorted(list(moe_layer_indices)):
        print(f"\n--- Processing MoE Layer {layer_idx} ---")
        
        gate_up_qs = torch.empty(num_experts, intermediate_size * 2, hidden_size, dtype=torch.int8)
        gate_up_d = torch.empty(num_experts, intermediate_size * 2, hidden_size // 32, dtype=torch.half)
        down_proj_qs = torch.empty(num_experts, hidden_size, intermediate_size, dtype=torch.int8)
        down_proj_d = torch.empty(num_experts, hidden_size, intermediate_size // 32, dtype=torch.half)
        
        layer_expert_weights = expert_weights_by_layer[layer_idx]
        with tqdm(total=len(layer_expert_weights), desc=f"Layer {layer_idx} experts") as pbar:
            for key, file_path in layer_expert_weights:
                with safe_open(file_path, "pt", "cpu") as f:
                    weight_fp32 = f.get_tensor(key).contiguous().to(torch.float32)
                
                match = expert_pattern.match(key)
                _, expert_idx_str, proj_name = match.groups()
                expert_idx = int(expert_idx_str)

                qs, d = nanovllm_ext.quantize_repack_weight(weight_fp32)
                del weight_fp32

                if proj_name == "gate_proj":
                    gate_up_qs[expert_idx, :intermediate_size, :] = qs
                    gate_up_d[expert_idx, :intermediate_size, :] = d
                elif proj_name == "up_proj":
                    gate_up_qs[expert_idx, intermediate_size:, :] = qs
                    gate_up_d[expert_idx, intermediate_size:, :] = d
                elif proj_name == "down_proj":
                    down_proj_qs[expert_idx] = qs
                    down_proj_d[expert_idx] = d
                
                del qs, d
                pbar.update(1)

        layer_tensors = {
            f"model.layers.{layer_idx}.mlp.gate_up_qs_stacked": gate_up_qs,
            f"model.layers.{layer_idx}.mlp.gate_up_d_stacked": gate_up_d,
            f"model.layers.{layer_idx}.mlp.down_proj_qs_stacked": down_proj_qs,
            f"model.layers.{layer_idx}.mlp.down_proj_d_stacked": down_proj_d,
        }
        
        output_file = os.path.join(output_path, f"model-layer-{layer_idx:02d}-quantized-experts.safetensors")
        print(f"Saving quantized experts for layer {layer_idx} to {output_file}...")
        save_file(layer_tensors, output_file, metadata={"format": "pt"})
        
        del gate_up_qs, gate_up_d, down_proj_qs, down_proj_d, layer_tensors

    # 5. Copy all other necessary non-tensor files
    print("\nCopying other necessary model files (config, tokenizer, etc.)...")
    for item in os.listdir(model_path):
        source_item = os.path.join(model_path, item)
        dest_item = os.path.join(output_path, item)
        if not item.endswith(".safetensors"):
            if os.path.isdir(source_item):
                shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
            else:
                shutil.copy2(source_item, dest_item)
    
    print("\n✅ Quantization complete!")
    print(f"The quantized model is ready at: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Robustly offline-quantizes MoE expert weights for nanovllm.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the directory containing the original Hugging Face MoE model (.safetensors files)."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the directory where the quantized model will be saved."
    )
    args = parser.parse_args()

    quantize_moe_robust(args.model_path, args.output_path)


if __name__ == "__main__":
    main()