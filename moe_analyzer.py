import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import torch
import os

class MoEAnalyzer:
    """
    Analyzes MoE expert activations per-layer, focusing on expert distribution
    and the overlap of chosen expert sets between adjacent tokens.
    """
    def __init__(self, moe_layer_indices: list[int], num_experts: int, top_k: int):
        self.moe_layer_indices = sorted(moe_layer_indices)
        self.num_experts = num_experts
        self.top_k = top_k
        self.reset()

    def reset(self):
        """Clears all collected data."""
        self.activations_by_layer = {idx: [] for idx in self.moe_layer_indices}
        self.sequences_by_layer = {idx: defaultdict(list) for idx in self.moe_layer_indices}
        self.overlap_counts_by_layer = {idx: [] for idx in self.moe_layer_indices}

    def log_activations(self, layer_idx: int, expert_indices_batch: torch.Tensor, seq_indices: torch.Tensor):
        if layer_idx not in self.moe_layer_indices: return

        expert_indices_cpu = expert_indices_batch.cpu().tolist()
        seq_ids_cpu = seq_indices.cpu().tolist()

        # Log for overall distribution
        self.activations_by_layer[layer_idx].extend(
            expert for expert_list in expert_indices_cpu for expert in expert_list
        )

        # Log for overlap analysis
        sequences_in_progress = self.sequences_by_layer[layer_idx]
        for seq_id, current_experts in zip(seq_ids_cpu, expert_indices_cpu):
            if seq_id in sequences_in_progress and sequences_in_progress[seq_id]:
                previous_experts = sequences_in_progress[seq_id][-1]
                
                # Calculate the size of the intersection between the two sets of experts
                overlap_size = len(set(previous_experts) & set(current_experts))
                self.overlap_counts_by_layer[layer_idx].append(overlap_size)
            
            sequences_in_progress[seq_id].append(current_experts)

    def get_results(self):
        return {
            'moe_layer_indices': self.moe_layer_indices,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'activations_by_layer': self.activations_by_layer,
            'overlap_counts_by_layer': self.overlap_counts_by_layer,
        }

    @classmethod
    def from_results(cls, results: dict):
        analyzer = cls(results['moe_layer_indices'], results['num_experts'], results['top_k'])
        analyzer.activations_by_layer = {int(k): v for k, v in results['activations_by_layer'].items()}
        analyzer.overlap_counts_by_layer = {int(k): v for k, v in results['overlap_counts_by_layer'].items()}
        return analyzer

    def plot_all(self, output_path: str):
        print("Generating plots for each MoE layer...")
        for layer_idx in self.moe_layer_indices:
            self.plot_distribution_for_layer(layer_idx, output_path)
            self.plot_overlap_distribution_for_layer(layer_idx, output_path)
        print("All plots generated.")

    def plot_distribution_for_layer(self, layer_idx: int, output_path: str):
        # This function is unchanged
        activations = self.activations_by_layer[layer_idx]
        if not activations: return
        counts = Counter(activations)
        labels, values = zip(*sorted(counts.items()))
        plt.figure(figsize=(16, 8))
        plt.bar(labels, values, color=sns.color_palette("viridis", len(labels)))
        plt.xlabel('Expert ID')
        plt.ylabel('Activation Frequency')
        plt.title(f'MoE Expert Activation Distribution (Layer {layer_idx})')
        plt.xticks(np.arange(0, self.num_experts, step=max(1, self.num_experts // 16)))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        filepath = os.path.join(output_path, f"expert_distribution_layer_{layer_idx:02d}.png")
        plt.savefig(filepath)
        plt.close()
        print(f"  - Distribution plot for layer {layer_idx} saved.")

    def plot_overlap_distribution_for_layer(self, layer_idx: int, output_path: str):
        """Plots the distribution of expert set overlaps between adjacent tokens."""
        overlaps = self.overlap_counts_by_layer[layer_idx]
        if not overlaps:
            print(f"  - No overlap data to plot for layer {layer_idx}.")
            return

        counts = Counter(overlaps)
        # Ensure all possible overlap values (0 to top_k) are represented
        labels = list(range(self.top_k + 1))
        values = [counts.get(i, 0) for i in labels]

        # Normalize to probabilities
        total_transitions = sum(values)
        if total_transitions > 0:
            probabilities = [v / total_transitions for v in values]
        else:
            probabilities = [0.0] * len(labels)

        plt.figure(figsize=(10, 6))
        plt.bar(labels, probabilities, color=sns.color_palette("magma", len(labels)))
        plt.xlabel(f'Number of Overlapping Experts (out of Top-{self.top_k})')
        plt.ylabel('Probability')
        plt.title(f'Expert Set Overlap Between Adjacent Tokens (Layer {layer_idx})')
        plt.xticks(labels)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        filepath = os.path.join(output_path, f"expert_overlap_distribution_layer_{layer_idx:02d}.png")
        plt.savefig(filepath)
        plt.close()
        print(f"  - Overlap distribution plot for layer {layer_idx} saved.")