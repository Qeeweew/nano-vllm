# ./moe_analyzer.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import torch
import os
import pandas as pd # <-- Added dependency for data manipulation and CSV export

class MoEAnalyzer:
    """
    Analyzes MoE expert activations per-layer, focusing on expert distribution,
    the overlap of chosen expert sets between adjacent tokens, and the number of
    unique experts activated over a sliding window of tokens.
    """
    def __init__(self, moe_layer_indices: list[int], num_experts: int, top_k: int):
        self.moe_layer_indices = sorted(moe_layer_indices)
        self.num_experts = num_experts
        self.top_k = top_k
        self.reset()

    def reset(self):
        """Clears all collected data."""
        self.activations_by_layer = {idx: [] for idx in self.moe_layer_indices}
        # sequences_by_layer is crucial for temporal analysis (overlap, sliding window)
        self.sequences_by_layer = {idx: defaultdict(list) for idx in self.moe_layer_indices}
        self.overlap_counts_by_layer = {idx: [] for idx in self.moe_layer_indices}

    def log_activations(self, layer_idx: int, expert_indices_batch: torch.Tensor, seq_indices: torch.Tensor):
        """Logs expert activations for a given layer from a batch of data."""
        if layer_idx not in self.moe_layer_indices: return

        expert_indices_cpu = expert_indices_batch.cpu().tolist()
        seq_ids_cpu = seq_indices.cpu().tolist()

        # Log for overall distribution
        self.activations_by_layer[layer_idx].extend(
            expert for expert_list in expert_indices_cpu for expert in expert_list
        )

        # Log for temporal analysis (overlap and sliding window)
        sequences_in_progress = self.sequences_by_layer[layer_idx]
        for seq_id, current_experts in zip(seq_ids_cpu, expert_indices_cpu):
            if seq_id in sequences_in_progress and sequences_in_progress[seq_id]:
                previous_experts = sequences_in_progress[seq_id][-1]
                
                # Calculate the size of the intersection for adjacent token overlap
                overlap_size = len(set(previous_experts) & set(current_experts))
                self.overlap_counts_by_layer[layer_idx].append(overlap_size)
            
            # Store the sequence of expert sets for each sequence ID
            sequences_in_progress[seq_id].append(current_experts)

    def get_results(self):
        """Returns all collected data for serialization."""
        return {
            'moe_layer_indices': self.moe_layer_indices,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'activations_by_layer': self.activations_by_layer,
            'overlap_counts_by_layer': self.overlap_counts_by_layer,
            # Add sequences data needed for the new analysis
            'sequences_by_layer': {k: dict(v) for k, v in self.sequences_by_layer.items()},
        }

    @classmethod
    def from_results(cls, results: dict):
        """Creates an analyzer instance from previously collected results."""
        analyzer = cls(results['moe_layer_indices'], results['num_experts'], results['top_k'])
        analyzer.activations_by_layer = {int(k): v for k, v in results['activations_by_layer'].items()}
        analyzer.overlap_counts_by_layer = {int(k): v for k, v in results['overlap_counts_by_layer'].items()}
        # Load sequences data needed for the new analysis
        if 'sequences_by_layer' in results:
             analyzer.sequences_by_layer = {int(k): defaultdict(list, v) for k, v in results['sequences_by_layer'].items()}
        return analyzer
    
    # --- START OF NEW METHODS ---

    def _calculate_sliding_window_stats_for_layer(self, layer_idx: int) -> dict:
        """
        Calculates the number of unique experts activated across sliding windows
        of various sizes for a specific layer.
        """
        WINDOW_SIZES = range(2, 6) # Windows of 2, 3, ..., 8 consecutive tokens
        stats = {size: Counter() for size in WINDOW_SIZES}
        
        sequences = self.sequences_by_layer.get(layer_idx, {}).values()
        if not sequences:
            return {}

        for expert_sets_sequence in sequences:
            if not expert_sets_sequence: continue
            
            for window_size in WINDOW_SIZES:
                if len(expert_sets_sequence) < window_size:
                    continue
                # Slide the window across the sequence
                for i in range(len(expert_sets_sequence) - window_size + 1):
                    window_of_expert_sets = expert_sets_sequence[i : i + window_size]
                    
                    # Flatten the list of lists and find unique experts
                    unique_experts_in_window = set(expert for subset in window_of_expert_sets for expert in subset)
                    num_unique = len(unique_experts_in_window)
                    
                    # Record the count for this number of unique experts
                    stats[window_size][num_unique] += 1
        
        return {k: v for k, v in stats.items() if v} # Return only non-empty stats

    def plot_and_save_sliding_window_analysis(self, layer_idx: int, output_path: str):
        """
        Generates and saves a heatmap plot and a CSV file for the sliding
        window analysis of unique expert activations.
        """
        stats = self._calculate_sliding_window_stats_for_layer(layer_idx)
        if not stats:
            print(f"  - No sliding window data to analyze for layer {layer_idx}.")
            return
            
        # 1. Prepare data for DataFrame and CSV
        data_for_df = []
        for window_size, counts in stats.items():
            for num_unique, count in counts.items():
                data_for_df.append({
                    "Window Size": window_size,
                    "Unique Experts": num_unique,
                    "Count": count,
                })
        
        if not data_for_df:
            print(f"  - No valid sliding window entries found for layer {layer_idx}.")
            return

        df = pd.DataFrame(data_for_df)
        # Calculate probability within each window size group
        df['Probability'] = df.groupby('Window Size')['Count'].transform(lambda x: x / x.sum())
        
        # 2. Save the data to CSV
        csv_filename = f"sliding_window_uniqueness_layer_{layer_idx:02d}.csv"
        csv_filepath = os.path.join(output_path, csv_filename)
        df.to_csv(csv_filepath, index=False)
        print(f"  - Sliding window data for layer {layer_idx} saved to {csv_filename}.")
        
        # 3. Create a pivot table for the heatmap
        try:
            pivot_df = df.pivot_table(
                index='Window Size', 
                columns='Unique Experts', 
                values='Probability',
                fill_value=0
            )
        except Exception as e:
            print(f"Could not create pivot table for layer {layer_idx}: {e}")
            return

        # 4. Generate and save the heatmap plot
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            pivot_df,
            annot=True,      # Show probabilities on the heatmap
            fmt=".2f",       # Format annotations to two decimal places
            cmap="viridis",
            linewidths=.5
        )
        plt.title(f'Probability of Unique Expert Count in Sliding Windows (Layer {layer_idx})')
        plt.xlabel('Number of Unique Experts Activated')
        plt.ylabel('Consecutive Token Window Size')
        
        plot_filename = f"sliding_window_uniqueness_layer_{layer_idx:02d}.png"
        plot_filepath = os.path.join(output_path, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        print(f"  - Sliding window heatmap for layer {layer_idx} saved to {plot_filename}.")

    # --- END OF NEW METHODS ---

    def plot_all(self, output_path: str):
        """Generates and saves all available analysis plots and data files."""
        print("Generating plots and data files for each MoE layer...")
        for layer_idx in self.moe_layer_indices:
            print(f"\n--- Analyzing Layer {layer_idx} ---")
            self.plot_distribution_for_layer(layer_idx, output_path)
            self.plot_overlap_distribution_for_layer(layer_idx, output_path)
            # Call the new analysis function
            self.plot_and_save_sliding_window_analysis(layer_idx, output_path)
        print("\nAll analyses generated.")

    def plot_distribution_for_layer(self, layer_idx: int, output_path: str):
        activations = self.activations_by_layer.get(layer_idx)
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
        overlaps = self.overlap_counts_by_layer.get(layer_idx)
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