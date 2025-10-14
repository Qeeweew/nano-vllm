# ./run_moe_analysis.py

import os
import argparse
from datasets import load_dataset, IterableDataset
from tqdm import tqdm

from nanovllm import LLM, SamplingParams
from moe_analyzer import MoEAnalyzer

def main(args):
    # 1. Initialize the LLM engine with your MoE model
    print(f"Loading MoE model from: {args.model_path}")
    llm = LLM(args.model_path, enforce_eager=True) # Use enforce_eager for simplicity
    
    # --- MODIFIED SECTION: DATASET LOADING ---
    # 2. Load the ShareGPT dataset using streaming to avoid large downloads
    print("Loading dataset 'shibing624/sharegpt_gpt4' in streaming mode...")
    # Using streaming=True is highly recommended for large datasets like ShareGPT
    dataset = load_dataset("shibing624/sharegpt_gpt4", split="train", streaming=True)
    
    # Take the desired number of samples from the stream
    dialogues: IterableDataset = dataset.take(args.num_samples)
    # --- END OF MODIFIED SECTION ---


    # --- MODIFIED SECTION: PROMPT FORMATTING ---
    # 3. Format prompts from the multi-turn ShareGPT conversations
    print("Formatting prompts from ShareGPT conversations...")
    prompts = []
    
    # The role names in ShareGPT ('human', 'gpt') need to be mapped to what the tokenizer expects ('user', 'assistant')
    role_mapping = {"human": "user", "gpt": "assistant"}

    for item in tqdm(dialogues, desc="Formatting prompts", total=args.num_samples):
        conversation = item.get('conversations', [])
        if not conversation:
            continue

        messages = []
        # We process the conversation turn by turn
        for turn in conversation:
            sender = turn.get('from')
            message_text = turn.get('value')
            
            if sender in role_mapping and message_text:
                role = role_mapping[sender]
                messages.append({"role": role, "content": message_text})
                if role == "user":
                    break
        
        # We only want to create a prompt if the conversation ends with a user turn,
        # which is the natural point for the assistant to respond.
        if messages and messages[-1]['role'] == 'user':
            try:
                # Apply the chat template to the entire conversation history
                prompt = llm.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True, # This adds the assistant's prompt prefix (e.g., "<|assistant|>")
                )
                prompts.append(prompt)
            except Exception as e:
                print(f"Warning: Could not apply chat template to a conversation. Error: {e}")
    
    if not prompts:
        print("Error: No valid prompts could be created from the dataset samples. The conversations might not end with a 'human' turn.")
        return
    # --- END OF MODIFIED SECTION ---

    # 4. Start tracking
    print(f"\nStarting MoE activation tracking on {len(prompts)} valid prompts...")
    llm.start_moe_tracking()

    # 5. Run generation
    sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    print(f"Generating completions...")
    _ = llm.generate(prompts, sampling_params, use_tqdm=True)

    # 6. Stop tracking and get results
    print("Stopping MoE tracking and collecting results...")
    results = llm.stop_moe_tracking()

    if not results:
        print("Error: Failed to retrieve tracking results.")
        return

    # 7. Analyze and visualize
    print("Analyzing results and generating plots for each MoE layer...")
    os.makedirs(args.output_dir, exist_ok=True)
    analyzer = MoEAnalyzer.from_results(results)

    # --- THIS IS THE ONLY CHANGE IN THIS FILE ---
    # Call the new master plotting function
    analyzer.plot_all(args.output_dir)
    # ---------------------------------------------

    print(f"\n✅ Layer-wise analysis complete!")
    print(f"Plots saved in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MoE expert usage in nanovllm using the ShareGPT dataset.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the quantized MoE model directory (e.g., Qwen3-30B-A3B-quantized).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128, # Increased default as we might skip some samples
        help="Number of dialogue samples to attempt to process from the dataset.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./moe_analysis_results_sharegpt", # New default output dir
        help="Directory to save the analysis plots.",
    )
    args = parser.parse_args()
    main(args)