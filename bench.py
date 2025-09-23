import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams

def benchmark(llm: LLM, batch_size: int, input_len: int = 1, output_len: int = 128):
    num_seqs = batch_size

    prompt_token_ids = [[randint(0, 10000) for _ in  range(input_len)] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)  # warm-up
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = num_seqs * (input_len + output_len)
    throughput = total_tokens / t
    print(f"Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")

def main():
    seed(0)
    path = os.path.expanduser("/root/autodl-tmp/OLMoE-1B-7B-0924/")
    # enforce_eager = True to disable cuda graph
    llm = LLM(path, enforce_eager=True, max_model_len=4096)

    print("Benchmarking Decoding Throughput...")
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        print(f"Batch size: {batch_size}")
        benchmark(llm, batch_size, 1, 128)
        print()
    print("Benchmarking Prefill Throughput...")

    for input_len in [16, 32, 64, 128, 256, 512, 1024]:
        print(f"Input length: {input_len}")
        benchmark(llm, 1, input_len, 1)
        print()

if __name__ == "__main__":
    main()
