import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams

def benchmark(llm: LLM, batch_size: int):
    num_seqs = batch_size
    max_input_len = 1
    max_ouput_len = 256

    prompt_token_ids = [[randint(0, 10000) for _ in  range(max_input_len)] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_ouput_len) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)  # warm-up
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")

def main():
    seed(0)
    path = os.path.expanduser("~/workspace/models/OLMoE-1B-7B-0924/")

    # enforce_eager = True to disable cuda graph
    llm = LLM(path, enforce_eager=True, max_model_len=4096)
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        print(f"Batch size: {batch_size}")
        benchmark(llm, batch_size)
        print()


if __name__ == "__main__":
    main()
