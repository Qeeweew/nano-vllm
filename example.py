import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/workspace/models/OLMoE-1B-7B-0924/")
    # tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=16)
    prompts = [
        "Once upon a time in the west,",
    ]
    # prompts = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "user", "content": prompt}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    #     for prompt in prompts
    # ]
    print("Warm-up run...")
    _ = llm.generate(prompts, sampling_params)

    print("Warm-up finished. Starting profiling...")
    
    # 创建 profiler 上下文
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], # 同时记录 CPU 和 GPU
        record_shapes=True,        # 记录张量的形状
        profile_memory=True,       # 记录内存分配/释放
        with_stack=True,           # 记录Python调用堆栈，方便溯源
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof_log') # 将结果保存到目录
    ) as prof:
        # 在这里执行你想要分析的代码
        with record_function("model_inference"): # 给这段代码起个名字，方便在Trace中查找
            outputs = llm.generate(prompts, sampling_params)

    # --- Profiling部分结束 ---

    # 打印 profiler 总结信息到控制台
    print("\n--- Profiler Summary (CPU+CUDA) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    print("\n--- Profiler Summary (CPU) ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    print(f"\nProfiling results saved to ./prof_log. Run 'tensorboard --logdir ./prof_log' to view.")

    # outputs = llm.generate(prompts, sampling_params)

    # 打印正常输出
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
