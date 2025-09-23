import os
import torch
# from torch.profiler import profile, record_function, ProfilerActivity
import torch_npu
from torch_npu.profiler import profile, ProfilerActivity
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/root/autodl-tmp/OLMoE-1B-7B-0924/")
    # tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0, max_tokens=4)
    prompts = [
        "Bitcoin is"
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
    _ = llm.generate(["test"], SamplingParams(temperature=0, max_tokens=1))

    # print("Warm-up finished. Starting profiling...")
    
    # # 创建 profiler 上下文
    experimental_config = torch_npu.profiler._ExperimentalConfig(
	    export_type=torch_npu.profiler.ExportType.Text,
	    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
	    msprof_tx=False,
	    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
	    l2_cache=False,
	    op_attr=False,
	    data_simplification=False,
	    record_op_args=False
    )

    with torch_npu.profiler.profile(
            activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU
                    ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_log"),
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            with_modules=True,
            with_flops=False,
            experimental_config=experimental_config) as prof:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            prof.step()
    # --- Profiling部分结束 ---

    # 打印 profiler 总结信息到控制台

    # outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # 打印正常输出
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
