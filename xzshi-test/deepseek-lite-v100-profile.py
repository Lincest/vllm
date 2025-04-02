# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# check for flash attn (v100 is not supported)
"""
ERROR:
[rank0]: RuntimeError: FlashAttention only supports Ampere GPUs or newer.
"""
import os
os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
print(f"VLLM_ATTENTION_BACKEND: {os.getenv('VLLM_ATTENTION_BACKEND')}, VLLM_USE_TRITON_FLASH_ATTN: {os.getenv('VLLM_USE_TRITON_FLASH_ATTN')}")


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "The model name is",
    "The largest ocean in the world is",
    "The formula for water is",
    "The speed of light is approximately",
    "The author of 'War and Peace' is",
    "The currency of Japan is",
    "The tallest mountain on Earth is",
    "The number of planets in our solar system is",
    "The chemical symbol for gold is",
    "The inventor of the telephone was",
    "The most populated country in the world is",
    "The year World War II ended was"
]
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
# ]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4,seed=410)

# https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/tree/main
# 大小：31.41 GB
llm = LLM(
    model="/home/sugon/xzshi/vllm/models/deepseek-v2-lite",
    cpu_offload_gb=20,
    max_model_len=512,
    gpu_memory_utilization=0.5, # DS_EXPERT_OFFLOAD: 0.5, 如果没有就 0.8
    enforce_eager=True,
    trust_remote_code=True,dtype="float32"
)


import torch.profiler as profiler
import time
import torch
def run_with_profiler():
    # 使用PyTorch Profiler进行分析
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler.schedule(wait=1, warmup=1, active=1),
        on_trace_ready=profiler.tensorboard_trace_handler('./vllm_profile_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        with_flops=True
    ) as prof:
# warmup
        for i in range(2):
            outputs = llm.generate(["Hello world, who are you?"], sampling_params)
            prof.step()

        # 简单性能测量
        start_time = time.time()
        torch.cuda.synchronize()  # 确保GPU操作完成

        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)

        # 结束计时并测量性能
        torch.cuda.synchronize()
        end_time = time.time()
        total_time = end_time - start_time
        tokens_generated = sum(len(output.outputs[0].text.split()) for output in outputs)
        throughput = tokens_generated / total_time
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"\n性能指标:\n总时间: {total_time:.2f}秒, 吞吐量: {throughput:.2f} tokens/秒, GPU内存: {gpu_mem:.2f}GB")

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            
        # 记录步骤
        prof.step()
    
    # 打印关键统计信息
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    return prof

prof = run_with_profiler()


