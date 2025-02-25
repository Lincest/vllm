# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
import os

# 示例提示
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# 创建采样参数对象
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="/home/sugon/xzshi/vllm/models/mixtral-gptq", 
    cpu_offload_gb=15,
    trust_remote_code=True,
    gpu_memory_utilization=0.7,
    max_model_len=2048,
    dtype="float16"
)

# 从提示生成文本
outputs = llm.generate(prompts, sampling_params)

# 打印输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
