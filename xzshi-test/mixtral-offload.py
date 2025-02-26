# SPDX-License-Identifier: Apache-2.0

# A100 x8 
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

# 加载 Mixtral 8x7B 模型
# https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
# 大小：约 87 GB
llm = LLM(
    model="/mnt/wcy/Mixtral-8x7B-Instruct-v0.1",  # 替换为你的模型路径
    cpu_offload_gb=40,  # 因为模型更大，增加 CPU 卸载内存
    trust_remote_code=True,
    enforce_eager=True,
    dtype="float16"  # 使用半精度以减少内存占用
)

# 从提示生成文本
outputs = llm.generate(prompts, sampling_params)

# 打印输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
