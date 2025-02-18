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
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/tree/main
# 大小：31.41 GB
llm = LLM(model="/home/sugon/xzshi/vllm/models/deepseek-v2-lite",cpu_offload_gb=20,trust_remote_code=True,dtype="float16")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
