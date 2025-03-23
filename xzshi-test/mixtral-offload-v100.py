# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
import os

# Batch = 16
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
#     "The model name is",
#     "The largest ocean in the world is",
#     "The formula for water is",
#     "The speed of light is approximately",
#     "The author of 'War and Peace' is",
#     "The currency of Japan is",
#     "The tallest mountain on Earth is",
#     "The number of planets in our solar system is",
#     "The chemical symbol for gold is",
#     "The inventor of the telephone was",
#     "The most populated country in the world is",
#     "The year World War II ended was"
# ]
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
    "The year World War II ended was",
    "The process of photosynthesis involves",
    "The longest river in the world is",
    "The human body contains approximately how many bones",
    "The distance from Earth to the Moon is about",
    "The capital of Australia is",
    "The first element on the periodic table is",
    "The national animal of China is",
    "The Renaissance period occurred during the",
    "The primary colors are",
    "The boiling point of water is",
    "The largest desert in the world is",
    "The composer of the Moonlight Sonata was",
    "The theory of relativity was proposed by",
    "The closest star to Earth is",
    "The Great Wall of China is approximately how long",
    "The smallest planet in our solar system is",
    "The main ingredient in bread is",
    "The number of continents on Earth is",
    "The national flower of Japan is",
    "The study of stars and planets is called",
    "The inventor of the light bulb was",
    "The main language spoken in Brazil is",
    "The atomic number of oxygen is",
    "The author of 'Hamlet' is",
    "The human heart has how many chambers",
    "The shape of the Earth is",
    "The main component of the sun is",
    "The largest mammal on Earth is",
    "The capital of Canada is",
    "The four seasons are",
    "The currency of the United Kingdom is",
    "The freezing point of water in Fahrenheit is",
    "The hardest natural substance on Earth is",
    "The number of elements in the periodic table is",
    "The inventor of the World Wide Web was",
    "The national bird of the United States is",
    "The square root of 144 is",
    "The dominant gas in Earth's atmosphere is",
    "The closest planet to the Sun is",
    "The birthplace of Leonardo da Vinci was",
    "The largest organ in the human body is",
    "The theory of evolution was proposed by",
    "The year the Titanic sank was",
    "The main ingredient in chocolate is",
    "The first human to walk on the Moon was",
    "The process of converting food into energy is called",
    "The official language of the United Nations is",
    "The age of the Earth is approximately"
]


# 创建采样参数对象
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="/home/sugon/xzshi/vllm/models/mixtral-gptq", 
    cpu_offload_gb=15,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    max_model_len=1024,
    max_num_seqs=4,
    scheduling_policy="priority",
    dtype="float16"
)

# 从提示生成文本
outputs = llm.generate(prompts, sampling_params)

# 打印输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
