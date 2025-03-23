#!/bin/bash

# 设置变量
MODEL_NAME="/home/sugon/xzshi/vllm/models/mixtral-gptq"  # 这里用你的模型名称，可以是本地路径或Hugging Face模型ID
BACKEND="openai" # 使用OpenAI的API 
NUM_PROMPTS=10  # 使用10个提示进行测试
ENDPOINT="/v1/completions"  # 你的API端点
HOST="localhost"  # 服务器主机
PORT="8000"  # 服务器端口

# 可以选择使用不同的数据集
# 1. 使用随机数据集（最简单的方式，不需要下载任何文件）
echo "正在使用随机数据集进行基准测试..."
python3 "/home/sugon/xzshi/vllm/vllm-src/benchmarks/benchmark_serving.py" \
  --backend ${BACKEND} \
  --host ${HOST} \
  --port ${PORT} \
  --endpoint ${ENDPOINT} \
  --model ${MODEL_NAME} \
  --dataset-name "random" \
  --num-prompts ${NUM_PROMPTS} \
  --random-input-len 256 \
  --random-output-len 20 # 128

# 2. 如果你想使用sonnet数据集（vLLM自带）
# echo "正在使用sonnet数据集进行基准测试..."
# python3 -m vllm.benchmarks.benchmark_serving \
#   --backend ${BACKEND} \
#   --host ${HOST} \
#   --port ${PORT} \
#   --endpoint ${ENDPOINT} \
#   --model ${MODEL_NAME} \
#   --dataset-name "sonnet" \
#   --dataset-path "vllm/benchmarks/sonnet.txt" \
#   --num-prompts ${NUM_PROMPTS}

# 3. 如果你想使用ShareGPT数据集（需要先下载）
# echo "正在使用ShareGPT数据集进行基准测试..."
# DATASET_PATH="./ShareGPT_V3_unfiltered_cleaned_split.json"
# 
# # 检查数据集是否存在，如果不存在则下载
# if [ ! -f "$DATASET_PATH" ]; then
#   echo "正在下载ShareGPT数据集..."
#   wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
# fi
# 
# python3 -m vllm.benchmarks.benchmark_serving \
#   --backend ${BACKEND} \
#   --host ${HOST} \
#   --port ${PORT} \
#   --endpoint ${ENDPOINT} \
#   --model ${MODEL_NAME} \
#   --dataset-name "sharegpt" \
#   --dataset-path ${DATASET_PATH} \
#   --num-prompts ${NUM_PROMPTS}

echo "基准测试完成！"
