python -m vllm.entrypoints.openai.api_server \
    --model /home/sugon/xzshi/vllm/models/mixtral-gptq \
    --host 0.0.0.0 \
    --port 8000 \
    --cpu-offload-gb 15 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --max-model-len 2048 \
    --dtype float16
