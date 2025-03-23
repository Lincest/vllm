python -m vllm.entrypoints.openai.api_server \
    --model /home/sugon/xzshi/vllm/models/mixtral-gptq \
    --host 0.0.0.0 \
    --port 8000 \
    --cpu-offload-gb 15 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --max-model-len 1024 \
    --max-num-seqs 4 \
    --scheduling-policy priority \
    --dtype float16
    
    # --max-model-len default: 1024
    # --scheduling-policy 默认是 FCFS 
