export DS_EXPERT_OFFLOAD=1
export DS_EXPERT_TRACE=0
export DS_FIXED_EXPERTS_COUNT=20
export DS_PRELOAD_EXPERTS_COUNT=30

python -m vllm.entrypoints.openai.api_server \
    --model /home/sugon/xzshi/vllm/models/deepseek-v2-lite \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --max-model-len 512 \
    --scheduling-policy priority \
    --dtype float32
    
    # --max-model-len default: 1024
    # --max-num-seqs 4 \ 最大的批处理大小
    # --scheduling-policy 默认是 FCFS 
