# VLLM 0.7.2 for expert offloading

> 默认为所有模型开启 expert offload

### 1 - instlal vllm 0.7.2

1. (optional) create conda env 
```shell
conda create -n vllm python=3.12 -y
conda activate vllm
```

2. install vllm
```shell
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```