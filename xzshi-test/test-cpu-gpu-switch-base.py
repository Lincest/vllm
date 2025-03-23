import torch
import torch.nn as nn
import time
from torch.profiler import profile, record_function, ProfilerActivity

# 定义SwitchTransformers相关组件
class SwitchTransformersConfig:
    def __init__(self):
        # 使用提供的具体配置参数
        self.vocab_size = 32128
        self.d_model = 768
        self.a_KV = 64
        self.d_ff = 2048
        self.expert_capacity = 64
        self.num_layers = 12
        self.num_sparse_encoder_layers = 3
        self.num_decoder_layers = 12
        self.num_sparse_decoder_layers = 3
        self.num_heads = 12
        self.num_experts = 8
        self.router_bias = False
        self.router_jitter_noise = 0.01
        self.router_dtype = 'float32'
        self.router_ignore_padding_tokens = False
        self.relative_attention_num_buckets = 32
        self.relative_attention_max_distance = 128
        self.dropout_rate = 0.1
        self.layer_norm_epsilon = 1e-06
        self.router_z_loss_coef = 0.001
        self.router_aux_loss_coef = 0.001
        self.initializer_factor = 1.0
        self.dense_act_fn = "relu"
        self.is_encoder_decoder = True
        self.add_router_probs = False
        self.use_cache = True
        self.pad_token_id = 0
        self.eos_token_id = 1

# 定义激活函数字典
ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "silu": torch.nn.functional.silu,
    "swish": torch.nn.functional.silu,
    # 确保包含配置中指定的激活函数
}

class SwitchTransformersDenseActDense(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]
        
    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

# torch.set_num_threads(8)
print(torch.__config__.show())

def main():
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，请确保你的环境支持GPU运算")
        return
    
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    
    # 初始化GPU和CPU模型及输入
    print("创建GPU和CPU模型及输入数据...")
    batch_size = 128 
    seq_len = 1024
    config = SwitchTransformersConfig()
    
    # GPU模型和输入
    gpu_model = SwitchTransformersDenseActDense(config).to('cuda')
    gpu_input = torch.randn(batch_size, seq_len, config.d_model, device='cuda')
    
    # CPU模型和输入
    cpu_model = SwitchTransformersDenseActDense(config).to('cpu')
    cpu_input = torch.randn(batch_size, seq_len, config.d_model, device='cpu')
    
    print(f"使用配置: d_model={config.d_model}, d_ff={config.d_ff}, act_fn={config.dense_act_fn}")
    
    # 创建模型实例
    model = SwitchTransformersDenseActDense(config).to('cpu')
    
    # 预热
    print("预热GPU和CPU模型...")
    gpu_output = gpu_model(gpu_input)
    cpu_output = cpu_model(cpu_input)
    torch.cuda.synchronize()  # 确保GPU操作完成
    
    # 开始profiler
    print("\n开始性能分析...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # 场景1：先GPU后CPU - 检查是否并行
        with record_function("GPU_then_CPU"):
            # 提交GPU计算任务（非阻塞）
            print("提交GPU SwitchTransformers计算...")
            for i in range(48):
                gpu_output = gpu_model(gpu_input)
            
            # 立即提交CPU计算任务，而不等待GPU完成
            print("提交CPU SwitchTransformers计算...")
            for i in range(4):
                cpu_output = cpu_model(cpu_input)
            
            # 等待所有计算完成
            torch.cuda.synchronize()
            print("两个计算任务都已完成")
    
    # 打印timeline，观察GPU和CPU操作是否重叠
    print("\n性能分析结果:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # 导出chrome trace以便可视化观察
    prof.export_chrome_trace("gpu_cpu_model_trace.json")
    print("\n已导出Chrome trace文件: gpu_cpu_model_trace.json")
    print("你可以在Chrome浏览器中访问chrome://tracing/并加载此文件来可视化执行情况")
    
    # 也输出到文本
    print("\n详细的事件序列:")
    for evt in prof.events():
        if evt.name.startswith('aten::') or evt.name.startswith('cudnn::'):
            continue  # 跳过底层操作以减少输出
        
        start_us = evt.time_range.start
        end_us = evt.time_range.end
        duration_ms = (end_us - start_us) / 1000
        device = "CUDA" if evt.cuda_time_total > 0 else "CPU "
        
        print(f"{device} | {evt.name:<20} | 开始: {start_us/1000:>8.2f}ms | 结束: {end_us/1000:>8.2f}ms | 持续: {duration_ms:>8.2f}ms")

if __name__ == "__main__":
    main()
