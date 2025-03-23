import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

# torch.set_num_threads(8)
print(torch.__config__.show())

def main():
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，请确保你的环境支持GPU运算")
        return
    
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建较大的矩阵以便观察计算效果
    size = 5000
    
    # 初始化GPU矩阵
    print("创建GPU矩阵...")
    a_gpu = torch.randn(size, size, device='cuda')
    b_gpu = torch.randn(size, size, device='cuda')
    
    # 初始化CPU矩阵
    print("创建CPU矩阵...")
    a_cpu = torch.randn(size, size, device='cpu')
    b_cpu = torch.randn(size, size, device='cpu')
    
    # 预热
    print("预热GPU和CPU...")
    _ = torch.matmul(a_gpu, b_gpu)
    _ = torch.matmul(a_cpu, b_cpu)
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
        gpu_num = 15
        cpu_num = 2
        with record_function(f"GPU_then_CPU g-{gpu_num}, c-{cpu_num}"):
            # 提交GPU计算任务（非阻塞）
            print("提交GPU矩阵乘法...")
            for i in range(15):
                c_gpu = torch.matmul(a_gpu, b_gpu)
            
            # 立即提交CPU计算任务，而不等待GPU完成
            print("提交CPU矩阵乘法...")
            for i in range(2):
                c_cpu = torch.matmul(a_cpu, b_cpu)
            
            # 等待所有计算完成
            torch.cuda.synchronize()
            print("两个矩阵乘法都已完成")
    
    # 打印timeline，观察GPU和CPU操作是否重叠
    print("\n性能分析结果:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # 导出chrome trace以便可视化观察
    prof.export_chrome_trace("gpu_cpu_matmul_trace.json")
    print("\n已导出Chrome trace文件: gpu_cpu_matmul_trace.json")
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
