import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def benchmark_tensor_creation(size_gb=1):
    """
    比较在GPU上创建tensor和从CPU加载tensor到GPU的时间
    
    参数:
        size_gb: tensor大小，以GB为单位
    """
    # 计算1GB对应的浮点数数量 (假设使用float32，每个元素4字节)
    num_elements = int(size_gb * (1024**3) / 4)
    
    # 计算接近的形状
    shape_dim = int(np.cbrt(num_elements))
    shape = (shape_dim, shape_dim, shape_dim)
    print(f"创建形状为 {shape} 的tensor (约 {size_gb:.2f} GB)")
    
    # 在GPU上直接创建tensor的时间
    torch.cuda.synchronize()  # 确保之前的CUDA操作已完成
    start_time = time.time()
    
    # 在GPU上创建全零tensor
    gpu_tensor = torch.zeros(shape, dtype=torch.float32, device='cuda')
    
    torch.cuda.synchronize()  # 确保tensor创建完成
    gpu_creation_time = time.time() - start_time
    print(f"在GPU上创建tensor的时间: {gpu_creation_time:.4f} 秒")
    
    # 测量从CPU加载到GPU的时间
    # 首先在CPU上创建tensor
    cpu_tensor = torch.zeros(shape, dtype=torch.float32)
    
    torch.cuda.synchronize()  # 确保之前的CUDA操作已完成
    start_time = time.time()
    
    # 将CPU tensor加载到GPU
    cpu_tensor = cpu_tensor.cuda()
    
    torch.cuda.synchronize()  # 确保加载完成
    cpu_to_gpu_time = time.time() - start_time
    print(f"从CPU加载tensor到GPU的时间: {cpu_to_gpu_time:.4f} 秒")
    
    # 计算比率
    ratio = cpu_to_gpu_time / gpu_creation_time
    print(f"比率 (CPU->GPU / GPU创建): {ratio:.2f}x")
    
    return {
        "gpu_creation_time": gpu_creation_time,
        "cpu_to_gpu_time": cpu_to_gpu_time,
        "ratio": ratio
    }

def run_size_experiments():
    """运行不同大小的tensor实验"""
    sizes = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # GB
    results = []
    
    print("开始运行不同大小的tensor实验...")
    for size in sizes:
        print(f"\n测试 {size} GB tensor:")
        result = benchmark_tensor_creation(size)
        results.append((size, result))
        
        # 清除GPU缓存以减少内存碎片的影响
        torch.cuda.empty_cache()
        # 给系统一些时间冷却
        time.sleep(1)
    
    return results

def plot_results(results):
    """绘制实验结果"""
    sizes = [r[0] for r in results]
    gpu_times = [r[1]["gpu_creation_time"] for r in results]
    cpu_to_gpu_times = [r[1]["cpu_to_gpu_time"] for r in results]
    ratios = [r[1]["ratio"] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    # 绘制时间对比
    plt.subplot(2, 1, 1)
    plt.plot(sizes, gpu_times, 'o-', label='GPU创建时间')
    plt.plot(sizes, cpu_to_gpu_times, 's-', label='CPU->GPU加载时间')
    plt.xlabel('Tensor大小 (GB)')
    plt.ylabel('时间 (秒)')
    plt.title('GPU创建 vs CPU->GPU加载时间对比')
    plt.grid(True)
    plt.legend()
    
    # 绘制比率
    plt.subplot(2, 1, 2)
    plt.plot(sizes, ratios, 'x-', color='red')
    plt.xlabel('Tensor大小 (GB)')
    plt.ylabel('CPU->GPU / GPU创建 比率')
    plt.title('时间比率随Tensor大小的变化')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('tensor_creation_benchmark.png')
    plt.show()

def main():
    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，请检查GPU设置")
        return
    
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    # 预热GPU，减少第一次运行的波动
    print("预热GPU...")
    warmup_tensor = torch.zeros((1000, 1000), device='cuda')
    del warmup_tensor
    torch.cuda.empty_cache()
    
    # 运行单次1GB测试
    print("\n运行单次1GB测试:")
    benchmark_tensor_creation(1.0)
    
    # 是否运行不同大小的实验
    response = input("\n是否运行不同大小的实验系列? (y/n): ")
    if response.lower() == 'y':
        results = run_size_experiments()
        plot_results(results)

if __name__ == "__main__":
    main()