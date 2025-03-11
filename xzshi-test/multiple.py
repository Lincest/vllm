import torch
import time

def setup_matrices(size=2000):
    """创建两对随机矩阵用于测试"""
    # 第一对矩阵 - 用于GPU
    A_gpu = torch.rand(size, size, device='cuda')
    B_gpu = torch.rand(size, size, device='cuda')
    
    # 第二对矩阵 - 用于CPU
    A_cpu = torch.rand(size, size, device='cpu')
    B_cpu = torch.rand(size, size, device='cpu')
    
    return A_gpu, B_gpu, A_cpu, B_cpu

def matrix_mult_gpu_only(A_gpu1, B_gpu1, A_gpu2, B_gpu2):
    """只用GPU计算两次矩阵乘法"""
    start_time = time.time()
    
    # 第一次矩阵乘法
    C_gpu1 = torch.matmul(A_gpu1, B_gpu1)
    # 同步设备以确保第一次乘法完成
    torch.cuda.synchronize()
    
    # 第二次矩阵乘法
    C_gpu2 = torch.matmul(A_gpu2, B_gpu2)
    # 同步设备以确保第二次乘法完成
    torch.cuda.synchronize()
    
    end_time = time.time()
    return end_time - start_time, C_gpu1, C_gpu2

def matrix_mult_gpu_cpu_parallel(A_gpu, B_gpu, A_cpu, B_cpu):
    """使用GPU和CPU并行计算矩阵乘法"""
    start_time = time.time()
    
    # 创建CUDA流
    stream = torch.cuda.Stream()
    
    # 在GPU上使用stream计算
    with torch.cuda.stream(stream):
        C_gpu = torch.matmul(A_gpu, B_gpu)
    
    # 同时在CPU上计算
    C_cpu = torch.matmul(A_cpu, B_cpu)
    
    # 等待GPU流完成
    stream.synchronize()
    
    end_time = time.time()
    return end_time - start_time, C_gpu, C_cpu

def verify_results(C_gpu1, C_gpu2, C_gpu, C_cpu):
    """验证计算结果是否一致"""
    # 将GPU结果移到CPU进行比较
    C_gpu_cpu = C_gpu.cpu()
    C_gpu1_cpu = C_gpu1.cpu()
    C_gpu2_cpu = C_gpu2.cpu()
    
    print("GPU只计算的两个结果是否相近:", torch.allclose(C_gpu1_cpu, C_gpu2_cpu, rtol=1e-3, atol=1e-3))
    print("GPU和CPU计算的结果是否相近:", torch.allclose(C_gpu_cpu, C_cpu, rtol=1e-3, atol=1e-3))

def main():
    print("正在准备矩阵...")
    # 测试不同大小的矩阵
    for size in [50, 100, 200, 500, 1000, 2000, 3000]:
        print(f"\n矩阵大小: {size}x{size}")
        
        # 创建矩阵
        A_gpu, B_gpu, A_cpu, B_cpu = setup_matrices(size)
        
        # 测试只用GPU计算两次
        print("执行仅GPU测试...")
        gpu_only_time, C_gpu1, C_gpu2 = matrix_mult_gpu_only(A_gpu, B_gpu, A_gpu, B_gpu)
        print(f"仅GPU计算两次所需时间: {gpu_only_time:.4f} 秒")
        
        # 测试GPU和CPU并行计算
        print("执行GPU和CPU并行测试...")
        parallel_time, C_gpu, C_cpu = matrix_mult_gpu_cpu_parallel(A_gpu, B_gpu, A_cpu, B_cpu)
        print(f"GPU和CPU并行计算所需时间: {parallel_time:.4f} 秒")
        
        # 计算速度提升
        speedup = gpu_only_time / parallel_time
        print(f"速度提升: {speedup:.2f}x")
        
        # 验证结果
        verify_results(C_gpu1, C_gpu2, C_gpu, C_cpu)
        
        # 清理内存
        del A_gpu, B_gpu, A_cpu, B_cpu, C_gpu1, C_gpu2, C_gpu, C_cpu
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 检查是否有可用的CUDA设备
    if torch.cuda.is_available():
        print(f"CUDA可用。设备: {torch.cuda.get_device_name(0)}")
        main()
    else:
        print("没有可用的CUDA设备。此演示需要GPU。")
