import torch
import time

def measure_transfer_speed(size_gb=1, num_trials=5):
    """
    测量数据传输速度（CPU到GPU或GPU到CPU）
    
    Args:
        size_gb: 要传输的张量大小（GB）
        num_trials: 测试次数
    
    Returns:
        平均传输速度（GB/秒）
    """
    # 计算要创建的元素数量
    # 一个float32元素占4字节，1GB = 2^30字节
    num_elements = int(size_gb * (2**30) / 4)
    
    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("GPU不可用，无法进行测试")
        return
    
    device = torch.device("cuda")
    
    transfer_speeds = []
    
    for trial in range(num_trials):
        # 创建CPU上的大张量
        tensor = torch.randn(num_elements, dtype=torch.float32)
        
        # 确保GPU缓存被清空
        torch.cuda.empty_cache()
        
        # 同步GPU，确保之前的操作已完成
        torch.cuda.synchronize()
        
        # 测量传输时间
        start_time = time.time()
        tensor_gpu = tensor.to(device)
        torch.cuda.synchronize()  # 等待传输完成
        end_time = time.time()
        
        # 计算传输速度
        transfer_time = end_time - start_time
        speed_gb_per_sec = size_gb / transfer_time
        
        transfer_speeds.append(speed_gb_per_sec)
        
        print(f"试验 {trial+1}: 传输 {size_gb:.2f} GB 用时 {transfer_time:.4f} 秒，速度 {speed_gb_per_sec:.2f} GB/秒")
        
        # 释放内存
        del tensor, tensor_gpu
        torch.cuda.empty_cache()
    
    # 计算平均速度和峰值速度
    avg_speed = sum(transfer_speeds) / len(transfer_speeds)
    max_speed = max(transfer_speeds)
    
    print(f"\n平均传输速度: {avg_speed:.2f} GB/秒")
    print(f"峰值传输速度: {max_speed:.2f} GB/秒")
    print(f"理论上每秒最大可传输: {avg_speed:.2f} GB")
    
    return avg_speed

# 测量不同大小的传输速度
for size in [0.1, 0.5, 1.0, 2.0]:
    print(f"\n测试 {size} GB 大小的张量传输:")
    measure_transfer_speed(size_gb=size)
