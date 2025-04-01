"""
测试 FusedMoE 中 w13_weight / w2_weight 的张量拆分存储和合并
"""
import torch
import gc
import time
import numpy as np

def split_and_merge_weights(w13_weight, split=True):
    """
    拆分或合并权重张量。
    
    参数:
        w13_weight: 要处理的张量
        split: 如果为True，则拆分张量；如果为False，则合并已拆分的张量
        
    返回:
        拆分模式: 返回包含多个拆分张量的列表
        合并模式: 返回合并后的单个张量
    """
    if split:
        # 获取张量的形状信息
        num_experts = w13_weight.shape[0]
        intermediate_size_per_partition = w13_weight.shape[1] // 2
        hidden_size = w13_weight.shape[2]
        dtype = w13_weight.dtype
        device = w13_weight.device
        
        # 将大张量拆分为num_experts个小张量
        split_weights = []
        for i in range(num_experts):
            # 提取每个专家的权重
            expert_weight = w13_weight[i].clone()
            split_weights.append(expert_weight)
            
        return split_weights
    else:
        # 假设输入是拆分后的权重列表，需要合并
        if not isinstance(w13_weight, list):
            raise ValueError("合并模式下，输入应该是张量列表")
            
        # 获取列表中第一个张量的形状信息
        intermediate_size_per_partition = w13_weight[0].shape[0] // 2
        hidden_size = w13_weight[0].shape[1]
        num_experts = len(w13_weight)
        first_device = w13_weight[0].device
        first_dtype = w13_weight[0].dtype
        
        # 创建一个新的空张量来存储合并后的结果(放在第一个张量的设备上)
        merged_weight = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=first_dtype,
            device=first_device
        )
        
        # 填充合并后的张量
        for i, expert_weight in enumerate(w13_weight):
            merged_weight[i] = expert_weight.to(first_device)
            
        return merged_weight

def calculate_tensor_size(tensor):
    """
    计算张量占用的内存大小（以MB为单位）
    """
    element_size = tensor.element_size()  # 每个元素占用的字节数
    num_elements = tensor.nelement()      # 元素总数
    size_bytes = element_size * num_elements
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def test_pin_memory_and_gpu_mixing():
    """
    测试权重拆分和合并功能，包括CPU pin_memory和GPU混合存储
    """
    print("="*80)
    print("开始测试权重拆分和合并功能...")
    print("="*80)
    
    # 检查是否有可用的CUDA设备
    if not torch.cuda.is_available():
        print("警告: 没有可用的CUDA设备，将使用CPU进行测试")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"使用设备: {device} - {torch.cuda.get_device_name(device)}")
        
    # 创建测试参数
    num_experts = 8
    intermediate_size_per_partition = 4096  # 使用更大的尺寸来测试带宽
    hidden_size = 4096
    params_dtype = torch.float16
    
    # 计算理论上的张量大小
    total_elements = num_experts * 2 * intermediate_size_per_partition * hidden_size
    element_size_bytes = torch.finfo(params_dtype).bits // 8
    theoretical_size_bytes = total_elements * element_size_bytes
    theoretical_size_mb = theoretical_size_bytes / (1024 * 1024)
    
    print(f"测试参数:")
    print(f"  - 专家数量: {num_experts}")
    print(f"  - 每个分区的中间大小: {intermediate_size_per_partition}")
    print(f"  - 隐藏层大小: {hidden_size}")
    print(f"  - 参数数据类型: {params_dtype}")
    print(f"  - 理论张量大小: {theoretical_size_mb:.2f} MB")
    
    print("\n创建测试权重张量...")
    
    # 测量GPU内存使用前
    if device.type == "cuda":
        torch.cuda.synchronize()
        gpu_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
    
    # 创建一个示例权重张量并放到GPU上
    start_time = time.time()
    w13_weight = torch.nn.Parameter(torch.empty(
        num_experts,
        2 * intermediate_size_per_partition,
        hidden_size,
        dtype=params_dtype,
        device=device))
    
    # 初始化权重
    torch.nn.init.normal_(w13_weight.data, mean=0.0, std=0.02)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    creation_time = time.time() - start_time
    
    # 测量GPU内存使用后
    if device.type == "cuda":
        torch.cuda.synchronize()
        gpu_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
        actual_gpu_mem_increase = gpu_mem_after - gpu_mem_before
    
    # 计算实际张量大小
    actual_size_mb = calculate_tensor_size(w13_weight)
    
    print(f"张量创建完成:")
    print(f"  - 创建时间: {creation_time:.4f} 秒")
    print(f"  - 实际张量大小: {actual_size_mb:.2f} MB")
    if device.type == "cuda":
        print(f"  - GPU内存增加: {actual_gpu_mem_increase:.2f} MB")
    
    # 记录原始权重的副本用于验证
    original_weight = w13_weight.clone()
    
    print("\n开始拆分权重...")
    start_time = time.time()
    split_weights = split_and_merge_weights(w13_weight, split=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    split_time = time.time() - start_time
    
    print(f"拆分完成:")
    print(f"  - 耗时: {split_time:.4f} 秒")
    print(f"  - 拆分后得到 {len(split_weights)} 个张量")
    print(f"  - 每个张量的形状: {split_weights[0].shape}")
    
    # 计算每个拆分张量的大小
    single_expert_size_mb = calculate_tensor_size(split_weights[0])
    print(f"  - 每个专家张量大小: {single_expert_size_mb:.2f} MB")
    print(f"  - 所有专家张量总大小: {single_expert_size_mb * num_experts:.2f} MB")
    
    print("\n将一半权重放在CPU的pin_memory，一半放在GPU...")
    cpu_experts = num_experts // 2
    
    # 测量CPU-GPU传输时间
    cpu_transfer_times = []
    for i in range(cpu_experts):
        # 移动到CPU并使用pin_memory
        start_time = time.time()
        split_weights[i] = split_weights[i].cpu()
        if hasattr(split_weights[i], 'pin_memory'):
            split_weights[i] = split_weights[i].pin_memory()
        cpu_transfer_times.append(time.time() - start_time)
    
    avg_cpu_transfer_time = np.mean(cpu_transfer_times)
    total_cpu_transfer_time = sum(cpu_transfer_times)
    total_cpu_transfer_size_mb = single_expert_size_mb * cpu_experts
    cpu_transfer_bandwidth = total_cpu_transfer_size_mb / total_cpu_transfer_time if total_cpu_transfer_time > 0 else 0
    
    print(f"CPU传输:")
    print(f"  - 总传输时间: {total_cpu_transfer_time:.4f} 秒")
    print(f"  - 平均每个专家传输时间: {avg_cpu_transfer_time:.4f} 秒")
    print(f"  - 总传输数据大小: {total_cpu_transfer_size_mb:.2f} MB")
    print(f"  - GPU -> CPU 传输带宽: {cpu_transfer_bandwidth:.2f} MB/s")
    
    # 记录每个专家权重的存储位置
    print("\n专家权重存储位置:")
    for i, weight in enumerate(split_weights):
        print(f"  - 专家 {i}: {weight.device}")
    
    # 释放原始权重以节省GPU内存
    del w13_weight
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("\n开始合并权重(全部放到GPU)...")
    # 确保GPU准备好
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # 运行多次以获取更准确的时间
    num_trials = 3
    merge_times = []
    
    for trial in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        merged_weight = split_and_merge_weights(split_weights, split=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        merge_time = time.time() - start_time
        merge_times.append(merge_time)
        
        # 只保留最后一次的结果
        if trial < num_trials - 1:
            del merged_weight
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    avg_merge_time = np.mean(merge_times)
    min_merge_time = min(merge_times)
    max_merge_time = max(merge_times)
    
    # 计算CPU->GPU传输带宽
    gpu_transfer_size_mb = single_expert_size_mb * cpu_experts
    gpu_transfer_bandwidth = gpu_transfer_size_mb / min_merge_time
    
    print(f"合并完成:")
    print(f"  - 平均耗时 ({num_trials}次试验): {avg_merge_time:.4f} 秒")
    print(f"  - 最短耗时: {min_merge_time:.4f} 秒")
    print(f"  - 最长耗时: {max_merge_time:.4f} 秒")
    print(f"  - CPU -> GPU 传输数据大小: {gpu_transfer_size_mb:.2f} MB")
    print(f"  - CPU -> GPU 传输带宽(基于最短时间): {gpu_transfer_bandwidth:.2f} MB/s")
    print(f"  - 合并后的张量形状: {merged_weight.shape}")
    print(f"  - 合并后的张量设备: {merged_weight.device}")
    
    # 验证合并后的权重是否与原始权重相同
    print("\n验证合并后的权重是否与原始权重相同...")
    
    # 确保两个张量在同一设备上进行比较
    if original_weight.device != merged_weight.device:
        original_weight = original_weight.to(merged_weight.device)
    
    is_equal = torch.allclose(original_weight, merged_weight, rtol=1e-4, atol=1e-4)
    max_diff = torch.max(torch.abs(original_weight - merged_weight)).item()
    
    print(f"  - 合并后的张量与原始张量是否相同: {is_equal}")
    print(f"  - 最大误差: {max_diff}")
    
    # 打印内存使用情况
    if device.type == "cuda":
        current_gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  - 当前GPU内存使用: {current_gpu_mem:.2f} MB")
    
    print("\n总结:")
    print(f"  - 拆分时间: {split_time:.4f} 秒")
    print(f"  - GPU -> CPU 传输时间: {total_cpu_transfer_time:.4f} 秒")
    print(f"  - GPU -> CPU 传输带宽: {cpu_transfer_bandwidth:.2f} MB/s")
    print(f"  - 合并时间(最佳): {min_merge_time:.4f} 秒")
    print(f"  - CPU -> GPU 传输带宽: {gpu_transfer_bandwidth:.2f} MB/s")
    
    print("\n测试完成!")

# 运行测试
if __name__ == "__main__":
    test_pin_memory_and_gpu_mixing()