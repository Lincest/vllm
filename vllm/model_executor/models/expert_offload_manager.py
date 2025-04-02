import queue
import torch
import threading
from vllm.logger import init_logger
from typing import (Dict)
from vllm.utils import is_pin_memory_available
import weakref
import json
import os

logger = init_logger(__name__)

class StreamContext:
    memory_stream: torch.cuda.Stream = None
    offload_stream: torch.cuda.Stream = None
    compute_stream: torch.cuda.Stream = None
    initialized = False

    @classmethod
    def init(cls):
        if not cls.initialized:
            cls.memory_stream = torch.cuda.Stream(priority=0)
            cls.offload_stream = torch.cuda.Stream(priority=0)
            cls.compute_stream = torch.cuda.current_stream()
            cls.initialized = True


# 添加预取管理器
class ExpertPreloadManager:
    def __init__(self):
        self.modules_dict = {}  # 存储层索引到模块的映射
        self.device = None
        self.start_layer_idx = None # 第一个有 CPU 专家的层 id
        self.last_layer_idx = None # 最后一个有 CPU 专家的层 id
        self.preload_layer = 1 # 需要预取的层数
        self.load_events = None
        StreamContext.init()

    def set_preload_layer(self, preload_layer: int):
        """
        设置预取的层数
        """
        print(f"[debug] 🎯 set preload layer = {preload_layer}")
        self.preload_layer = preload_layer

    def set_start_layer_idx(self, last_layer_idx: int):
        """
        设置 offload 开始的层
        """
        self.start_layer_idx = last_layer_idx

    def set_last_layer_idx(self, last_layer_idx: int):
        """
        设置 offload 结束的层
        """
        self.last_layer_idx = last_layer_idx
        self.load_events = {
            k:torch.cuda.Event() for k in self.modules_dict.keys() # layer_idx: Event
        }

    def load_guard(self, layer_idx: int):
        """
        等待 layer_idx 的参数加载完毕
        """
        self.load_events[layer_idx].synchronize()


    def register_module(self, layer_idx: int, module: torch.nn.Module, device: torch.device):
        """注册模块以便后续预取"""
        self.modules_dict[layer_idx] = module
        self.device = device

    def prefetch_layer(self, layer_idx: int):
        """异步预取接下来 layer 的参数"""
        if layer_idx > self.last_layer_idx:
            layer_idx = self.start_layer_idx + (layer_idx - self.last_layer_idx - 1)

        print(f"[debug] 🎯 预取 layer = {layer_idx}")

        if layer_idx not in self.modules_dict:
            return
        
        next_module = self.modules_dict[layer_idx]
        for module in next_module.children():
            if module.__class__.__name__ == 'DeepseekV2MoE':
                self.load_expert(module)

        self.load_events[layer_idx] = torch.cuda.Event()
        self.load_events[layer_idx].record(StreamContext.memory_stream)
            
    def release_current_layer(self, current_layer_idx: int):
        """异步释放当前层参数回CPU"""
        if current_layer_idx not in self.modules_dict:
            return
            
        current_module = self.modules_dict[current_layer_idx]
        self.offload(current_module)
            
    def load_expert(self, m): 
        stream = StreamContext.memory_stream
        for module in m.children():
            self.load_expert(module)

        for key, param in m._parameters.items():
            if param is None:
                continue  

            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                with torch.cuda.stream(stream):
                    param_applied = param.cuda(non_blocking=True)
            param.pin_cpu_data = param.data
            param.data = param_applied
            out_param = param

            if param.grad is not None:
                with torch.no_grad():
                    with torch.cuda.stream(stream):
                        grad_applied = param.grad.cuda(non_blocking=True)

                out_param.grad.pin_cpu_data = param.grad.data
                out_param.grad.data = grad_applied

        return m

    def offload(self, m, copy=False):
        for module in m.children():
            self.offload(module, copy)

        for key, param in m._parameters.items():
            if param is None:
                continue
            if not hasattr(param, 'pin_cpu_data'):
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                with torch.cuda.stream(StreamContext.offload_stream):
                    param_applied = param.pin_cpu_data
                    if copy:
                        param_applied.copy_(param.data, non_blocking=True)
                    param.data = param_applied
            out_param = param

            if param.grad is not None:
                with torch.no_grad():
                    with torch.cuda.stream(StreamContext.offload_stream):
                        grad_applied = param.grad.pin_cpu_data
                        if copy:
                            grad_applied.copy_(param.grad.data, non_blocking=True)
                        out_param.grad.data = grad_applied
        return m

class DeepSeekModuleManager:
    """
    可以指定的环境变量: 
        - DS_EXPERT_OFFLOAD: 是否开启 offload 功能 (控制逻辑在 vllm-src/vllm/model_executor/models/utils.py)
        - DS_EXPERT_STATS_PATH: 专家激活统计数据保存路径，默认为当前文件所在目录
        - DS_EXPERT_TRACE: 设置为"1"开启专家激活追踪，用于收集和保存专家激活统计信息
        - DS_FIXED_EXPERTS_COUNT: 每层固定加载的专家数量，这些专家将始终保留在GPU中
        - DS_PRELOAD_EXPERTS_COUNT: 预加载下一层的专家数量
    """
    _instance = None
    
    # 单例模式
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepSeekModuleManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        StreamContext.init()
        # 存储所有 Fused MoE 模块的弱引用 (prefix: module), e.g. ('model.layers.3.mlp.experts': FusedMoE(..))
        self.moe_modules = {} 
        # 单独的一个合并的 w13_weight 和 w2_weight
        self.w13_weight = None
        self.w2_weight = None
        # 存储所有拆分后的专家参数, layer_prefix + expert_id 可以唯一确定一个专家参数
        self.expert_params = {}  # 格式: {layer_prefix: {expert_id: {"w13": param, "w2": param}}}
        # 当前加载到 gpu 的参数
        self.loaded_params = [] # [(w13, w2)]

        # 专家激活统计
        self.expert_stats = {}  # 格式: {layer_prefix: {expert_id: count}}
        self.stats_file = os.path.join(os.getenv("DS_EXPERT_STATS_PATH", os.path.dirname(os.path.abspath(__file__))), "expert_activation_stats.json")
        self.is_tracing = os.getenv("DS_EXPERT_TRACE", "0") == "1"
        print(f"[debug] 📁 {self.stats_file=}, {self.is_tracing=}")

        # 加载历史统计数据
        self.load_expert_stats()

        # 固定专家
        self.fixed_experts_count = int(os.getenv("DS_FIXED_EXPERTS_COUNT", "5"))  # 每层固定的专家数量
        self.fixed_experts = {}  # 格式: {layer_prefix: set(id1, id2, ..)}
        self.initialize_fixed_experts()

        # 预加载相关设置
        self.preload_experts_count = int(os.getenv("DS_PRELOAD_EXPERTS_COUNT", "25"))  # 预加载的专家数量
        self.layer_prefixes = []  # 所有层的前缀列表，按照顺序排列
        self.preloaded_params = []  # 预加载的参数列表，格式与loaded_params相同


    def load_expert_stats(self):
        """
        从文件加载专家激活统计数据
        """
        # 统计文件路径
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    # 加载JSON数据，需要将字符串键转为整数
                    stats_data = json.load(f)
                    # 转换expert_id从字符串到整数
                    for layer_prefix, experts in stats_data.items():
                        self.expert_stats[layer_prefix] = {int(expert_id): count for expert_id, count in experts.items()}
                print(f"[info] Loaded expert activation statistics from {self.stats_file}")
        except Exception as e:
            logger.error(f"Failed to load expert statistics: {e}")
            # 如果加载失败，初始化一个空字典
            self.expert_stats = {}

    def save_expert_stats(self):
        """
        将专家激活统计数据按照每层专家的激活热度排序后保存到文件
        """
        try:
            # 创建一个新的有序字典，按照激活热度排序
            sorted_stats = {}
            for layer_prefix, experts in self.expert_stats.items():
                # 按激活次数排序（从高到低）
                sorted_experts = {
                    str(expert_id): count 
                    for expert_id, count in sorted(
                        experts.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                }
                sorted_stats[layer_prefix] = sorted_experts
            
            # 保存排序后的统计数据
            with open(self.stats_file, 'w') as f:
                json.dump(sorted_stats, f, indent=2)
            print(f"[info] Saved sorted expert activation statistics to {self.stats_file}")
        except Exception as e:
            logger.error(f"Failed to save expert statistics: {e}")

    def update_expert_stats(self, layer_prefix, expert_ids):
        """
        更新专家激活统计
        
        Args:
            layer_prefix (str): 模块的前缀
            expert_ids (torch.Tensor): 被激活的专家ID
        """
        # 确保该层在统计数据中存在
        if layer_prefix not in self.expert_stats:
            self.expert_stats[layer_prefix] = {}
        
        # 更新每个专家的激活次数
        for expert_id in expert_ids:
            expert_id_int = expert_id.item()
            if expert_id_int not in self.expert_stats[layer_prefix]:
                self.expert_stats[layer_prefix][expert_id_int] = 0
            self.expert_stats[layer_prefix][expert_id_int] += 1

    def initialize_fixed_experts(self):
        """
        根据历史统计选择每层的top N专家作为固定专家
        """
        if not self.expert_stats:
            print("[info] No expert statistics available, skipping fixed experts initialization")
            return
            
        for layer_prefix, experts in self.expert_stats.items():
            # 按激活次数排序（从高到低）
            sorted_experts = sorted(experts.items(), key=lambda x: x[1], reverse=True)
            # 取前N个专家ID作为固定专家
            top_experts = {expert_id for expert_id, _ in sorted_experts[:self.fixed_experts_count]}
            self.fixed_experts[layer_prefix] = top_experts
            
        print(f"[info] Initialized fixed experts: {self.fixed_experts}")

    def init_w13_w2_weight(self, w13_weight, w2_weight):
        device = torch.cuda.current_device()
        if self.w13_weight is None and self.w2_weight is None: 
            # Initialize empty tensors on the specified CUDA device
            self.w13_weight = torch.empty_like(w13_weight, device=device)
            self.w2_weight = torch.empty_like(w2_weight, device=device)
            print(f"🎯 [debug] init w13_weight and w2_weight on {device=}, {self.w13_weight.shape=}, {self.w2_weight.shape=}")


    def get_experts_with_topk_ids(self, layer_prefix, topk_ids):
        """
        根据topk_ids将选中的专家从CPU加载到GPU，并填充到self.w13_weight和self.w2_weight中
        
        Args:
            layer_prefix (str): 模块的前缀，例如 'model.layers.3.mlp.experts'
            topk_ids (torch.Tensor): 批次中选择的 topk 专家ID (vllm/model_executor/layers/fused_moe/layer.py)
        
        Returns:
            tuple: (w13_weight, w2_weight) 填充后的权重
        """
        assert self.loaded_params == [], "Loaded params should be empty before loading new ones, should invoke offload_experts() after each FusedMoE forward"

        module_ref = self.moe_modules.get(layer_prefix)
        if module_ref is None:
            logger.error(f"Module with prefix {layer_prefix} not found")
            return None, None
            
        module = module_ref()
        if module is None:
            logger.error(f"Module with prefix {layer_prefix} has been garbage collected")
            return None, None
        
        # 所有的唯一专家ID
        unique_ids = torch.unique(topk_ids)
        self.update_expert_stats(layer_prefix, unique_ids) # 更新专家激活统计
        # 该层的固定专家
        fixed_experts = self.fixed_experts.get(layer_prefix, set())
        # 计算命中率
        hit_rate = len(set(unique_ids.tolist()) & fixed_experts) / max(len(unique_ids), 1) * 100 if unique_ids.numel() > 0 else 0.0
        print(f"[debug] Loading experts {unique_ids.tolist()} for {layer_prefix}, 固定专家集合: {fixed_experts}, 🎯命中率: {hit_rate}%")

        # 确保self.w13_weight和self.w2_weight已初始化
        if self.w13_weight is None or self.w2_weight is None:
            logger.error(f"w13_weight or w2_weight not initialized for {layer_prefix}")
            return None, None
        
        # 检查是否有专家参数
        if layer_prefix not in self.expert_params:
            logger.error(f"No expert parameters found for {layer_prefix}")
            return None, None
        
        # 获取该层的专家参数
        layer_experts = self.expert_params[layer_prefix]
        
        stream = StreamContext.memory_stream

        # synchronize for preload
        stream.synchronize()
        with torch.cuda.stream(stream):
            # 将参数复制到合并的权重中的对应位置
            # 使用非阻塞方式将选定的专家参数加载到GPU (on_demand)
            for expert_id in unique_ids:
                expert_id_int = expert_id.item()  # 转换为Python整数
                if expert_id_int not in layer_experts:
                    logger.error(f"Expert {expert_id_int} not found in {layer_prefix}")
                    continue
                
                # 加载专家参数到GPU
                w13_param = self.load_param(layer_experts[expert_id_int]["w13"])
                w2_param = self.load_param(layer_experts[expert_id_int]["w2"])

                # 记录，方便等下卸载
                if expert_id_int not in fixed_experts: # 只卸载非固定专家
                    self.loaded_params.append((w13_param, w2_param))
                
                with torch.cuda.stream(stream):
                    self.w13_weight[expert_id_int].copy_(w13_param.data, non_blocking=True)
                    self.w2_weight[expert_id_int].copy_(w2_param.data, non_blocking=True)

        stream.synchronize()

        # 卸载当前层专家(已经复制到了 w13 和 w2 不会影响计算) 
        self.offload_experts()
        
        print(f"[debug] Loaded {len(unique_ids)} experts to GPU for {layer_prefix}")

        # 预取下一层专家
        self.preload_next_layer_experts(layer_prefix)

        if self.is_tracing:
            self.save_expert_stats()
        
        # 返回合并后的权重
        return self.w13_weight, self.w2_weight

    def offload_experts(self):
        """
        卸载当前加载到GPU的专家参数
        """
        if not self.loaded_params:
            return
            
        # 单个CUDA流上下文，减少上下文切换
        with torch.no_grad():
            with torch.cuda.stream(StreamContext.offload_stream):
                for w13_param, w2_param in self.loaded_params:
                    # 内联 offload_param 逻辑
                    if w13_param is not None and hasattr(w13_param, 'pin_cpu_data'):
                        w13_param.data = w13_param.pin_cpu_data
                    
                    if w2_param is not None and hasattr(w2_param, 'pin_cpu_data'):
                        w2_param.data = w2_param.pin_cpu_data
        
        # 清空已加载的参数列表
        self.loaded_params.clear()
        
    def register_moe_module(self, layer_prefix, module):
        """注册一个MoE模块到管理器"""
        if layer_prefix in self.moe_modules:
            logger.error(f"Module with layer_prefix {layer_prefix} already registered.")
            return
        self.moe_modules[layer_prefix] = weakref.ref(module)
        print(f"[debug] DeepSeekModuleManager register module: {layer_prefix} -> {module.__class__.__name__}")
        
    def get_all_moe_modules(self):
        """获取所有活跃的 FusedMoE 模块"""
        return {prefix: ref() for prefix, ref in self.moe_modules.items() if ref() is not None}

    def split_w13_w2_weight(self, layer_prefix):
        """
        将 w13_weight 和 w2_weight 按照 num_experts 维度拆分，以便可以按专家ID访问
        只拆分已经 offload 到 cpu 的权重
        
        Args:
            layer_prefix (str): 模块的前缀，例如 'model.layers.3.mlp.experts'
        """
        module_ref = self.moe_modules.get(layer_prefix)
        if module_ref is None:
            logger.error(f"Module with prefix {layer_prefix} not found")
            return
            
        module = module_ref()
        if module is None:
            logger.error(f"Module with prefix {layer_prefix} has been garbage collected")
            return
            
        # 检查权重是否存在
        if not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
            logger.error(f"Module {layer_prefix} does not have w13_weight or w2_weight")
            return
            
        # 检查权重是否在CPU上
        print(f"🎯 [debug] {module.w13_weight.device.type=}, {module.w2_weight.device.type=}")
        if module.w13_weight.device.type == "cpu" and module.w2_weight.device.type == "cpu":
            print(f"[debug] Splitting weights for {layer_prefix} on CPU")
            
            # 获取专家数量
            num_experts = module.w13_weight.size(0)

            # 在manager中为该层创建专家参数字典
            if layer_prefix not in self.expert_params:
                self.expert_params[layer_prefix] = {}
            
            # 拆分权重并存储到manager中
            for i in range(num_experts):
                # 创建w13权重的pin memory版本
                w13_pinned = module.w13_weight[i].clone().pin_memory()
                # 创建w2权重的pin memory版本
                w2_pinned = module.w2_weight[i].clone().pin_memory()
                
                self.expert_params[layer_prefix][i] = {
                    "w13": torch.nn.Parameter(w13_pinned),
                    "w2": torch.nn.Parameter(w2_pinned)
                }
            
            # 初始化一次 w13_weight / w2_weight 释放原始权重
            self.init_w13_w2_weight(w13_weight=module.w13_weight, w2_weight=module.w2_weight)
            module.w13_weight = None
            module.w2_weight = None
            
            # 触发垃圾回收
            import gc
            gc.collect()
            print(f"[debug] Released original weights for {layer_prefix}")
        else:
            # 如果权重不在CPU上，只记录一下但不执行拆分
            print(f"[debug] Weights for {layer_prefix} are not on CPU, skipping split")
    
    def preload_next_layer_experts(self, current_layer_prefix):
        """
        预加载的逻辑(TODO: 之后换成多线程 + 优先级队列)

        基于当前处理的层，预取下一层的热门专家
        如果当前层是最后一层，则预取第一层的专家
        
        Args:
            current_layer_prefix (str): 当前处理的层前缀
        """
        # 首先确保所有层的前缀已排序
        if not self.layer_prefixes:
            self.layer_prefixes = sorted(self.moe_modules.keys())
        
        # 查找当前层在排序列表中的位置
        try:
            current_index = self.layer_prefixes.index(current_layer_prefix)
        except ValueError:
            print(f"[warning] Current layer {current_layer_prefix} not found in layer prefixes")
            return
        
        # 如果当前层已经是最后一层，则预取第一层
        if current_index >= len(self.layer_prefixes) - 1:
            print(f"[debug] Current layer {current_layer_prefix} is the last layer, preloading first layer")
            next_layer_prefix = self.layer_prefixes[0]  # 循环回到第一层
        else:
            next_layer_prefix = self.layer_prefixes[current_index + 1]
        
        # 如果没有下一层的统计数据，跳过预取
        if next_layer_prefix not in self.expert_stats:
            print(f"[debug] No statistics for next layer {next_layer_prefix}, skipping preload")
            return
        
        # 卸载之前预取的参数
        self.offload_preloaded_experts()
        
        # 根据热度排序获取下一层的热门专家
        sorted_experts = sorted(
            self.expert_stats[next_layer_prefix].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 取前N个专家进行预取
        preload_expert_ids = [expert_id for expert_id, _ in sorted_experts[:self.preload_experts_count]]
        print(f"[debug] Preloading top {len(preload_expert_ids)} experts for next layer {next_layer_prefix}: {preload_expert_ids}")
        
        # 检查下一层的专家参数是否存在
        if next_layer_prefix not in self.expert_params:
            print(f"[warning] No expert parameters for next layer {next_layer_prefix}")
            return
        
        # 预取专家参数
        next_layer_experts = self.expert_params[next_layer_prefix]
        fixed_experts = self.fixed_experts.get(next_layer_prefix, set())
        with torch.cuda.stream(StreamContext.memory_stream):
            for expert_id in preload_expert_ids:
                if expert_id not in next_layer_experts:
                    print(f"[warning] Expert {expert_id} not found in {next_layer_prefix}")
                    continue
                
                # 预加载专家参数到GPU
                w13_param = self.load_param(next_layer_experts[expert_id]["w13"])
                w2_param = self.load_param(next_layer_experts[expert_id]["w2"])
                
                # 记录预加载的参数
                if expert_id not in fixed_experts:
                    self.preloaded_params.append((w13_param, w2_param))
        
        print(f"[debug] Preloaded {len(self.preloaded_params)} experts for next layer {next_layer_prefix}")

    def offload_preloaded_experts(self):
        """
        卸载预加载的专家参数
        """
        if not self.preloaded_params:
            return
            
        with torch.no_grad():
            with torch.cuda.stream(StreamContext.offload_stream):
                for w13_param, w2_param in self.preloaded_params:
                    # 内联 offload_param 逻辑
                    if w13_param is not None and hasattr(w13_param, 'pin_cpu_data'):
                        w13_param.data = w13_param.pin_cpu_data
                    
                    if w2_param is not None and hasattr(w2_param, 'pin_cpu_data'):
                        w2_param.data = w2_param.pin_cpu_data
        
        self.preloaded_params.clear()

    def load_param(self, param):
        if param is None:
            return None
            
        param_applied = param.cuda(non_blocking=True)
        
        param.pin_cpu_data = param.data
        param.data = param_applied
        
        return param
