import queue
import torch
import threading
from vllm.logger import init_logger
from typing import (Dict)
from vllm.utils import is_pin_memory_available

logger = init_logger(__name__)

class StreamContext:
    memory_stream: torch.cuda.Stream = None
    unload_stream: torch.cuda.Stream = None
    compute_stream: torch.cuda.Stream = None
    initialized = False

    @classmethod
    def init(cls):
        if not cls.initialized:
            cls.memory_stream = torch.cuda.Stream(priority=0)
            cls.unload_stream = torch.cuda.Stream(priority=0)
            cls.compute_stream = torch.cuda.current_stream()
            cls.initialized = True


# 添加预取管理器
class ExpertPreloadManager:
    def __init__(self):
        self.next_layer_idx = None
        self.current_layer_idx = None
        self.modules_dict = {}  # 存储层索引到模块的映射
        self.device = None
        self.start_layer_idx = None # 第一个有 CPU 专家的层 id
        self.last_layer_idx = None # 最后一个有 CPU 专家的层 id
        self.param_data_dict = {} #  # 格式: {layer_idx: {param_name: cpu_data}}
        StreamContext.init()

    def set_start_layer_idx(self, last_layer_idx: int):
        self.start_layer_idx = last_layer_idx

    def set_last_layer_idx(self, last_layer_idx: int):
        self.last_layer_idx = last_layer_idx

    def register_module(self, layer_idx: int, module: torch.nn.Module, device: torch.device):
        """注册模块以便后续预取"""
        self.modules_dict[layer_idx] = module
        self.device = device

    def register_cpu_param(self, layer_idx: int, param_name: str, cpu_data: torch.Tensor):
        print(f"[debug] 🎯 register cpu data: layer_{layer_idx}_{param_name}")
        if layer_idx not in self.param_data_dict:
            self.param_data_dict[layer_idx] = {}
        self.param_data_dict[layer_idx][param_name] = cpu_data

    def prefetch_next_layer(self, next_layer_idx: int):
        """异步预取下一层参数"""
        if next_layer_idx == self.last_layer_idx:
            next_layer_idx = self.start_layer_idx

        self.next_layer_idx = next_layer_idx

        if self.next_layer_idx not in self.modules_dict:
            return
            
        next_module = self.modules_dict[next_layer_idx]
        
        # 使用单独的CUDA流进行异步预取
        with torch.cuda.stream(StreamContext.memory_stream):
            # logger.info(f"🚀 预取层 {next_layer_idx} 的参数到 GPU")
            for name, p in next_module.named_parameters():
                if p.device.type == "cpu" and "expert" in name:  # 只处理CPU上的expert参数
                    p.data = p.data.to(self.device, non_blocking=True)

    def release_current_layer(self, current_layer_idx: int):
        """异步释放当前层参数回CPU"""
        if current_layer_idx not in self.modules_dict:
            return
            
        current_module = self.modules_dict[current_layer_idx]
            
        # logger.info(f"♻️ 释放层 {current_layer_idx} 的参数回 CPU")
        # 使用单独的CUDA流进行异步释放
        with torch.cuda.stream(StreamContext.unload_stream):
            for name, p in current_module.named_parameters():
                if p.device.type != "cpu" and "expert" in name:
                    # print(f"[debug] 🎯 offload data: layer_{current_layer_idx}_{name}")
                    p.data = self.param_data_dict[current_layer_idx][name]
        # print()        

    # def load_expert(self, m: nn.Module): 
    #     """
    #     和 load_experts_by_keys 结合使用      
    #     """
    #     stream = StreamContext.memory_stream
    #     for module in m.children():
    #         self.load_expert(module)

    #     for key, param in m._parameters.items():
    #         if param is None:
    #             continue
    #         # Tensors stored in modules are graph leaves, and we don't want to
    #         # track autograd history of `param_applied`, so we have to use
    #         # `with torch.no_grad():`
    #         with torch.no_grad():
    #             with torch.cuda.stream(stream):
    #                 param_applied = param.cuda(non_blocking=True)
    #         param.pin_cpu_data = param.data
    #         param.data = param_applied
    #         out_param = param

    #         if param.grad is not None:
    #             with torch.no_grad():
    #                 with torch.cuda.stream(stream):
    #                     grad_applied = param.grad.cuda(non_blocking=True)

    #             out_param.grad.pin_cpu_data = param.grad.data
    #             out_param.grad.data = grad_applied

    #     for key, buf in m._buffers.items():
    #         if buf is not None:
    #             with torch.cuda.stream(stream):
    #                 m._buffers[key] = buf.cuda(non_blocking=True)
    #                 m._buffers[key].pin_cpu_data = buf
    #     return m

    # def unload_expert(self, m: nn.Module, copy=False):
    #     for module in m.children():
    #         self.unload_expert(module, copy)

    #     for key, param in m._parameters.items():
    #         if param is None:
    #             continue
    #         # Tensors stored in modules are graph leaves, and we don't want to
    #         # track autograd history of `param_applied`, so we have to use
    #         # `with torch.no_grad():`
    #         with torch.no_grad():
    #             with torch.cuda.stream(StreamContext.unload_stream):
    #                 param_applied = param.pin_cpu_data
    #                 if copy:
    #                     param_applied.copy_(param.data, non_blocking=True)
    #                 param.data = param_applied
    #         out_param = param

    #         if param.grad is not None:
    #             with torch.no_grad():
    #                 with torch.cuda.stream(StreamContext.unload_stream):
    #                     grad_applied = param.grad.pin_cpu_data
    #                     if copy:
    #                         grad_applied.copy_(param.grad.data, non_blocking=True)
    #                     out_param.grad.data = grad_applied

    #     for key, buf in m._buffers.items():
    #         if buf is not None:
    #             with torch.cuda.stream(StreamContext.unload_stream):
    #                 buf_applied = buf.pin_cpu_data
    #                 if copy:
    #                     buf_applied.copy_(buf, non_blocking=True)
    #                 m._buffers[key] = buf_applied
    #     return m