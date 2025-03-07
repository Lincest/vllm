import queue
import torch
import threading
from vllm.logger import init_logger
from typing import (Dict)
from vllm.utils import is_pin_memory_available

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
        self.next_layer_idx = None
        self.current_layer_idx = None
        self.modules_dict = {}  # 存储层索引到模块的映射
        self.device = None
        self.start_layer_idx = None # 第一个有 CPU 专家的层 id
        self.last_layer_idx = None # 最后一个有 CPU 专家的层 id
        StreamContext.init()

    def set_start_layer_idx(self, last_layer_idx: int):
        self.start_layer_idx = last_layer_idx

    def set_last_layer_idx(self, last_layer_idx: int):
        self.last_layer_idx = last_layer_idx

    def register_module(self, layer_idx: int, module: torch.nn.Module, device: torch.device):
        """注册模块以便后续预取"""
        self.modules_dict[layer_idx] = module
        self.device = device

    def prefetch_next_layer(self, next_layer_idx: int):
        """异步预取下一层参数"""
        if next_layer_idx == self.last_layer_idx:
            next_layer_idx = self.start_layer_idx

        self.next_layer_idx = next_layer_idx

        if self.next_layer_idx not in self.modules_dict:
            return
            
        next_module = self.modules_dict[next_layer_idx]
        for module in next_module.children():
            if module.__class__.__name__ == 'DeepSeekV2MoE':
                self.load_expert(module)
        self.load_expert(next_module)

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