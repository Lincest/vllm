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