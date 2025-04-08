import torch
from vllm.logger import init_logger
from typing import (Dict)
import weakref
import json
import queue
import threading
import time
import os
import psutil
import gc

logger = init_logger(__name__)

class StreamContext:
    memory_stream: torch.cuda.Stream = None
    offload_stream: torch.cuda.Stream = None
    compute_stream: torch.cuda.Stream = None
    preload_stream: torch.cuda.Stream = None
    initialized = False

    @classmethod
    def init(cls):
        if not cls.initialized:
            cls.memory_stream = torch.cuda.Stream(priority=-1)
            cls.preload_stream = torch.cuda.Stream(priority=0)
            cls.offload_stream = torch.cuda.Stream(priority=0)
            cls.compute_stream = torch.cuda.current_stream()
            cls.initialized = True

class PreloadDaemon:
    def __init__(self, manager):
        """
        初始化预加载守护线程
        
        Args:
            manager: DeepSeekModuleManager实例的引用
        """
        self.manager = manager
        self.preload_queue = []  # 简单列表代替优先级队列
        self.next_layer_prefix = None  # 下一个要处理的层
        self.shutdown_flag = threading.Event()  # 用于停止线程的标志
        self.lock = threading.Lock()  # 单一锁保护所有状态
        
        # 启动守护线程
        self.daemon_thread = threading.Thread(target=self._preload_worker, daemon=True)
        self.daemon_thread.start()
    
    def schedule_preload(self, layer_prefix: str):
        """安排一个层的专家预加载任务"""
        with self.lock:
            # offload 之前的预加载 
            self.manager.offload_preloaded_experts()

            # 清空现有队列(放弃之前的预加载)
            self.preload_queue = []
            self.next_layer_prefix = layer_prefix
            
            # 如果该层存在统计数据，添加专家到预加载队列
            if layer_prefix in self.manager.expert_stats:
                sorted_experts = sorted(
                    self.manager.expert_stats[layer_prefix].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # 提取需要预加载的专家ID
                preload_expert_ids = [expert_id for expert_id, _ in sorted_experts[:self.manager.preload_experts_count]]
                fixed_experts = self.manager.fixed_experts.get(layer_prefix, set())
                
                # 过滤掉固定专家
                preload_expert_ids = [eid for eid in preload_expert_ids if eid not in fixed_experts]
                
                # 直接添加到队列
                self.preload_queue = preload_expert_ids
                
                print(f"[debug] 已安排 {layer_prefix} 的预加载，共 {len(preload_expert_ids)} 个专家")
    
    def notify_layer_loading(self, layer_prefix: str):
        """通知daemon线程当前层的加载开始了，应停止预加载"""
        with self.lock:
            # 简单地清空队列即可
            if layer_prefix == self.next_layer_prefix:
                origin_len = len(self.preload_queue)
                self.preload_queue = []
                print(f"[debug] 清空预加载队列, 剩余未加载的专家个数: {origin_len}，因为层 {layer_prefix} 的加载已开始")
    
    def shutdown(self):
        """关闭守护线程"""
        self.shutdown_flag.set()
        if self.daemon_thread.is_alive():
            self.daemon_thread.join(timeout=1.0)  # 等待线程结束，最多1秒
    
    def _preload_worker(self):
        """预加载工作线程的主循环"""
        while not self.shutdown_flag.is_set():
            # 获取当前要处理的专家和层
            expert_id = None
            layer_prefix = None
            
            with self.lock:
                if self.preload_queue and self.next_layer_prefix:
                    expert_id = self.preload_queue.pop(0)  # 取出并移除第一个元素
                    layer_prefix = self.next_layer_prefix
            
            # 如果有任务，执行预加载
            if expert_id is not None and layer_prefix is not None:
                success = self._load_expert(layer_prefix, expert_id)
                if not success:
                    # print("[debug] 预加载失败")
                    continue
            else:
                # 没有任务，休眠一小段时间
                time.sleep(0.01)
    
    def _load_expert(self, layer_prefix: str, expert_id: int):
        """
        预加载单个专家
        
        Args:
            layer_prefix: 层前缀
            expert_id: 专家ID
            
        Returns:
            bool: 是否成功预加载
        """
        try:
            # 检查队列是否已清空
            with self.lock:
                if layer_prefix != self.next_layer_prefix or not self.preload_queue:
                    print(f"[debug] 取消预加载专家 {expert_id}，因为队列已清空或层前缀已变更")
                    return False
            
            # 检查专家是否存在
            layer_experts = self.manager.expert_params.get(layer_prefix, {})
            local_expert_id = self.manager.get_local_expert_id(expert_id)
            
            if local_expert_id not in layer_experts:
                print(f"[warning] 专家 {local_expert_id} 在 {layer_prefix} 中不存在")
                return False
            
            # 获取专家参数
            w13_param = layer_experts[local_expert_id]["w13"]
            w2_param = layer_experts[local_expert_id]["w2"]
            
            # 使用预加载流加载参数
            with torch.cuda.stream(StreamContext.preload_stream):
                loaded_w13 = self.manager.load_param(w13_param)
                loaded_w2 = self.manager.load_param(w2_param)
            StreamContext.preload_stream.synchronize()
            
            # 记录预加载的参数
            self.manager.preloaded_params.append((loaded_w13, loaded_w2))
            
            # print(f"[debug] 成功预加载专家 {layer_prefix} {expert_id} - (local id: {local_expert_id}) ")
            return True
            
        except Exception as e:
            print(f"[error] 预加载专家 {expert_id} 从层 {layer_prefix} 失败: {e}")
            return False

class DeepSeekModuleManager:
    """
    可以指定的环境变量: 
        - DS_EXPERT_OFFLOAD: 是否开启 offload 功能 (控制逻辑在 vllm-src/vllm/model_executor/models/utils.py)
        - DS_EXPERT_STATS_PATH: 专家激活统计数据保存路径，默认为当前文件所在目录
        - DS_EXPERT_TRACE: 设置为"1"开启专家激活追踪，用于收集和保存专家激活统计信息
        - DS_FIXED_EXPERTS_COUNT: 每层固定加载的专家数量，这些专家将始终保留在GPU中
        - DS_PRELOAD_EXPERTS_COUNT: 预加载下一层的专家数量
        - DS_USE_DAEMON: 是否使用守护线程进行预加载
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

        # 是否开启了 expert offload
        self.is_expert_offload_enabled = os.getenv("DS_EXPERT_OFFLOAD", "1") == "1" # 默认启用

        # 专家激活统计
        self.expert_stats = {}  # 格式: {layer_prefix: {expert_id: count}}
        self.stats_file = os.path.join(os.getenv("DS_EXPERT_STATS_PATH", os.path.dirname(os.path.abspath(__file__))), "expert_activation_stats.json")
        self.is_tracing = os.getenv("DS_EXPERT_TRACE", "0") == "1"
        print(f"[debug] 📁 {self.stats_file=}, {self.is_tracing=}")

        # 加载历史统计数据
        self.load_expert_stats()

        # 专家并行 
        self.expert_map = None # List[int] -> int (global_expert_id: local_expert_id)

        # 固定专家
        self.fixed_experts_count = int(os.getenv("DS_FIXED_EXPERTS_COUNT", "0"))  # 每层固定的专家数量
        self.fixed_experts = {}  # 格式: {layer_prefix: set(id1, id2, ..)}
        self.initialize_fixed_experts()

        # 预加载相关设置
        self.preload_experts_count = int(os.getenv("DS_PRELOAD_EXPERTS_COUNT", "50"))  # 预加载的专家数量
        self.layer_prefixes = []  # 所有层的前缀列表，按照顺序排列
        self.preloaded_params = []  # 预加载的参数列表，格式与loaded_params相同

        # preload daemon
        self.use_daemon = os.getenv("DS_USE_DAEMON", "1") == "1"  # 是否使用守护线程进行预加载
        if self.use_daemon: 
            self.preload_daemon = PreloadDaemon(self)

    def register_expert_map(self, expert_map):
        """
        用于处理专家并行 (EP)

        - expert_map (Optional[torch.Tensor]): A tensor of shape
            (global_num_experts,) mapping from global to local index.
            Contains -1 for experts not assigned to the current rank.
            Returns None if ep_size is 1.
        """
        # expert parallel shard, the value is -1.
        if expert_map is None:
            logger.warning("Expert map is None, skipping registration expert_map for EP.")
            return 
        self.expert_map = expert_map.cpu().tolist()

    def get_local_expert_id(self, global_expert_id: int) -> int:
        """
        global_expert_id -> local_expert_id
        """
        if self.expert_map is None:
            return global_expert_id
        return self.expert_map[global_expert_id]

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
        
        # 所有的 global 唯一专家ID
        global_unique_ids = torch.unique(topk_ids)
        self.update_expert_stats(layer_prefix, global_unique_ids) # 更新专家激活统计
        # 该层的固定专家
        fixed_experts = self.fixed_experts.get(layer_prefix, set())
        # 计算命中率
        hit_rate = len(set(global_unique_ids.tolist()) & fixed_experts) / max(len(global_unique_ids), 1) * 100 if global_unique_ids.numel() > 0 else 0.0
        print(f"[debug] Loading experts {global_unique_ids.tolist()} for {layer_prefix}, 固定专家集合: {fixed_experts}, 🎯命中率: {hit_rate}%")

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
        if self.use_daemon:
            self.preload_daemon.notify_layer_loading(layer_prefix)
        else:
            StreamContext.preload_stream.synchronize()

        with torch.cuda.stream(stream):
            # 将参数复制到合并的权重中的对应位置
            # 使用非阻塞方式将选定的专家参数加载到GPU (on_demand)
            for expert_id in global_unique_ids:
                global_expert_id_int = expert_id.item()  # 转换为Python整数
                local_expert_id_int = self.get_local_expert_id(global_expert_id_int) # 获取本地专家ID

                if local_expert_id_int not in layer_experts:
                    logger.error(f"Expert {local_expert_id_int} not found in {layer_prefix}")
                    continue
                
                # 加载专家参数到GPU
                w13_param = self.load_param(layer_experts[local_expert_id_int]["w13"])
                w2_param = self.load_param(layer_experts[local_expert_id_int]["w2"])

                # 只卸载非固定专家, 专家的固定是全局的
                if global_expert_id_int not in fixed_experts: 
                    self.loaded_params.append((w13_param, w2_param))
                
                with torch.cuda.stream(stream):
                    self.w13_weight[local_expert_id_int].copy_(w13_param.data, non_blocking=True)
                    self.w2_weight[local_expert_id_int].copy_(w2_param.data, non_blocking=True)

        # 卸载当前层专家(已经复制到了 w13 和 w2 不会影响计算) 
        stream.synchronize()

        self.offload_experts()
        
        print(f"[debug] Loaded {len(global_unique_ids)} experts to GPU for {layer_prefix}")

        # 预取下一层专家
        if self.use_daemon:
            # 获取下一层的前缀
            next_layer_idx = -1
            if not self.layer_prefixes:
                self.layer_prefixes = sorted(self.moe_modules.keys(), key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])
            
            try:
                current_idx = self.layer_prefixes.index(layer_prefix)
                if current_idx >= len(self.layer_prefixes) - 1:
                    next_layer_idx = 0  # 循环回到第一层
                else:
                    next_layer_idx = current_idx + 1
                    
                next_layer_prefix = self.layer_prefixes[next_layer_idx]
                # 安排下一层的预加载
                self.preload_daemon.schedule_preload(next_layer_prefix)
                
            except ValueError:
                print(f"[warning] 当前层 {layer_prefix} 未在层前缀列表中找到")
        else:
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

        print(f"[debug] offload {len(self.loaded_params)} experts")
            
        # offload params
        with torch.no_grad():
            with torch.cuda.stream(StreamContext.offload_stream):
                for w13_param, w2_param in self.loaded_params:
                    if w13_param is not None and hasattr(w13_param, 'pin_cpu_data'):
                        w13_param.data = w13_param.pin_cpu_data
                    
                    if w2_param is not None and hasattr(w2_param, 'pin_cpu_data'):
                        w2_param.data = w2_param.pin_cpu_data
        
        # 清空已加载的参数列表
        self.loaded_params.clear()

    def offload_preloaded_experts(self):
        """
        卸载预加载的专家参数
        """
        if not self.preloaded_params:
            return
            
        print(f"[debug] offload preloaded {len(self.preloaded_params)} experts")
        with torch.no_grad():
            with torch.cuda.stream(StreamContext.offload_stream):
                for w13_param, w2_param in self.preloaded_params:
                    # 内联 offload_param 逻辑
                    if w13_param is not None and hasattr(w13_param, 'pin_cpu_data'):
                        w13_param.data = w13_param.pin_cpu_data
                    
                    if w2_param is not None and hasattr(w2_param, 'pin_cpu_data'):
                        w2_param.data = w2_param.pin_cpu_data
        
        self.preloaded_params.clear()
        
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
        if not self.is_expert_offload_enabled:
            print("🎯 [debug] 未开启专家卸载功能，开启方式: export DS_EXPERT_OFFLOAD=1")
            return 

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
            
            # 获取专家数量 (local expert num)
            num_experts = module.w13_weight.size(0)

            # 在manager中为该层创建专家参数字典
            if layer_prefix not in self.expert_params:
                self.expert_params[layer_prefix] = {}
            
            bbefore_mem = f"{psutil.virtual_memory().used / (1024 ** 3):.2f}"
            # 拆分权重并存储到manager中
            for i in range(num_experts):
                # # 创建w13权重的pin memory版本
                # w13_pinned = module.w13_weight[i].clone().pin_memory()
                # # 创建w2权重的pin memory版本
                # w2_pinned = module.w2_weight[i].clone().pin_memory()
                
                # self.expert_params[layer_prefix][i] = {
                #     "w13": torch.nn.Parameter(w13_pinned),
                #     "w2": torch.nn.Parameter(w2_pinned)
                # }

                # 预分配已经pin_memory的张量
                w13_pinned = torch.empty_like(module.w13_weight[i], pin_memory=True)
                w2_pinned = torch.empty_like(module.w2_weight[i], pin_memory=True)
                
                # in-place复制
                w13_pinned.copy_(module.w13_weight[i])
                w2_pinned.copy_(module.w2_weight[i])
                
                self.expert_params[layer_prefix][i] = {
                    "w13": torch.nn.Parameter(w13_pinned),
                    "w2": torch.nn.Parameter(w2_pinned)
                }
            
            # 初始化一次 w13_weight / w2_weight 释放原始权重
            self.init_w13_w2_weight(w13_weight=module.w13_weight, w2_weight=module.w2_weight)

            # ++=================== debug =============
            import gc
            
            before_mem = f"{psutil.virtual_memory().used / (1024 ** 3):.2f}"
            print("[debug] w13_weight.device = ", module.w13_weight.device)

            module.w13_weight = None
            module.w2_weight = None
            # 触发垃圾回收
            gc.collect()
            import ctypes 
            ctypes.pythonapi.PyGC_Collect()
            torch.cuda.empty_cache()

            print(f"[debug] Released original weights for {layer_prefix}, memory: {bbefore_mem}GB -> {before_mem}GB -> {psutil.virtual_memory().used / (1024 ** 3):.2f}GB")
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
            self.layer_prefixes = sorted(self.moe_modules.keys(), key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])
        
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
        preload_global_expert_ids = [expert_id for expert_id, _ in sorted_experts[:self.preload_experts_count]]
        print(f"[debug] Preloading top {len(preload_global_expert_ids)} experts for next layer {next_layer_prefix}: {preload_global_expert_ids}")
        
        # 检查下一层的专家参数是否存在
        if next_layer_prefix not in self.expert_params:
            print(f"[warning] No expert parameters for next layer {next_layer_prefix}")
            return
        
        # 预取专家参数
        next_layer_experts = self.expert_params[next_layer_prefix]
        fixed_experts = self.fixed_experts.get(next_layer_prefix, set())
        with torch.cuda.stream(StreamContext.preload_stream):
            for global_expert_id in preload_global_expert_ids:
                local_expert_id = self.get_local_expert_id(global_expert_id)
                if local_expert_id not in next_layer_experts:
                    print(f"[warning] Expert {local_expert_id} not found in {next_layer_prefix}")
                    continue
                
                # 预加载专家参数到GPU
                w13_param = self.load_param(next_layer_experts[local_expert_id]["w13"])
                w2_param = self.load_param(next_layer_experts[local_expert_id]["w2"])

                # 记录预加载的参数
                if global_expert_id not in fixed_experts:
                    self.preloaded_params.append((w13_param, w2_param))
        
        print(f"[debug] Preloaded {len(self.preloaded_params)} experts for next layer {next_layer_prefix}")

    def load_param(self, param):
        if param is None:
            return None

        if param.device.type == 'cuda':  # 这个很重要，防止一个参数被 load 两次导致 pin_cpu_data 指向一个 gpu 上的数据
            return param
            
        param_applied = param.cuda(non_blocking=True)
        
        param.pin_cpu_data = param.data
        param.data = param_applied
        
        return param

    def __del__(self):
        """析构函数，确保守护线程正确关闭"""
        if hasattr(self, 'preload_daemon'):
            self.preload_daemon.shutdown()
