"""长期记忆管理器（异步流水线版本）

特性:
- FAISS 异步搜索
- 预取缓存机制
- 后台线程池
- 非阻塞查询

使用方式:
    # 同步版本（默认）
    from long_term_memory import LongTermMemory
    
    # 异步版本（性能优化）
    from long_term_memory_async import LongTermMemoryAsync
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import faiss
from config import Config
from concurrent.futures import ThreadPoolExecutor, Future
import threading


class LongTermMemoryAsync:
    """长期记忆管理器（CPU 存储 + FAISS 索引 + 异步流水线）"""
    
    def __init__(self, m=1024, n=1024):
        self.m, self.n = m, n
        self.vector_dim = m * n
        
        # ========== 内存存储（热数据）==========
        self.memory_bank: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.usage_count: List[int] = []
        
        # 批量存储配置
        self.batch_size = Config.flow.LONG_TERM_BATCH_SIZE
        self.current_batch_start = 0
        
        # 存储目录
        self.storage_dir = Path(Config.flow.LONG_TERM_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # ========== FAISS 索引 ==========
        self.index: Optional[faiss.IndexFlatIP] = None
        self.faiss_to_global_idx: List[int] = []
        self._init_faiss_index()
        
        # ========== 异步组件 ==========
        # 后台线程池（FAISS 搜索专用）
        self.executor = ThreadPoolExecutor(
            max_workers=1, 
            thread_name_prefix="faiss_search"
        )
        
        # 预取缓存
        self.prefetch_cache: Dict[str, torch.Tensor] = {}
        self.prefetch_futures: List[Future] = []
        
        # 锁
        self.lock = threading.Lock()
        
        print(f"[LongTermMemoryAsync] 已初始化 (dim={self.vector_dim})")
        
        self._load_existing_memories()
    
    def _init_faiss_index(self):
        """初始化 FAISS 内积索引"""
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.faiss_to_global_idx = []
    
    def _load_existing_memories(self):
        """加载最近的批次文件到内存"""
        if not self.storage_dir.exists():
            return
        
        batch_files = sorted(self.storage_dir.glob("memory_*.pt"))
        recent_files = batch_files[-min(3, len(batch_files)):]
        
        for file_path in recent_files:
            try:
                data = torch.load(file_path, map_location='cpu')
                for x2, remain in zip(data['x2_list'], data['remain_list']):
                    self.memory_bank.append((x2, remain))
                    self.usage_count.append(0)
            except Exception as e:
                print(f"加载失败 {file_path.name}: {e}")
        
        # 加载 FAISS 索引
        faiss_path = self.storage_dir / "index.faiss"
        if faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
        
        self.current_batch_start = len(self.memory_bank)
    
    def record(self, x2: torch.Tensor, remain: torch.Tensor):
        """记录单条轨迹（同步操作，确保数据一致性）"""
        x2_cpu = x2.detach().cpu().clone()
        remain_cpu = remain.detach().cpu().clone()
        
        with self.lock:
            self.memory_bank.append((x2_cpu, remain_cpu))
            self.usage_count.append(0)
            
            # 更新 FAISS 索引
            x2_flat = x2_cpu.view(-1).numpy().astype('float32')
            x2_norm = x2_flat / (np.linalg.norm(x2_flat) + 1e-8)
            self.index.add(x2_norm.reshape(1, -1))
            self.faiss_to_global_idx.append(len(self.memory_bank) - 1)
        
        # 检查是否需要保存
        if len(self.memory_bank) - self.current_batch_start >= self.batch_size:
            self._save_current_batch()
    
    def query_async(self, x_query: torch.Tensor, topk: int = None) -> Future:
        """
        异步查询长期记忆（立即返回 Future）
        
        Args:
            x_query: 查询向量（GPU 或 CPU）
            topk: 返回数量
        
        Returns:
            Future[torch.Tensor] - 未来的结果
        """
        if topk is None:
            topk = Config.flow.LONG_TERM_TOPK
        
        # 提交到后台线程
        future = self.executor.submit(
            self._query_sync, 
            x_query.detach().cpu(), 
            topk
        )
        
        self.prefetch_futures.append(future)
        
        # 清理已完成的任务
        self.prefetch_futures = [f for f in self.prefetch_futures if not f.done()]
        
        return future
    
    def prefetch(self, x_queries: List[torch.Tensor], act_indices: List[int]):
        """
        批量预取长期记忆（针对多个 act_idx）
        
        Args:
            x_queries: [x_1, x_2, ...] - 查询列表
            act_indices: [act_1, act_2, ...] - 对应的类别索引
        """
        # 为每个查询启动后台搜索
        for i, (x, act_idx) in enumerate(zip(x_queries, act_indices)):
            cache_key = f"act{act_idx}_seq{id(x)}_{i}"
            
            # 如果已缓存，跳过
            if cache_key in self.prefetch_cache:
                continue
            
            # 启动异步查询
            future = self.query_async(x)
            self.prefetch_futures.append(future)
            
            # 注册回调：结果到达时存入缓存
            def on_complete(fut, key=cache_key):
                try:
                    result = fut.result()
                    with self.lock:
                        self.prefetch_cache[key] = result
                except Exception as e:
                    print(f"[LT Memory Async] Prefetch error: {e}")
            
            future.add_done_callback(on_complete)
    
    def get_prefetched(self, x_query: torch.Tensor, act_idx: int, query_id: int = None) -> Optional[torch.Tensor]:
        """
        获取预取的结果（非阻塞）
        
        Returns:
            已缓存的长期记忆，或 None（如果还没准备好）
        """
        cache_key = f"act{act_idx}_seq{id(x_query)}_{query_id or 0}"
        
        with self.lock:
            return self.prefetch_cache.pop(cache_key, None)
    
    def wait_all_prefetch(self):
        """等待所有预取完成（同步点）"""
        for future in self.prefetch_futures:
            try:
                future.result()
            except Exception as e:
                print(f"[LT Memory Async] Wait error: {e}")
        
        self.prefetch_futures.clear()
    
    def _query_sync(self, x_query_cpu: torch.Tensor, topk: int) -> torch.Tensor:
        """
        同步查询实现（在后台线程执行）
        
        Args:
            x_query_cpu: CPU 上的查询向量
            topk: 返回数量
        
        Returns:
            长期记忆残差（CPU 张量）
        """
        if len(self.memory_bank) == 0:
            return torch.zeros(self.m, self.n)
        
        # FAISS 搜索
        x2_flat = x_query_cpu.view(-1).numpy().astype('float32')
        x2_norm = x2_flat / (np.linalg.norm(x2_flat) + 1e-8)
        
        # 限制 topk 不超过总数
        actual_topk = min(topk, len(self.memory_bank))
        
        if actual_topk == 0:
            return torch.zeros(self.m, self.n)
        
        # FAISS 搜索（CPU 密集型）
        D, I = self.index.search(x2_norm.reshape(1, -1), actual_topk)
        
        # 加权平均（根据相似度）
        similarities = torch.from_numpy(D[0])  # (actual_topk,)
        indices = torch.from_numpy(I[0])
        
        # softmax 权重
        weights = torch.softmax(similarities, dim=0)
        
        # 提取记忆
        memories = torch.stack([self.memory_bank[idx][0] for idx in indices])  # (k, m, n)
        weighted_memory = torch.sum(weights[:, None, None] * memories, dim=0)
        
        # 更新使用计数
        for idx in indices:
            self.usage_count[idx] += 1
        
        return weighted_memory
    
    def query(self, x_query: torch.Tensor, topk: int = None) -> torch.Tensor:
        """
        同步查询接口（向后兼容）
        
        如果有预取缓存，直接返回；否则阻塞查询
        """
        # 先检查缓存
        cache_key = f"act0_seq{id(x_query)}_0"
        
        with self.lock:
            cached = self.prefetch_cache.pop(cache_key, None)
            if cached is not None:
                print(f"[LT Memory Async] Cache hit!")
                return cached
        
        # 没有缓存，阻塞查询
        return self._query_sync(x_query.detach().cpu(), topk or Config.flow.LONG_TERM_TOPK)
    
    def _save_current_batch(self):
        """保存当前批次到磁盘"""
        batch_end = len(self.memory_bank)
        batch_x2 = [item[0] for item in self.memory_bank[self.current_batch_start:batch_end]]
        batch_remain = [item[1] for item in self.memory_bank[self.current_batch_start:batch_end]]
        
        import time
        timestamp = int(time.time())
        file_path = self.storage_dir / f"memory_{timestamp}_{self.current_batch_start}.pt"
        
        torch.save({
            'x2_list': batch_x2,
            'remain_list': batch_remain,
            'indices': self.faiss_to_global_idx[self.current_batch_start:batch_end]
        }, file_path)
        
        # 保存 FAISS 索引
        faiss_path = self.storage_dir / "index.faiss"
        faiss.write_index(self.index, str(faiss_path))
        
        print(f"[LT Memory Async] 保存 {batch_end - self.current_batch_start} 条记忆到 {file_path.name}")
        
        self.current_batch_start = batch_end
        
        # 清理旧文件
        self._cleanup_old_batches(max_batches=5)
    
    def _cleanup_old_batches(self, max_batches: int = 5):
        """清理旧的批次文件"""
        batch_files = sorted(self.storage_dir.glob("memory_*.pt"))
        
        if len(batch_files) > max_batches:
            files_to_remove = batch_files[:-max_batches]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    print(f"[LT Memory Async] 删除旧批次：{file_path.name}")
                except Exception as e:
                    print(f"[LT Memory Async] 删除失败：{e}")
    
    def shutdown(self):
        """关闭线程池"""
        self.wait_all_prefetch()
        self.executor.shutdown(wait=True)
    
    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except:
            pass
