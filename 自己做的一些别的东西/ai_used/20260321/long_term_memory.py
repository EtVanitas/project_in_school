"""长期记忆管理器 - 滑动窗口模式"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import faiss
import time
from config import Config


class LongTermMemory:
    """长期记忆管理器 - 滑动窗口模式
    
    核心机制：
    1. 训练时收集轨迹 [x2, remain] 对
    2. 每文件训练完后构建 FAISS 索引
    3. 滑动窗口保留最新 max_trajectories 条
    4. 查询时用 x2 找最相似的 x2，返回对应的 remain
    """
    
    def __init__(self, m=1024, n=1024):
        self.m, self.n = m, n
        self.vector_dim = m * n
        
        # 轨迹数据（CPU 存储）
        self.trajectories: List[Tuple[torch.Tensor, torch.Tensor]] = []
        
        # 批量保存阈值
        self.batch_size = Config.flow.LONG_TERM_BATCH_SIZE
        
        # 最大轨迹数量（滑动窗口，只保留最新 N 条）
        self.max_trajectories = Config.flow.LONG_TERM_MAX_TRAJECTORIES
        
        # 存储目录
        self.storage_dir = Path(Config.flow.LONG_TERM_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # FAISS 索引（延迟构建）
        self.index: Optional[faiss.IndexFlatIP] = None
        self.has_index = False
        
        # ========== 尝试从缓存恢复 ==========
        self._load_from_cache()
        # ====================================
    
    def _load_from_cache(self):
        """从缓存加载轨迹和 FAISS 索引"""
        cache_path = self.storage_dir / "trajectories_cache.pt"
        
        if cache_path.exists():
            try:
                data = torch.load(cache_path, map_location='cpu')
                x2_list = data['x2_list']
                remain_list = data['remain_list']
                
                self.trajectories = [(x2, remain) for x2, remain in zip(x2_list, remain_list)]
                print(f"[长期记忆] 从缓存恢复：{len(self.trajectories)} 条轨迹")
                
                # 同时加载 FAISS 索引
                faiss_path = self.storage_dir / "index.faiss"
                if faiss_path.exists():
                    self.index = faiss.read_index(str(faiss_path))
                    self.has_index = True
                    print(f"[长期记忆] FAISS 索引已加载")
            except Exception as e:
                print(f"[长期记忆] 缓存加载失败：{e}")
    
    def record(self, x2: torch.Tensor, remain: torch.Tensor):
        """记录单条轨迹（假设输入已在 CPU 上，只做 clone）"""
        # 假设输入已经是 CPU 张量（由 alice_main.py 保证）
        # 只需要 clone 避免后续修改
        x2_cpu = x2.clone()
        remain_cpu = remain.clone()
        
        self.trajectories.append((x2_cpu, remain_cpu))
        
        if len(self.trajectories) >= self.batch_size:
            self._save_to_disk()
    
    def record_batch(self, records: List[Tuple[torch.Tensor, torch.Tensor]]):
        """批量记录轨迹（推荐用法，输入应已在 CPU 上）"""
        for x2, remain in records:
            self.record(x2, remain)
        
        # 关键：如果还有剩余数据（不足 batch_size），也保存到磁盘
        if self.trajectories:
            self._save_to_disk()
    
    def _save_to_disk(self):
        """保存当前批次到磁盘"""
        if not self.trajectories:
            return
        
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        file_path = self.storage_dir / f"trajectory_{timestamp}.pt"
        
        torch.save({
            'x2_list': [t[0] for t in self.trajectories],
            'remain_list': [t[1] for t in self.trajectories],
        }, file_path)
        
        print(f"[长期记忆] 保存 {len(self.trajectories)} 条轨迹 → {file_path.name}")
        self.trajectories.clear()
    
    def query(self, x_query: torch.Tensor, topk: int = None) -> torch.Tensor:
        """查询长期记忆 - 有索引用 FAISS，无索引返回零张量"""
        if topk is None:
            topk = Config.flow.LONG_TERM_TOPK
        
        # 检查是否有有效的索引和轨迹
        if self.has_index and self.index.ntotal > 0:
            # 如果 trajectories 为空，说明需要从缓存加载
            if len(self.trajectories) == 0:
                print(f"[长期记忆] trajectories 为空，尝试从缓存加载...")
                self._load_from_cache()
            
            # 如果还是空的，返回零张量
            if len(self.trajectories) == 0:
                print(f"[长期记忆] 无法加载轨迹，返回零张量")
                return torch.zeros(self.m, self.n, device=x_query.device)
            
            # 检查 FAISS 索引和 trajectories 是否同步
            if self.index.ntotal != len(self.trajectories):
                print(f"[警告] FAISS 索引 ({self.index.ntotal}) 和 trajectories ({len(self.trajectories)}) 不同步！")
                print(f"[长期记忆] 重新构建索引...")
                self.build_index()
            
            return self._query_faiss(x_query, topk)
        
        return torch.zeros(self.m, self.n, device=x_query.device)
    
    def _query_faiss(self, x_query: torch.Tensor, topk: int) -> torch.Tensor:
        """FAISS 相似度搜索"""
        # 使用 detach() 分离梯度，然后转换为 numpy
        x_query_np = x_query.detach().cpu().view(-1).numpy().astype('float32')
        x_query_norm = x_query_np / (np.linalg.norm(x_query_np) + 1e-8)
        
        k = min(topk, self.index.ntotal)
        if k == 0:
            print(f"[警告] FAISS 索引为空，无法查询")
            return torch.zeros(self.m, self.n, device=x_query.device)
        
        scores, indices = self.index.search(x_query_norm.reshape(1, -1), k)
        
        # 过滤掉无效的索引（超出 trajectories 范围的）
        valid_remains = []
        valid_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.trajectories):
                valid_remains.append(self.trajectories[idx][1].to(x_query.device))
                valid_scores.append(score)
            else:
                print(f"[警告] 跳过无效索引 idx={idx}, trajectories 长度={len(self.trajectories)}")
        
        if len(valid_remains) == 0:
            print(f"[警告] 没有有效的记忆轨迹，返回零张量")
            return torch.zeros(self.m, self.n, device=x_query.device)
        
        weights = torch.tensor(valid_scores, device=x_query.device)
        weights = F.softmax(weights, dim=0)
        
        return torch.einsum('i,ijk->jk', weights, torch.stack(valid_remains))
    
    def build_index(self):
        """构建 FAISS 索引（滑动窗口模式）
        
        流程：
        1. 加载所有 trajectory_*.pt 文件（本轮新轨迹）
        2. 从内存复制旧轨迹（上一轮的 500 条）
        3. 合并新旧轨迹
        4. 滑动窗口筛选（保留最新 max_trajectories 条）
        5. 构建 FAISS 索引并保存
        6. 保存到 trajectories_cache.pt（持久化）
        7. 清理临时 trajectory_*.pt 文件
        """
        print("\n[长期记忆] 开始构建 FAISS 索引...")
        
        # ========== 1. 从磁盘加载所有轨迹文件（本轮新轨迹）==========
        new_trajectories = []
        batch_files = sorted(self.storage_dir.glob("trajectory_*.pt"))
        
        for file_path in batch_files:
            try:
                data = torch.load(file_path, map_location='cpu')
                for x2, remain in zip(data['x2_list'], data['remain_list']):
                    new_trajectories.append((x2, remain))
            except Exception as e:
                print(f"加载失败 {file_path.name}: {e}")
        
        print(f"[长期记忆] 加载了 {len(new_trajectories)} 条新轨迹（本轮）")
        
        if len(new_trajectories) == 0:
            print("[长期记忆] 无新数据可构建索引")
            return
        
        # ========== 2. 从内存复制旧轨迹（上一轮的 500 条）==========
        old_trajectories = []
        if self.has_index and self.index is not None and self.index.ntotal > 0:
            old_trajectories = self.trajectories.copy()
            print(f"[长期记忆] 从内存中保留 {len(old_trajectories)} 条旧轨迹（上轮）")
        
        # ========== 3. 合并新旧轨迹 ==========
        all_trajectories = old_trajectories + new_trajectories
        print(f"[长期记忆] 合并后总轨迹数：{len(all_trajectories)} 条")
        
        # ========== 4. 滑动窗口：只保留最新的 max_trajectories 条 ==========
        if len(all_trajectories) > self.max_trajectories:
            num_removed = len(all_trajectories) - self.max_trajectories
            all_trajectories = all_trajectories[num_removed:]  # 保留最新的
            print(f"[长期记忆] 滑动窗口：删除 {num_removed} 条旧轨迹，保留最新 {len(all_trajectories)} 条")
        
        # ========== 5. 构建 FAISS 索引 ==========
        self.index = faiss.IndexFlatIP(self.vector_dim)
        
        vectors = []
        for x2, _ in all_trajectories:
            x2_flat = x2.view(-1).numpy().astype('float32')
            x2_norm = x2_flat / (np.linalg.norm(x2_flat) + 1e-8)
            vectors.append(x2_norm)
        
        self.index.add(np.array(vectors, dtype='float32'))
        
        # 更新内存中的轨迹列表
        self.trajectories = all_trajectories
        self.has_index = True
        
        print(f"[长期记忆] FAISS 索引构建完成：{self.index.ntotal} 条向量")
        
        # ========== 6. 保存 FAISS 索引到磁盘 ==========
        faiss_path = self.storage_dir / "index.faiss"
        faiss.write_index(self.index, str(faiss_path))
        print(f"[长期记忆] 索引已保存到 {faiss_path}")
        
        # ========== 7. 保存到 trajectories_cache.pt（持久化）==========
        cache_path = self.storage_dir / "trajectories_cache.pt"
        torch.save({
            'x2_list': [t[0] for t in all_trajectories],
            'remain_list': [t[1] for t in all_trajectories],
        }, cache_path)
        print(f"[长期记忆] 缓存已保存到 {cache_path} ({len(all_trajectories)} 条轨迹)")
        
        # ========== 8. 清理临时轨迹文件 ==========
        if batch_files:
            for file_path in batch_files:
                try:
                    file_path.unlink()
                    print(f"[长期记忆] 清理临时轨迹文件：{file_path.name}")
                except Exception as e:
                    print(f"清理失败 {file_path.name}: {e}")
    
    def clear(self):
        """清空内存中的轨迹（不清除磁盘文件）"""
        self.trajectories.clear()
    
    def clear_index(self):
        """清空索引（用于重建）"""
        self.index = None
        self.has_index = False
        self.trajectories.clear()
        print("[长期记忆] 索引已清空")
