"""路径生成器模块
前 9 层固定 [0-8]，后 15 层可以跳层或回溯
维护 40 条轨迹的池子，每次随机选一条进行变异后使用
"""

import random
import json
import os
from typing import List, Tuple
from config import PathConfig, ModelConfig, RLConfig


class SimplePathGenerator:
    """简单的路径生成器：维护 40 条轨迹，随机选择并变异"""
    
    def __init__(self):
        """初始化路径生成器（直接使用全局配置）"""
        # 使用全局配置
        model_config = ModelConfig()
        path_config = PathConfig()
        rl_config = RLConfig()
        
        self.total_layers = model_config.num_layers
        self.fixed_prefix = model_config.fixed_layers
        self.num_exploration_paths = rl_config.num_exploration_paths
        self.trajectory_file = path_config.stage1a_path_pool_file
        
        # 尝试加载历史轨迹，如果失败则初始化 40 条顺序路径
        if not self.load_trajectories():
            base_path = list(range(self.total_layers))
            self.trajectory_pool: List[Tuple[List[int], float]] = [(base_path, 0.0) for _ in range(40)]
    
    def add_trajectory(self, path: List[int], loss: float):
        """添加轨迹到池中，如果重复则更新 loss，否则替换最大 loss 的轨迹"""
        # 检查是否已存在该轨迹
        for i, (existing_path, _) in enumerate(self.trajectory_pool):
            if existing_path == path:
                # 轨迹已存在，更新 loss
                self.trajectory_pool[i] = (path, loss)
                return
        
        # 轨迹不存在，找到 loss 最大的并替换（因为 loss 越小越好）
        max_idx = max(range(len(self.trajectory_pool)), key=lambda i: self.trajectory_pool[i][1])
        self.trajectory_pool[max_idx] = (path, loss)

    def sample(self) -> List[int]:
        """随机选一条轨迹，变异后返回"""
        # 随机选择一条
        base_path, _ = random.choice(self.trajectory_pool)
        
        # 变异：50% 后向跳转（删除），50% 前向回溯（插入）
        return self._mutate(base_path)

    def _mutate(self, base_path: List[int]) -> List[int]:
        """变异：随机选第 9 层之后的一层，50% 删除，50% 插入前一层"""
        # 确保路径至少有固定层
        if len(base_path) < self.fixed_prefix:
            return base_path
        
        # 在第 9 层之后随机选一个位置
        idx = random.randint(self.fixed_prefix, len(base_path) - 1)
        current_layer = base_path[idx]
        
        if random.random() < 0.5:
            # 后向跳转：删除当前层（如果是最后一层就删自己，否则删后一层）
            if idx == len(base_path) - 1:
                new_path = base_path[:-1]
            else:
                new_path = base_path[:idx] + base_path[idx+1:]
        else:
            # 前向回溯：在当前层后面插入前一层（复制）
            backjump_to = current_layer - 1
            if backjump_to < self.fixed_prefix:
                return base_path  # 不能回到前 9 层，保持原样
            
            # 在当前层后面插入：[..., current, backjump, next, ...]
            new_path = base_path[:idx+1] + [backjump_to] + base_path[idx+1:]
        
        # 确保以最后一层结尾
        if not new_path or new_path[-1] != self.total_layers - 1:
            new_path.append(self.total_layers - 1)
        
        # 限制最大长度
        if len(new_path) > 30:
            new_path = new_path[:30]
            if new_path[-1] != self.total_layers - 1:
                new_path.append(self.total_layers - 1)
        
        return new_path
    
    def generate_multiple_paths(self) -> List[List[int]]:
        """生成多条路径（数量从 RLConfig 获取）"""
        paths = []
        for _ in range(self.num_exploration_paths):
            path = self.sample()
            paths.append(path)
        return paths
    
    def save_trajectories(self):
        """保存轨迹到 JSON 文件"""
        os.makedirs(os.path.dirname(self.trajectory_file), exist_ok=True)
        data = {
            'trajectories': [
                {'path': path, 'loss': loss}
                for path, loss in self.trajectory_pool
            ]
        }
        with open(self.trajectory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"保存了 {len(self.trajectory_pool)} 条轨迹到 {self.trajectory_file}")
    
    def load_trajectories(self) -> bool:
        """从 JSON 文件加载轨迹，如果不存在则返回 False"""
        if not os.path.exists(self.trajectory_file):
            return False
        
        try:
            with open(self.trajectory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.trajectory_pool = [
                (item['path'], item['loss'])
                for item in data['trajectories']
            ]
            
            # 确保有 40 条
            while len(self.trajectory_pool) < 40:
                base_path = list(range(self.total_layers))
                self.trajectory_pool.append((base_path, 0.0))
            
            return True
        except Exception as e:
            print(f"加载轨迹失败：{e}")
            return False
        