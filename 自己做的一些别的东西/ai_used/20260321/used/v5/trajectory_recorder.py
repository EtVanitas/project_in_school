"""轨迹记录器 - 精简版 v4.0

功能：
1. 只记录必要信息：(iteration, activate_idx, remain_value)
2. 纯数据记录，不包含梯度
3. 用于长期记忆更新和可视化
"""

import torch
from typing import List, Dict, Tuple
from collections import defaultdict


class BatchTrajectoryRecorder:
    """精简版轨迹记录器（只记录必要信息）"""
    
    def __init__(self, max_trajectories: int = 100):
        self.max_trajectories = max_trajectories
        # 格式：inference_id -> [(iteration, activate_idx, remain_value), ...]
        self.trajectories: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
        self.current_recorders: Dict[str, dict] = {}  # 用于兼容旧代码
        self.completed_trajectories = []  # 用于长期记忆
    
    def start_inference(self, inference_id: str):
        """开始一个新的推理"""
        self.trajectories[inference_id] = []
        self.current_recorders[inference_id] = {'steps': [], 'successful': False}
    
    def record(self, iteration: int, activate_idx: int, remain_value: float):
        """
        记录单步轨迹（简化版）
        
        Args:
            iteration: 时间步
            activate_idx: activate 索引
            remain_value: remain 值（纯数据，无梯度）
        """
        # 记录到所有活跃的 inference 中
        for inference_id in self.current_recorders.keys():
            self.trajectories[inference_id].append((iteration, activate_idx, float(remain_value)))
    
    def record_step(self, inference_id: str, **kwargs):
        """兼容旧版本的接口（已废弃，但保留以避免报错）"""
        # 旧版本接口的简化实现
        if inference_id not in self.current_recorders:
            self.start_inference(inference_id)
        
        step_info = {
            'iteration': kwargs.get('iteration', 0),
            'activated_count': len(kwargs.get('activated_indices', [])),
        }
        self.current_recorders[inference_id]['steps'].append(step_info)
    
    def end_inference(self, inference_id: str, successful: bool = False):
        """结束一个推理"""
        if inference_id in self.current_recorders:
            self.current_recorders[inference_id]['successful'] = successful
            
            if successful and len(self.completed_trajectories) < self.max_trajectories:
                # 保存成功轨迹
                trajectory_data = {
                    'inference_id': inference_id,
                    'steps': self.trajectories.get(inference_id, []),
                }
                self.completed_trajectories.append(trajectory_data)
            
            del self.current_recorders[inference_id]
    
    def get_trajectory(self, inference_id: str) -> List[Tuple[int, int, float]]:
        """获取指定推理的完整轨迹"""
        return self.trajectories.get(inference_id, [])
    
    def get_statistics(self, inference_id: str) -> Dict:
        """获取轨迹统计信息"""
        trajectory = self.trajectories.get(inference_id, [])
        
        if len(trajectory) == 0:
            return {}
        
        # 统计每个 activate 的出现频率
        activate_freq = defaultdict(int)
        for _, act_idx, _ in trajectory:
            activate_freq[act_idx] += 1
        
        # 统计 remain 值的分布
        remain_values = [val for _, _, val in trajectory]
        
        return {
            'total_steps': len(trajectory),
            'unique_activates': len(activate_freq),
            'top_activates': sorted(activate_freq.items(), key=lambda x: x[1], reverse=True)[:10],
            'avg_remain_value': sum(remain_values) / len(remain_values) if remain_values else 0.0,
            'min_remain_value': min(remain_values) if remain_values else 0.0,
            'max_remain_value': max(remain_values) if remain_values else 0.0,
        }
    
    def clear(self, inference_id: str = None):
        """清除轨迹"""
        if inference_id:
            self.trajectories.pop(inference_id, None)
            self.current_recorders.pop(inference_id, None)
        else:
            self.trajectories.clear()
            self.current_recorders.clear()
            self.completed_trajectories.clear()
    
    def get_long_term_memory_data(self) -> Dict[int, List[torch.Tensor]]:
        """
        从完成的轨迹中提取长期记忆更新数据
        
        Returns:
            Dict[activate_idx, List[Tensor]] - 每个 activate 的成功轨迹数据
        """
        memory_data = defaultdict(list)
        
        for trajectory_data in self.completed_trajectories:
            steps = trajectory_data['steps']
            
            for step in steps:
                # 提取成功的激活信息
                if isinstance(step, tuple) and len(step) >= 3:
                    _, act_idx, remain_val = step[:3]
                    # 这里可以进一步处理，提取对应的 X 矩阵
                    # 目前简化为只记录 remain 值
                    memory_data[act_idx].append(torch.tensor(remain_val))
        
        return memory_data