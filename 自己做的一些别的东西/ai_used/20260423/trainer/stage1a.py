"""Stage1A: 批量路径搜索 + 轨迹收集"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import os
import sys
import random

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, ACTION_OUTPUT
from models import LayeredModelLoader
from models.model_manager import ModelManager
from utils.data_manager import DataManager
from utils.path_generator import SimplePathGenerator
from utils.components import build_state_dict, init_rope, precompute_rope, rebuild_base_probs_batch, select_best_paths_and_collect_triples


class Stage1ATrainer:
    """阶段 1A 训练器 - 批量路径搜索 + 轨迹收集"""
    
    def __init__(self):
        # 直接使用全局配置
        self.model_config = ModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化数据管理器、模型管理器、分层模型加载器和路径生成器
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.layered_loader = LayeredModelLoader()
        self.path_gen = SimplePathGenerator()
        
        # 初始化 RoPE
        init_rope(self.layered_loader, self.device)
        
        # 训练状态
        self.global_step = 0
        
        # 轨迹缓冲区（用于批量保存）
        self.traj_buffer: List[Dict] = []
        
        # 尝试加载最新模型续训
        self.model_manager.try_load_latest_lora_model(self.layered_loader, current_stage='stage1a')
    
    def train_epoch(self) -> Dict:
        """训练单个 epoch（批处理模式）
        
        遍历所有长度文件夹，每个文件夹随机采样一批次处理
        """
        print(f"\n{'='*60}")
        print(f"Stage1A 训练流程（批处理模式）")
        print(f"{'='*60}")
        
        # 加载路径记录
        path_records = self.data_manager.load_path_records()
        
        total_samples = 0
        total_loss = 0
        batch_count = 0
        
        # 遍历所有长度文件夹
        length_folders = self.data_manager.get_length_folders()
        
        for folder_name in length_folders:
            seq_len = int(folder_name.replace('len_', ''))
            text_folder = os.path.join(self.data_manager.text_dir, folder_name)
            base_folder = os.path.join(self.data_manager.base_dir, folder_name)
            
            if not os.path.exists(base_folder):
                continue
            
            # 获取样本文件列表
            sample_files = self.data_manager.get_sample_files(text_folder)
            
            if len(sample_files) < self.model_config.batch_size:
                print(f"  跳过 {folder_name}: 样本数不足 ({len(sample_files)} < {self.model_config.batch_size})")
                continue
            
            try:
                # 随机采样批次（最多 batch_size 个）
                actual_batch_size = min(self.model_config.batch_size, len(sample_files))
                batch_samples = random.sample(sample_files, actual_batch_size)
                
                # 处理批次
                batch_results = self._process_batch(seq_len, batch_samples, text_folder, base_folder)
                
                # 更新路径记录
                for result in batch_results:
                    path_tuple = tuple(result['best_path'])
                    if path_tuple not in path_records:
                        path_records[path_tuple] = []
                    
                    # 添加样本信息
                    sample_info = {
                        'len_key': f"len_{seq_len}",
                        'sample_name': result['sample_name']
                    }
                    path_records[path_tuple].append(sample_info)
                    
                    total_samples += 1
                    total_loss += result['best_loss']
                
                batch_count += 1
                self.global_step += 1
                
                # 每 128 批次保存一次（路径池 + 路径记录 + 轨迹数据）
                if batch_count % 128 == 0:
                    avg_loss = total_loss / max(1, total_samples)
                    print(f"\n  [进度] 已处理 {batch_count} 批, {total_samples} 样本, 平均 loss={avg_loss:.4f}")

                    self.path_gen.save_trajectories()
                    self.data_manager.save_path_records(path_records)
                    if len(self.traj_buffer) > 0:
                        traj_count = len(self.traj_buffer)
                        self.data_manager.save_trajectory_batch(self.traj_buffer, self.global_step)
                        self.traj_buffer.clear()
                        print(f"  [保存] 已保存 {traj_count} 条轨迹到 batch_{self.global_step:06d}.pt\n")
                
                # 每 1024 批次保存一次
                if batch_count % 1024 == 0:
                    avg_loss = total_loss / max(1, total_samples)
                    self.model_manager.save_lora_model(self.layered_loader, avg_loss, stage_group='stage1')
                    print(f"  [模型] LoRA 模型已保存 (loss={avg_loss:.4f})\n")
                
            except Exception as e:
                print(f"批次处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / max(1, total_samples)
        print(f"\n训练完成：平均 loss={avg_loss:.4f}, 样本数={total_samples}, 批次数={batch_count}")
        
        # 最终保存
        self.path_gen.save_trajectories()
        self.data_manager.save_path_records(path_records)
        traj_count = len(self.traj_buffer)
        if traj_count > 0:
            self.data_manager.save_trajectory_batch(self.traj_buffer, self.global_step)
            self.traj_buffer.clear()
            print(f"  [最终保存] 已保存 {traj_count} 条剩余轨迹")
        
        # 保存最终的 LoRA 模型
        self.model_manager.save_lora_model(self.layered_loader, avg_loss, stage_group='stage1')
        print(f"  [最终保存] LoRA 模型已保存 (loss={avg_loss:.4f})")
        
        print(f"\n✓ Stage1A 完成！")
        print(f"  - 平均 Loss: {avg_loss:.4f}")
        print(f"  - 样本数量：{total_samples}")
        print(f"  - 轨迹数量：{traj_count}\n")
        
        return {
            'avg_loss': avg_loss,
            'sample_count': total_samples,
            'batch_count': batch_count
        }
    
    def _process_batch(self, seq_len: int, sample_files: List[str], 
                       text_folder: str, base_folder: str) -> List[Dict]:
        """处理一个批次的样本（真正的GPU批处理：batch_size个样本同时经过每条path）"""

        # 预计算 RoPE
        self.precomputed_cos, self.precomputed_sin = precompute_rope(
            seq_len, self.device, self.model_config.hidden_size
        )
        
        # 批量加载样本数据
        batch_input_ids, batch_base_data, batch_sample_names = self.data_manager.load_batch_from_folders(
            sample_files, text_folder, base_folder, self.device
        )
        
        if batch_input_ids is None:
            print(f"  跳过批次：无法加载任何样本")
            return []
        
        # 采样多条路径
        paths = self.path_gen.generate_multiple_paths()
        
        # 对每条路径进行批处理前向传播
        all_losses = []
        all_trajectories = []
        
        with torch.no_grad():
            for path_idx, path in enumerate(paths):
                batch_trajectories, batch_ce_losses = self._forward_along_path_batch(
                    batch_input_ids, batch_base_data, path
                )
                
                all_losses.append(batch_ce_losses)  # [batch_size]
                all_trajectories.append(batch_trajectories)
                
                # [显存优化] 对于长序列，每条路径后清理显存
                if seq_len > 1000:
                    del batch_trajectories, batch_ce_losses
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # 为每个样本选择最优路径并收集所有路径的三元组
        results, all_triples = select_best_paths_and_collect_triples(
            paths, all_losses, all_trajectories
        )
        
        # 为每个样本更新路径池
        for sample_idx, result in enumerate(results):
            self.path_gen.add_trajectory(result['best_path'], result['best_loss'])
            # 添加样本名称到结果中
            result['sample_name'] = batch_sample_names[sample_idx]
        
        # 添加到缓冲区
        self.traj_buffer.extend(all_triples)
        
        # 释放显存：清理中间变量
        del all_losses, all_trajectories, batch_input_ids, batch_base_data
        torch.cuda.empty_cache()
        
        return results
    
    def _forward_along_path_batch(self, batch_input_ids: torch.Tensor, 
                                  batch_base_data: List[Dict], 
                                  path: List[int]) -> Tuple[List[List[Tuple[Dict, int]]], List[float]]:
        """沿路径批处理前向传播（无梯度），记录状态和动作"""
        batch_size = batch_input_ids.shape[0]
        
        # 执行整个路径（包括前9层固定层 + 动态层）
        hidden = self.layered_loader.model.model.embed_tokens(batch_input_ids)  # [batch_size, seq_len, hidden]
        
        batch_trajectories = [[] for _ in range(batch_size)]
        batch_ce_losses = [None] * batch_size
        
        # 批量重建基准分布（使用通用函数）
        batch_base_probs = rebuild_base_probs_batch(
            batch_base_data, self.model_config.vocab_size, self.device
        )
        
        # 沿路径执行所有层
        for step_idx, layer_idx in enumerate(path):
            # 执行当前层（批处理）
            hidden = self.layered_loader.layers[layer_idx](
                hidden,
                position_embeddings=(self.precomputed_cos, self.precomputed_sin)
            )
            
            # 只处理动态层（layer_idx >= 9）
            if layer_idx >= self.model_config.fixed_layers:
                # 提取每个样本的状态（最后一个token的隐藏状态）
                last_hidden = hidden[:, -1, :]  # [batch_size, hidden_size]
                step_idx_value = step_idx + 1
                
                # 确定动作
                if step_idx == len(path) - 1:
                    # 最后一层：批量计算所有样本的交叉熵
                    action = ACTION_OUTPUT
                    logits = self.layered_loader.model.lm_head(last_hidden)  # [batch_size, vocab_size]
                    model_log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, vocab_size]
                    
                    # 为每个样本计算loss
                    for sample_idx in range(batch_size):
                        ce_loss = -(batch_base_probs[sample_idx] * model_log_probs[sample_idx]).sum().item()
                        batch_ce_losses[sample_idx] = ce_loss
                else:
                    next_layer = path[step_idx + 1]
                    action = next_layer - self.model_config.fixed_layers
                
                # 为每个样本构建状态字典（必须逐个，因为输出是Python dict）
                for sample_idx in range(batch_size):
                    state_dict = build_state_dict(
                        hidden_state=last_hidden[sample_idx],  # [hidden_size]
                        layer_idx=layer_idx,
                        step_idx=step_idx_value,
                        seq_len=batch_input_ids.shape[1],
                    )
                    batch_trajectories[sample_idx].append((state_dict, action))
        
        return batch_trajectories, batch_ce_losses
