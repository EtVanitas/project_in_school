"""Stage1B: RL 网络训练 + LoRA 微调"""

import torch
import torch.nn.functional as F
from typing import List, Dict
import os
import sys
import random

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, RLConfig, LoRAConfig
from models.networks import MLPActorCritic
from models.model_loader import LayeredModelLoader
from models.model_manager import ModelManager
from utils.data_manager import DataManager
from utils.components import train_actor_critic_batch, init_rope, precompute_rope, rebuild_base_probs_batch


class Stage1BTrainer:
    """阶段 1B 训练器（RL 网络训练 + LoRA 微调）"""
    
    def __init__(self):
        # 直接使用全局配置
        self.model_config = ModelConfig()
        self.rl_config = RLConfig()
        self.lora_config = LoRAConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化数据管理器、模型管理器和分层模型加载器
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.layered_loader = LayeredModelLoader()
        
        # 初始化 RoPE
        init_rope(self.layered_loader, self.device)
        
        # 初始化 Actor-Critic
        self.actor_critic = MLPActorCritic().to(self.device)

        # 尝试加载最新 LoRA 模型和 RL 模型
        lora_loaded = self.model_manager.load_lora_model(self.layered_loader, stage_group='stage1', use_best=False)
        rl_loaded = self.model_manager.load_rl_model(self.actor_critic, None, stage_group='stage1', use_best=False)
        
        if not lora_loaded:
            print("警告: 未找到 LoRA 模型，将使用初始权重")
        if not rl_loaded:
            print("警告: 未找到 RL 模型，Actor-Critic 将随机初始化")
        
        # 优化器
        self.lora_optimizer = torch.optim.AdamW(
            self.layered_loader.get_trainable_params(),
            lr=self.lora_config.lr,
            betas=self.lora_config.betas,
            weight_decay=self.lora_config.weight_decay
        )
        # 使用单一优化器管理 Actor-Critic 的所有参数
        self.rl_optimizer = torch.optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.rl_config.lr,
            betas=self.rl_config.betas,
            weight_decay=self.rl_config.weight_decay
        )
    
    def train_epoch(self, cleanup_after: bool = True) -> Dict:
        """训练单个 epoch"""
        print(f"\n{'='*60}")
        print(f"Stage1B 训练流程")
        print(f"{'='*60}")
        
        # 加载路径记录
        path_records = self.data_manager.load_path_records()
        if not path_records:
            print("警告: 没有路径记录，跳过 LoRA 训练")
            lora_stats = {'skipped': True}
        else:
            # 训练 LoRA
            lora_stats = self._train_lora(path_records)
        
        # 加载轨迹数据
        all_triples = self.data_manager.load_all_trajectories()
        if not all_triples:
            print("警告: 没有找到轨迹数据，跳过 RL 训练")
            rl_stats = {'skipped': True}
        else:
            # 训练 Actor-Critic
            rl_stats = self._train_actor_critic(all_triples, cleanup_after)
        
        print(f"\n✓ Stage1B 完成！")
        print(f"  - LoRA: {lora_stats}")
        print(f"  - RL: {rl_stats}\n")
        
        return {
            'lora': lora_stats,
            'rl': rl_stats
        }
    
    def _train_lora(self, path_records: Dict) -> Dict:
        """训练 LoRA（按 path 分组，相同长度样本批处理） """
        print(f"\n开始 LoRA 训练...")
        print(f"共有 {len(path_records)} 个不同的 path")
        
        total_loss = 0
        total_samples = 0
        processed_paths = 0
        
        # 逐个 path 训练
        for path_key, sample_list in path_records.items():
            try:
                # 恢复 path tuple（兼容字符串和 tuple 两种格式）
                if isinstance(path_key, str):
                    import ast
                    path = ast.literal_eval(path_key)  # 安全地解析字符串
                elif isinstance(path_key, (list, tuple)):
                    path = list(path_key)  # 已经是列表或元组，直接转换
                else:
                    print(f"警告: 未知的 path 格式: {type(path_key)}，跳过")
                    continue
                
                # 按长度分组样本
                length_groups = {}
                for sample_info in sample_list:
                    len_key = sample_info['len_key']
                    if len_key not in length_groups:
                        length_groups[len_key] = []
                    length_groups[len_key].append(sample_info['sample_name'])
                
                # 对每个长度组批处理训练
                for len_key, sample_names in length_groups.items():
                    seq_len = int(len_key.replace('len_', ''))
                    
                    # 如果样本数太多，需要分批处理
                    max_batch_size = self.model_config.batch_size  # 从配置读取，默认4
                    
                    # 按 max_batch_size 分组
                    for batch_start in range(0, len(sample_names), max_batch_size):
                        batch_sample_names = sample_names[batch_start:batch_start + max_batch_size]
                        
                        # 批量加载样本
                        batch_input_ids, batch_base_data = self.data_manager.load_batch_samples(
                            batch_sample_names, len_key, self.device
                        )
                        
                        if batch_input_ids is None:
                            continue
                        
                        # 批量训练
                        loss = self._train_lora_batch(batch_input_ids, batch_base_data, path, seq_len)
                        
                        total_loss += loss * len(batch_input_ids)
                        total_samples += len(batch_input_ids)
                        
                        # 立即释放批次数据
                        del batch_input_ids, batch_base_data
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                processed_paths += 1
                
                if processed_paths % 10 == 0:
                    avg_loss = total_loss / max(1, total_samples)
                    print(f"  已处理 {processed_paths} 个 path, {total_samples} 样本, 平均 loss={avg_loss:.4f}")
                
            except Exception as e:
                print(f"Path {path_key} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / max(1, total_samples)
        print(f"LoRA 训练完成：平均 loss={avg_loss:.4f}, 样本数={total_samples}, path数={processed_paths}")
        
        # 保存 LoRA 模型
        self.model_manager.save_lora_model(self.layered_loader, avg_loss, stage_group='stage1')
        
        return {
            'avg_loss': avg_loss,
            'sample_count': total_samples,
            'path_count': processed_paths
        }
    
    def _train_lora_batch(self, batch_input_ids: torch.Tensor, batch_base_data: List[Dict], 
                         path: List[int], seq_len: int) -> float:
        """训练一个批次的样本（相同长度，相同 path，真正GPU批处理）"""
        batch_size = batch_input_ids.shape[0]
        
        # 预计算 RoPE（直接调用，无需包装）
        self.precomputed_cos, self.precomputed_sin = precompute_rope(
            seq_len, self.device, self.model_config.hidden_size
        )
        
        # 批处理前向传播
        total_loss = self._forward_with_grad_batch(batch_input_ids, batch_base_data, path)
        
        # 计算平均 loss
        avg_loss = total_loss / batch_size
        
        # 反向传播
        avg_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.layered_loader.get_trainable_params(), 
                                      self.rl_config.grad_clip)
        
        # 更新参数
        self.lora_optimizer.step()
        self.lora_optimizer.zero_grad()
        
        # 获取 loss 值后立即释放梯度图
        loss_value = avg_loss.item()
        del total_loss, avg_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss_value
    
    def _forward_with_grad_batch(self, batch_input_ids: torch.Tensor, 
                                batch_base_data: List[Dict],
                                path: List[int]) -> torch.Tensor:
        """沿路径批处理前向传播（带梯度）"""
        batch_size = batch_input_ids.shape[0]
        
        # 执行整个路径
        hidden = self.layered_loader.model.model.embed_tokens(batch_input_ids)  # [batch_size, seq_len, hidden]
        
        # 批量重建基准分布（使用通用函数）
        batch_base_probs = rebuild_base_probs_batch(
            batch_base_data, self.model_config.vocab_size, self.device
        )
        
        # [显存优化] 定义单层的 forward 函数（用于梯度检查点）
        def layer_forward(h, layer_idx, cos, sin):
            """单层的前向传播（需要保存输入以支持 checkpoint）"""
            return self.layered_loader.layers[layer_idx](
                h,
                position_embeddings=(cos, sin)
            )
        
        # 沿路径执行所有层（使用梯度检查点）
        for step_idx, layer_idx in enumerate(path):
            # [关键优化] 使用梯度检查点，不保存中间激活值
            # 反向传播时会重新计算这一层
            hidden = torch.utils.checkpoint.checkpoint(
                layer_forward,
                hidden,
                layer_idx,
                self.precomputed_cos,
                self.precomputed_sin,
                use_reentrant=False  # 推荐使用非重入模式
            )
            
            # 到达终点时计算交叉熵
            if step_idx == len(path) - 1:
                last_hidden = hidden[:, -1, :]  # [batch_size, hidden_size]
                
                # 批量计算所有样本的logits和log_probs
                logits = self.layered_loader.model.lm_head(last_hidden)  # [batch_size, vocab_size]
                model_log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, vocab_size]
                
                # 对每个样本计算loss并求和
                total_loss = 0
                for sample_idx in range(batch_size):
                    ce_loss = -(batch_base_probs[sample_idx] * model_log_probs[sample_idx]).sum()
                    total_loss += ce_loss
                
                return total_loss
        
        raise ValueError("路径未到达终点")
    
    def _train_actor_critic(self, all_triples: List[Dict], cleanup_after: bool = True) -> Dict:
        """训练 Actor-Critic 网络"""
        print(f"\n开始 Actor-Critic 训练...")
        print(f"共加载 {len(all_triples)} 条三元组")
        
        # 分批训练
        batch_size = self.model_config.batch_size * 8
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        batch_count = 0
        
        random.shuffle(all_triples)
        
        for batch_start in range(0, len(all_triples), batch_size):
            batch_triples = all_triples[batch_start:batch_start + batch_size]
            
            try:
                # 训练这个 batch
                actor_loss, critic_loss, batch_loss = train_actor_critic_batch(
                    actor_critic=self.actor_critic,
                    rl_optimizer=self.rl_optimizer,
                    batch=batch_triples,
                    lm_head=self.layered_loader.model.lm_head,
                    device=self.device,
                    grad_clip=self.rl_config.grad_clip
                )
                
                # 检查 loss 是否为 nan 或 inf
                if not (torch.isfinite(torch.tensor(actor_loss)) and torch.isfinite(torch.tensor(critic_loss))):
                    print(f"  警告: 检测到 nan/inf loss (actor={actor_loss}, critic={critic_loss})，跳过该批次")
                    continue
                
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_loss += batch_loss
                batch_count += 1
                
                # 每 100 个 batch 输出一次进度
                if batch_count % 100 == 0:
                    avg_actor = total_actor_loss / batch_count
                    avg_critic = total_critic_loss / batch_count
                    avg_total = total_loss / batch_count
                    print(f"  [进度] 已训练 {batch_count} batches, Actor={avg_actor:.4f}, Critic={avg_critic:.4f}, Total={avg_total:.4f}")
                
            except Exception as e:
                import traceback
                print(f"RL 批次训练失败: {e}")
                print(f"详细错误信息:\n{traceback.format_exc()}")
                continue
        
        if batch_count == 0:
            return {'skipped': True}
        
        avg_actor_loss = total_actor_loss / batch_count
        avg_critic_loss = total_critic_loss / batch_count
        avg_total_loss = total_loss / batch_count
        
        print(f"Actor-Critic 训练完成：总 loss={avg_total_loss:.4f}")
        
        # 保存 RL 模型
        self.model_manager.save_rl_model(self.actor_critic, None, 0, avg_total_loss, stage_group='stage1')
        
        # 清理轨迹文件
        if cleanup_after:
            self.data_manager.cleanup_training_data()
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'total_loss': avg_total_loss,
            'traj_count': len(all_triples)
        }
