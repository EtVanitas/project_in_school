"""LSTM策略引导的动态路径微调

核心特性:
- 使用RNNActorCritic (LSTM)进行策略预测
- 直接使用词表概率分布作为输入
- LoRA和RL联合训练
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

from config import ModelConfig, RLConfig, LoRAConfig, ACTION_OUTPUT, DEFAULT_DEVICE, DEFAULT_DTYPE
from models.networks import RNNActorCritic
from models.model_loader import LayeredModelLoader
from models.model_manager import ModelManager
from utils.data_loader import DataLoader
from utils.components import init_rope, compute_rope, train_actor_critic, forward_with_grad


class Trainer:
    """训练器 - RNN策略引导的动态路径微调"""
    
    def __init__(self):
        # 配置
        self.model_config = ModelConfig()
        self.rl_config = RLConfig()
        self.lora_config = LoRAConfig()
        self.device = DEFAULT_DEVICE
        self.dtype = DEFAULT_DTYPE
        
        # 初始化组件
        self.data_loader = DataLoader()
        self.model_manager = ModelManager()
        self.layered_loader = LayeredModelLoader()
        
        # 初始化 RNN Actor-Critic
        self.actor_critic = RNNActorCritic().to(dtype=self.dtype, device=self.device)
        
        # 初始化 RoPE
        init_rope(self.layered_loader, self.device)
        
        # 尝试加载最新模型(优先最优模型,其次最新)
        if not self.model_manager.load_lora_model(self.layered_loader, use_best=True):
            print("未找到 LoRA 模型，使用初始权重")
        
        if not self.model_manager.load_rl_model(self.actor_critic, None, use_best=True):
            print("未找到 RL 模型，使用随机初始化")
        
        # 优化器
        self.lora_optimizer = torch.optim.AdamW(
            self.layered_loader.get_trainable_params(),
            lr=self.lora_config.lr,
            betas=self.lora_config.betas,
            weight_decay=self.lora_config.weight_decay
        )
        self.rl_optimizer = torch.optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.rl_config.lr,
            betas=self.rl_config.betas,
            weight_decay=self.rl_config.weight_decay
        )
        
        # 轨迹缓冲区 - 存储完整轨迹序列
        self.traj_buffer: List[Dict] = []
        self.token_count_since_last_rl_update = 0
    
    def _explore_single_token(self, current_input_ids: torch.Tensor, 
                                target_token: torch.Tensor,
                                seq_len: int) -> Tuple[List[int], float, List[Dict]]:
        """对单个token执行路径探索 (基于Rollout机制)
        
        Args:
            current_input_ids: 当前输入序列 [seq_len]
            target_token: 目标token (scalar)
            seq_len: 当前序列长度
        
        Returns:
            best_path: 最优路径的动作序列
            best_ce_loss: 最小交叉熵loss
            all_completed_paths: 所有完成的路径和rollout选项
        """
        max_path_length = self.model_config.max_steps - self.model_config.fixed_layers
        
        # 预计算 RoPE
        cos, sin = compute_rope(seq_len, self.device, self.model_config.hidden_size)
        
        # 前9层固定执行（无梯度）
        with torch.no_grad():
            hidden = self.layered_loader.model.model.embed_tokens(current_input_ids.unsqueeze(0))
            
            for layer_idx in range(self.model_config.fixed_layers):
                layer = self.layered_loader.layers[layer_idx]
                hidden = layer(hidden, position_embeddings=(cos, sin))
        
        # 重置LSTM hidden state (新token开始)
        self.actor_critic.reset_hidden_state()
        
        # 初始化路径
        path = []
        trajectories = []
        actual_layer = self.model_config.fixed_layers - 1  # 初始在layer 8
        step_idx = 0
        
        completed_paths = []
        
        # 单路径探索
        while len(path) < max_path_length:
            with torch.no_grad():
                # Final Norm
                hidden = self.layered_loader.model.model.norm(hidden)
                last_hidden = hidden[:, -1, :]
                
                # LM Head
                logits = self.layered_loader.model.lm_head(last_hidden)
                vocab_probs = F.softmax(logits, dim=-1).squeeze(0)  # [151936]
                
                # 过滤低概率token，只保留概率>阈值的token
                threshold = self.rl_config.prob_threshold
                mask = vocab_probs > threshold
                vocab_probs_filtered = vocab_probs * mask.float()  # 低于阈值的设为0
                
                # 获取target_token的预测概率 (用于RL价值评估，使用原始概率)
                target_prob = vocab_probs[target_token].item()
                
                # 裁剪概率到 [0, 1] 范围，避免数值异常
                target_prob = max(0.0, min(1.0, target_prob))
                
                # 构建状态字典 (用于RL轨迹记录，使用过滤后的概率)
                state_dict = {
                    'prob_vector': vocab_probs_filtered,  # [151936] 过滤后的词表概率
                    'layer_idx': actual_layer,
                }
                
                # LSTM策略网络采样动作 (使用过滤后的概率)
                action_logits, _ = self.actor_critic(
                    prob_vector=vocab_probs_filtered,  # [151936] 1D tensor
                    layer_index=actual_layer,  # 标量
                    is_reset=False
                )
                action_probs = F.softmax(action_logits, dim=-1)
            
            # 采样动作
            if len(path) >= max_path_length - 1:
                action = ACTION_OUTPUT
            else:
                action = torch.multinomial(action_probs, 1).item()
            
            # 记录如果现在输出会怎样
            completed_paths.append({
                'path': path.copy() + [ACTION_OUTPUT],
                'trajectories': trajectories.copy() + [(state_dict, ACTION_OUTPUT)],
                'reward': target_prob,
            })
            
            # 继续主线
            trajectories.append((state_dict, action))
            path.append(action)
            
            # 执行动作
            next_layer = self.model_config.fixed_layers + action  # 9 + [0-18] = [9-27]

            if action == ACTION_OUTPUT or next_layer >= self.model_config.num_layers:
                break
            else:
                # 跳转动作：执行到目标层
                layer = self.layered_loader.layers[next_layer]
                hidden = layer(hidden, position_embeddings=(cos, sin))
                
                actual_layer = next_layer
                step_idx += 1

        # 选择最优路径: 从所有rollout选项中找reward最大的
        if not completed_paths:
            return [], float('inf'), []
        
        best_option = max(completed_paths, key=lambda x: x['reward'])
        best_path = best_option['path']
        best_reward = best_option['reward']  # reward现在是概率, 越大越好
        
        return best_path, best_reward, completed_paths
    
    def _update_rl_network(self):
        """更新RL网络(Actor-Critic)"""
        
        if not self.traj_buffer:
            return
        print(f"\n开始RL网络更新...")
        print(f"轨迹缓冲区大小: {len(self.traj_buffer)} 条完整轨迹")
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        trajectory_count = 0
        
        # 逐条轨迹训练 (保持时序完整性)
        for traj_idx, trajectory in enumerate(self.traj_buffer):
            try:
                actor_loss, critic_loss, traj_loss = train_actor_critic(
                    actor_critic=self.actor_critic,
                    rl_optimizer=self.rl_optimizer,
                    trajectory=trajectory,
                    device=self.device,
                )
                
                # 检查 loss 是否为 nan 或 inf
                if not (torch.isfinite(torch.tensor(actor_loss)) and torch.isfinite(torch.tensor(critic_loss))):
                    print(f"    Trajectory {traj_idx}: 警告 nan/inf loss, 跳过")
                    continue
                
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_loss += traj_loss
                trajectory_count += 1
                
                # 每10条轨迹或最后一条时输出进度
                if (traj_idx + 1) % 10 == 0 or traj_idx == len(self.traj_buffer) - 1:
                    print(f"    Trajectory {traj_idx + 1}/{len(self.traj_buffer)}: "
                          f"actor={actor_loss:.4f}, critic={critic_loss:.4f}")
                
            except Exception as e:
                import traceback
                print(f"RL 轨迹训练失败 (traj {traj_idx}): {e}")
                print(f"详细错误信息:\n{traceback.format_exc()}")
                continue
        
        if trajectory_count == 0:
            print("警告: 没有有效的轨迹,跳过RL更新")
            return
        
        avg_actor_loss = total_actor_loss / trajectory_count
        avg_critic_loss = total_critic_loss / trajectory_count
        avg_total_loss = total_loss / trajectory_count
        print(f"Actor-Critic 训练完成：actor loss={avg_actor_loss:.4f}, critic loss={avg_critic_loss:.4f}, 总 loss={avg_total_loss:.4f}")
        print(f"成功训练 {trajectory_count}/{len(self.traj_buffer)} 条轨迹")
        
        # 清空缓冲区
        self.traj_buffer.clear()
        torch.cuda.empty_cache()
        
        print("RL网络更新完成\n")
    
    def _process_single_pair(self, input_ids: torch.Tensor, 
                              output_ids: torch.Tensor) -> Dict:
        """处理单个input-output对
        
        逐个token进行teacher forcing训练:
        - 第1步: input -> 预测output[0]
        - 第2步: input + output[0] -> 预测output[1]
        - ...
        - 第n步: input + output[:n-1] -> 预测output[n]
        
        Returns:
            stats: 包含total_loss, processed_token_count (实际处理的token数)
        """
        total_ce_loss = 0
        token_count = 0  # 实际成功处理的token数
        
        # 对output的每个token进行训练
        for token_idx in range(len(output_ids)):
            # 构建当前输入: input + output[:token_idx]
            if token_idx > 0:
                current_input = torch.cat([input_ids, output_ids[:token_idx]])
            else:
                current_input = input_ids
            
            target_token = output_ids[token_idx]
            current_seq_len = len(current_input)
            
            try:
                # 步骤1: 探索最优路径 (无梯度)
                best_path, best_reward, all_completed_paths = self._explore_single_token(
                    current_input, target_token, current_seq_len
                )
                
                # 步骤2: 从路径中找reward最大的 (用于LoRA训练)
                best_option = max(all_completed_paths, key=lambda x: x['reward'])
                best_path = best_option['path']
                best_reward = best_option['reward']
                
                # 收集轨迹用于RL训练
                for path_option in all_completed_paths:
                    self.traj_buffer.append({
                        'states': [s for s, a in path_option['trajectories']],
                        'actions': [a for s, a in path_option['trajectories']],
                        'final_reward': path_option['reward'],
                        'length': len(path_option['trajectories'])
                    })
                
                # [调试] 输出最优路径信息
                if best_path is not None:
                    print(f"    Token {token_idx}: reward={best_reward:.4f}, path_len={len(best_path)}"
                          f"→ 最优路径: path={best_path}")
                
                # 步骤3: 带梯度沿最优路径前向传播 (用于LoRA训练)
                ce_loss_with_grad = forward_with_grad(
                    layered_loader=self.layered_loader,
                    input_ids=current_input,
                    path=best_path,
                    target_token=target_token,
                    seq_len=current_seq_len,
                    device=self.device,
                )
                
                # 步骤4: 立即反向传播更新LoRA
                ce_loss_with_grad.backward()
                torch.nn.utils.clip_grad_norm_(self.layered_loader.get_trainable_params(), 
                                               self.lora_config.grad_clip)
                self.lora_optimizer.step()
                self.lora_optimizer.zero_grad()
                
                # 累积统计信息
                total_ce_loss += ce_loss_with_grad.item()
                token_count += 1  # 成功处理一个token
                
                # 每处理完一个token,检查是否需要更新RL网络
                self.token_count_since_last_rl_update += 1
                should_update_rl = self.token_count_since_last_rl_update >= self.rl_config.rl_update_interval
                
                if should_update_rl:
                    print(f"\n  [RL更新] token_count={self.token_count_since_last_rl_update}, "
                          f"traj_count={len(self.traj_buffer)}")
                    self._update_rl_network()
                    self.token_count_since_last_rl_update = 0
                
                # 清理显存
                del best_path, all_completed_paths, ce_loss_with_grad
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Token {token_idx} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return {
            'total_loss': total_ce_loss,
            'token_count': token_count
        }
    
    def train_epoch(self, start_idx: int = 0, max_samples: int = None) -> Dict:
        """训练单个epoch,支持断点续训
        
        Args:
            start_idx: 起始样本索引(用于断点续训)
            max_samples: 最大处理样本数,None表示处理完整个数据集
        """
        print(f"\n{'='*60}")
        print(f"训练流程")
        print(f"{'='*60}")
        
        # 1. 加载数据集
        dataset = self.data_loader.load_jsonl_dataset()
        total_samples = len(dataset)
        print(f"数据集大小: {total_samples} 条样本")
        
        # 尝试从 checkpoint 恢复
        if start_idx == 0:
            start_idx = self.data_loader.load_checkpoint()
            if start_idx > 0:
                print(f"从 checkpoint 恢复，起始索引: {start_idx}")
        
        if max_samples is None:
            max_samples = total_samples - start_idx
        
        end_idx = min(start_idx + max_samples, total_samples)
        print(f"本次训练范围: [{start_idx}, {end_idx}), 共 {end_idx - start_idx} 个样本")
        
        # 2. 训练循环
        total_ce_loss = 0
        total_tokens = 0
        sample_count = 0
        
        for idx in range(start_idx, end_idx):
            try:
                sample = dataset[idx]
                input_text = sample['input']
                output_text = sample['output']
                
                # [调试输出] 开始处理样本
                print(f"\n[Sample {idx}] input_len={len(input_text)}, output_len={len(output_text)}")
                
                # Tokenize
                tokenizer = self.layered_loader.tokenizer
                input_ids, output_ids, seq_len = self.data_loader.tokenize_input_output_pair(
                    input_text, output_text, tokenizer, self.device
                )
                
                # 处理单个pair
                stats = self._process_single_pair(input_ids, output_ids)
                
                total_ce_loss += stats['total_loss']
                total_tokens += stats['token_count']
                sample_count += 1
                
                # [调试输出] 样本处理结果
                avg_loss = total_ce_loss / max(1, total_tokens)
                print(f"  [Sample {idx}] tokens={stats['token_count']}, avg_loss={avg_loss:.4f}, "
                      f"traj_buffer={len(self.traj_buffer)}")
                
                # 清理显存
                del input_ids, output_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 每5个样本保存checkpoint并更新模型
                if sample_count % 5 == 0:
                    avg_loss = total_ce_loss / max(1, total_tokens)
                    print(f"\n已处理 {sample_count} 个样本, 平均CE={avg_loss:.4f}")
                    
                    # 如果还有剩余的三元组,更新RL网络
                    if len(self.traj_buffer) > 0:
                        print(f"\n  [Checkpoint] 保存前更新剩余 {len(self.traj_buffer)} 个三元组")
                        self._update_rl_network()
                        self.token_count_since_last_rl_update = 0
                    
                    # 保存模型
                    self.model_manager.save_lora_model(self.layered_loader, avg_loss)
                    self.model_manager.save_rl_model(self.actor_critic, None, 0, avg_loss)
                    
                    # 保存 checkpoint (记录已处理的样本数)
                    current_idx = idx + 1  # 当前已处理到的索引
                    self.data_loader.save_checkpoint(processed_count=current_idx, avg_loss=avg_loss)
                    
            except Exception as e:
                print(f"跳过样本 {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 3. 处理剩余的缓冲区
        if len(self.traj_buffer) > 0:
            print(f"\n处理剩余的 {len(self.traj_buffer)} 个三元组")
            self._update_rl_network()
        
        # 4. 最终保存
        avg_ce_loss = total_ce_loss / max(1, total_tokens)
        self.model_manager.save_lora_model(self.layered_loader, avg_ce_loss)
        self.model_manager.save_rl_model(self.actor_critic, None, 0, avg_ce_loss)
        
        # 保存最终 checkpoint
        self.data_loader.save_checkpoint(processed_count=end_idx, avg_loss=avg_ce_loss)
        
        print(f"\nEpoch 完成！")
        print(f"  - 平均 CE Loss: {avg_ce_loss:.4f}")
        print(f"  - 样本数量: {sample_count}")
        print(f"  - Token总数: {total_tokens}\n")
        
        return {
            'avg_ce_loss': avg_ce_loss,
            'sample_count': sample_count,
            'total_tokens': total_tokens,
            'trajectory_count': len(self.traj_buffer)
        }
