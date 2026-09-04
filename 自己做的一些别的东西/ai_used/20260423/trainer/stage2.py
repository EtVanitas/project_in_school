"""Stage2: 策略引导的动态路径微调"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import os
import sys
import random

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, RLConfig, LoRAConfig, PathConfig, ACTION_OUTPUT
from models.networks import MLPActorCritic
from models.model_loader import LayeredModelLoader
from models.model_manager import ModelManager
from utils.data_manager import DataManager
from utils.components import build_state_dict, compute_path_values, train_actor_critic_batch, init_rope, precompute_rope


class PathInfo:
    """路径信息（用于 DFS 栈）"""
    def __init__(self, path: List[int], trajectories: List[Tuple[Dict, int]], 
                 hidden_at_branch: Optional[torch.Tensor] = None,
                 actual_layer: int = 8):
        self.path = path.copy()  # 动态动作序列（不包含前9层）
        self.trajectories = trajectories.copy()  # (state, action) 二元组
        self.hidden_at_branch = hidden_at_branch  # 分支点的 hidden state
        self.actual_layer = actual_layer  # 当前实际所在的层索引（从8开始）
        self.ce_loss: Optional[float] = None  # 交叉熵损失


class Stage2Trainer:
    """阶段 2 训练器（策略引导的动态路径微调）"""
    
    def __init__(self):
        # 直接使用全局配置
        self.model_config = ModelConfig()
        self.rl_config = RLConfig()
        self.lora_config = LoRAConfig()
        self.path_config = PathConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化数据管理器、模型管理器和分层模型加载器
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.layered_loader = LayeredModelLoader()

        # 初始化 Actor-Critic
        self.actor_critic = MLPActorCritic().to(self.device)
        
        # 初始化 RoPE
        init_rope(self.layered_loader, self.device)
        
        # 尝试加载最新模型（优先 Stage2 自己的，其次 Stage1B）
        self.model_manager.try_load_latest_lora_model(self.layered_loader, current_stage='stage2a')
        self.model_manager.try_load_latest_rl_model(self.actor_critic, None, current_stage='stage2b')
        
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
        
        # JSONL 数据集路径
        self.jsonl_path = r"C:\Users\35201\.vscode\ai\stage2_small_dataset.jsonl"
        self.checkpoint_file = os.path.join(self.path_config.trajectory_dir, "stage2_checkpoint.json")
        
        # 轨迹缓冲区（用于批量保存）
        self.traj_buffer: List[Dict] = []  # 累积所有探索路径的三元组
        self.token_count_since_last_rl_update = 0  # 上次RL更新后的token计数
    
    def _process_single_token_dfs(self, current_input_ids: torch.Tensor, 
                                   target_token: torch.Tensor,
                                   seq_len: int) -> Tuple[List[int], float, List[Tuple[Dict, int]], List[PathInfo]]:
        """对单个token执行无梯度DFS搜索
        
        Args:
            current_input_ids: 当前输入序列 [seq_len]
            target_token: 目标token (scalar)
            seq_len: 当前序列长度
        
        Returns:
            best_path: 最优路径的动作序列
            best_ce_loss: 最小交叉熵loss
            best_trajectories: 最优路径的轨迹二元组
            all_completed_paths: 所有完成的路径(用于生成三元组)
        """
        max_path_length = self.model_config.max_steps - self.model_config.fixed_layers
        
        # 预计算 RoPE
        cos, sin = precompute_rope(seq_len, self.device, self.model_config.hidden_size)
        
        # 前9层固定执行（无梯度）
        with torch.no_grad():
            hidden = self.layered_loader.model.model.embed_tokens(current_input_ids.unsqueeze(0))
            
            for layer_idx in range(self.model_config.fixed_layers):
                layer = self.layered_loader.layers[layer_idx]
                hidden = layer(hidden, position_embeddings=(cos, sin))
            
            # 第9层的输出作为分支起点
            hidden_at_start = hidden.clone()
        
        # 初始化路径栈（只包含动态层，前9层固定执行）
        initial_path = []  # 动态动作序列（初始为空）
        initial_trajectories = []
        path_stack = [PathInfo(
            path=initial_path,
            trajectories=initial_trajectories,
            hidden_at_branch=hidden_at_start,
            actual_layer=self.model_config.fixed_layers - 1  # 当前在第8层
        )]
        
        completed_paths = []
        
        # DFS 处理所有路径
        while path_stack:
            current_path_info = path_stack.pop()
            path = current_path_info.path
            trajectories = current_path_info.trajectories
            hidden = current_path_info.hidden_at_branch.clone()
            actual_layer = current_path_info.actual_layer
            
            step_idx = len(path)
            
            while len(path) < max_path_length:
                with torch.no_grad():
                    # 提取当前状态
                    last_hidden = hidden[:, -1, :]
                    
                    # 构建状态字典
                    state_dict = build_state_dict(
                        hidden_state=last_hidden.squeeze(0),
                        layer_idx=actual_layer,
                        step_idx=step_idx + 1,
                        seq_len=seq_len
                    )
                    
                    # 策略网络采样动作
                    logits = self.layered_loader.model.lm_head(last_hidden)
                    vocab_probs = F.softmax(logits, dim=-1)
                    
                    action_logits, _ = self.actor_critic(
                        vocab_probs=vocab_probs,
                        layer_index=torch.tensor([actual_layer], device=self.device),
                        context_length=torch.tensor([seq_len], device=self.device),
                        step_idx=torch.tensor([step_idx + 1], device=self.device)
                    )
                    action_probs = F.softmax(action_logits, dim=-1)[0]
                    
                    # 检查是否产生分支
                    should_branch, branch_actions = self._should_branch(action_probs)
                    
                    if should_branch and len(branch_actions) > 1:
                        # 产生分支：为每个分支动作创建新路径
                        for branch_action in branch_actions:
                            new_path = path.copy()
                            new_path.append(branch_action)
                            
                            new_trajectories = trajectories.copy()
                            new_trajectories.append((state_dict, branch_action))
                            
                            # 保留分支点的 hidden 和 actual_layer
                            new_path_info = PathInfo(
                                path=new_path,
                                trajectories=new_trajectories,
                                hidden_at_branch=hidden.clone(),
                                actual_layer=actual_layer
                            )
                            path_stack.append(new_path_info)
                        
                        # 当前路径不需要继续探索，直接处理下一条路径
                        break
                    else:
                        # 不分支，正常采样一个动作
                        # 如果接近最大步数，强制输出
                        if len(path) >= max_path_length - 1:
                            action = ACTION_OUTPUT
                        else:
                            action = torch.multinomial(action_probs, 1).item()
                        
                        # 记录二元组
                        trajectories.append((state_dict, action))
                        path.append(action)
                    
                    # 执行动作
                    if action == ACTION_OUTPUT:
                        # 输出动作：计算交叉熵
                        logits = self.layered_loader.model.lm_head(last_hidden)
                        ce_loss = F.cross_entropy(logits, target_token.unsqueeze(0))
                        
                        # 记录输出动作到轨迹（但不加入 path）
                        trajectories.append((state_dict, ACTION_OUTPUT))
                        
                        current_path_info.ce_loss = ce_loss.item()
                        completed_paths.append(current_path_info)
                        break
                    else:
                        # 跳转动作：直接执行目标层
                        next_layer = action + self.model_config.fixed_layers
                        
                        # 只执行下一层（不是所有中间层）
                        if next_layer < len(self.layered_loader.layers):
                            layer = self.layered_loader.layers[next_layer]
                            hidden = layer(hidden, position_embeddings=(cos, sin))
                        
                        # 更新实际所在的层和步数索引
                        actual_layer = next_layer
                        step_idx += 1
            else:
                # 超过最大长度，强制输出
                with torch.no_grad():
                    last_hidden = hidden[:, -1, :]
                    logits = self.layered_loader.model.lm_head(last_hidden)
                    ce_loss = F.cross_entropy(logits, target_token.unsqueeze(0))
                    
                    # 添加输出动作到轨迹（但不加入 path）
                    state_dict = build_state_dict(
                        hidden_state=last_hidden.squeeze(0),
                        layer_idx=actual_layer,
                        step_idx=step_idx + 1,
                        seq_len=seq_len
                    )
                    trajectories.append((state_dict, ACTION_OUTPUT))
                    # 注意：path 不添加 ACTION_OUTPUT，因为它不是跳转动作
                    
                    current_path_info.ce_loss = ce_loss.item()
                    completed_paths.append(current_path_info)
        
        # 选择最优路径（CE Loss 最小）
        if not completed_paths:
            raise RuntimeError("没有找到完成的路径")
        
        best_path_info = min(completed_paths, key=lambda p: p.ce_loss if p.ce_loss is not None else float('inf'))
        
        if best_path_info.ce_loss is None:
            raise RuntimeError("最优路径没有 loss")
        
        # [调试输出] DFS搜索统计
        print(f"    [DFS] 探索{len(completed_paths)}条路径, 最优loss={best_path_info.ce_loss:.4f}, "
              f"路径长度={len(best_path_info.path)}, "
              f"平均loss={sum(p.ce_loss for p in completed_paths)/len(completed_paths):.4f}")
        
        return best_path_info.path, best_path_info.ce_loss, best_path_info.trajectories, completed_paths
    
    def _forward_along_path_with_grad(self, input_ids: torch.Tensor, path: List[int],
                                       target_token: torch.Tensor, seq_len: int) -> torch.Tensor:
        """沿给定路径有梯度地前向传播(使用梯度检查点)
        
        Args:
            input_ids: 输入 token IDs [seq_len]
            path: 动态动作序列（不包含前9层）
            target_token: 目标token
            seq_len: 序列长度
        
        Returns:
            ce_loss: 交叉熵损失 tensor（带计算图）
        """
        # 预计算 RoPE
        cos, sin = precompute_rope(seq_len, self.device, self.model_config.hidden_size)
        
        # 前9层固定执行
        hidden = self.layered_loader.model.model.embed_tokens(input_ids.unsqueeze(0))
        for layer_idx in range(self.model_config.fixed_layers):
            layer = self.layered_loader.layers[layer_idx]
            hidden = layer(hidden, position_embeddings=(cos, sin))
        
        # 定义单层的 forward 函数（用于梯度检查点）
        def layer_forward(h, layer_idx, cos, sin):
            """单层的前向传播（需要保存输入以支持 checkpoint）"""
            return self.layered_loader.layers[layer_idx](
                h,
                position_embeddings=(cos, sin)
            )
        
        step_idx = 0
        actual_layer = self.model_config.fixed_layers - 1
        
        # 沿路径执行所有层（使用梯度检查点）
        for action in path:
            if action == ACTION_OUTPUT:
                # 输出动作：计算交叉熵
                last_hidden = hidden[:, -1, :]
                logits = self.layered_loader.model.lm_head(last_hidden)
                ce_loss = F.cross_entropy(logits, target_token.unsqueeze(0))
                return ce_loss
            else:
                # 跳转动作：直接执行目标层
                next_layer = action + self.model_config.fixed_layers
                
                # 只执行下一层（不是所有中间层）
                if next_layer < len(self.layered_loader.layers):
                    # [关键优化] 使用梯度检查点，不保存中间激活值
                    hidden = torch.utils.checkpoint.checkpoint(
                        layer_forward,
                        hidden,
                        next_layer,
                        cos,
                        sin,
                        use_reentrant=False  # 推荐使用非重入模式
                    )
                
                # 更新实际层和步数索引
                actual_layer = next_layer
                step_idx += 1
        
        # [兜底] 如果路径结束仍未输出，强制计算loss
        last_hidden = hidden[:, -1, :]
        logits = self.layered_loader.model.lm_head(last_hidden)
        ce_loss = F.cross_entropy(logits, target_token.unsqueeze(0))
        return ce_loss
    
    def _process_single_pair(self, input_ids: torch.Tensor, 
                              output_ids: torch.Tensor) -> Dict:
        """处理单个input-output对
        
        逐个token进行teacher forcing训练:
        - 第1步: input -> 预测output[0]
        - 第2步: input + output[0] -> 预测output[1]
        - ...
        - 第n步: input + output[:n-1] -> 预测output[n]
        
        Returns:
            stats: 包含total_loss, token_count
        """
        total_ce_loss = 0
        token_count = len(output_ids)
        
        # [调试输出] 样本信息
        print(f"  \u25b6 处理样本: input_len={len(input_ids)}, output_len={token_count}")
        
        for token_idx in range(token_count):
            # 构建当前输入: input + output[:token_idx]
            if token_idx > 0:
                current_input = torch.cat([input_ids, output_ids[:token_idx]])
            else:
                current_input = input_ids
            
            target_token = output_ids[token_idx]
            current_seq_len = len(current_input)
            
            try:
                # 步骤1: 无梯度DFS搜索最优路径
                best_path, ce_loss_no_grad, _, all_completed_paths = self._process_single_token_dfs(
                    current_input, target_token, current_seq_len
                )
                
                # [调试输出] Token处理结果
                print(f"    Token {token_idx}: loss={ce_loss_no_grad:.4f}, path_len={len(best_path)}, "
                      f"explored={len(all_completed_paths)} paths, path={best_path}")
                
                # 步骤2: 收集所有探索路径的三元组(不只是最优路径)
                for path_info in all_completed_paths:
                    if path_info.trajectories and path_info.ce_loss is not None:
                        triples = compute_path_values(path_info.trajectories, path_info.ce_loss)
                        self.traj_buffer.extend(triples)
                
                # 步骤3: 带梯度沿最优路径前向传播
                ce_loss_with_grad = self._forward_along_path_with_grad(
                    current_input, best_path, target_token, current_seq_len
                )
                
                # 步骤4: 立即反向传播更新LoRA
                ce_loss_with_grad.backward()
                torch.nn.utils.clip_grad_norm_(self.layered_loader.get_trainable_params(), 
                                               self.lora_config.grad_clip)
                self.lora_optimizer.step()
                self.lora_optimizer.zero_grad()
                
                # 累积统计信息
                total_ce_loss += ce_loss_no_grad  # 使用无梯度的loss作为统计
                
                # [关键] 每处理完一个token,检查是否需要更新RL网络
                self.token_count_since_last_rl_update += 1
                should_update_rl = self.token_count_since_last_rl_update >= 100  # 每100个token
                
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
                continue
        
        return {
            'total_loss': total_ce_loss,
            'token_count': token_count
        }
    
    def _update_rl_network(self):
        """更新RL网络(Actor-Critic)
        
        从traj_buffer中分批训练Actor-Critic
        触发条件: 每100个token或10000个三元组
        """
        if not self.traj_buffer:
            return
        
        print(f"\n开始RL网络更新...")
        print(f"轨迹缓冲区大小: {len(self.traj_buffer)}")
        
        # 分批训练
        batch_size = self.model_config.batch_size * 8
        random.shuffle(self.traj_buffer)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        batch_count = 0
        
        for batch_start in range(0, len(self.traj_buffer), batch_size):
            batch = self.traj_buffer[batch_start:batch_start + batch_size]
            
            try:
                actor_loss, critic_loss, batch_loss = train_actor_critic_batch(
                    actor_critic=self.actor_critic,
                    rl_optimizer=self.rl_optimizer,
                    batch=batch,
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
                
                # [关键修复] 每个batch后清理显存，防止泄漏
                del batch, actor_loss, critic_loss, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                import traceback
                print(f"RL 批次训练失败: {e}")
                print(f"详细错误信息:\n{traceback.format_exc()}")
                continue
        
        if batch_count == 0:
            print("警告: 没有有效的batch,跳过RL更新")
            return
        
        avg_actor_loss = total_actor_loss / batch_count
        avg_critic_loss = total_critic_loss / batch_count
        avg_total_loss = total_loss / batch_count
        print(f"Actor-Critic 训练完成：actor loss={avg_actor_loss:.4f}, critic loss={avg_critic_loss:.4f}, 总 loss={avg_total_loss:.4f}")
        
        # 清空缓冲区
        self.traj_buffer.clear()
        torch.cuda.empty_cache()
        
        print("RL网络更新完成\n")
    
    def _should_branch(self, action_probs: torch.Tensor) -> Tuple[bool, List[int]]:
        """判断是否应该产生分支"""
        # 获取 top-2 概率和索引
        top2_probs, top2_indices = torch.topk(action_probs, 2)
        
        # 分支条件：第二大概率 > 0.2 且 > 最大概率 × 0.8
        if top2_probs[1] > self.rl_config.branch_prob_threshold and \
           top2_probs[1] > top2_probs[0] * self.rl_config.branch_prob_ratio:
            return True, top2_indices.tolist()
        
        return False, []
    
    def train_epoch(self, start_idx: int = 0, max_samples: int = None, 
                    cleanup_after: bool = True) -> Dict:
        """训练单个epoch,支持断点续训
        
        Args:
            start_idx: 起始样本索引(用于断点续训)
            max_samples: 最大处理样本数,None表示处理完整个数据集
            cleanup_after: 训练完成后是否清理轨迹数据
        """
        print(f"\n{'='*60}")
        print(f"Stage2 训练流程")
        print(f"{'='*60}")
        
        # 1. 加载数据集
        dataset = self.data_manager.load_jsonl_dataset(self.jsonl_path)
        total_samples = len(dataset)
        print(f"数据集大小: {total_samples} 条样本")
        
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
                
                # Tokenize
                tokenizer = self.layered_loader.tokenizer
                input_ids, output_ids, seq_len = self.data_manager.tokenize_input_output_pair(
                    input_text, output_text, tokenizer, self.device
                )
                
                # 处理单个pair
                stats = self._process_single_pair(input_ids, output_ids)
                
                total_ce_loss += stats['total_loss']
                total_tokens += stats['token_count']
                sample_count += 1
                self.token_count_since_last_rl_update += stats['token_count']
                
                # 清理显存
                del input_ids, output_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # [调试输出] 样本完成统计
                if sample_count % 10 == 0 or idx == end_idx - 1:
                    avg_loss = total_ce_loss / max(1, total_tokens)
                    print(f"\u2713 样本 {sample_count}/{end_idx-start_idx}: "
                          f"avg_loss={avg_loss:.4f}, tokens={total_tokens}, "
                          f"traj_buffer={len(self.traj_buffer)}")
                
                # 每5个样本保存checkpoint并更新模型
                if sample_count % 5 == 0:
                    avg_loss = total_ce_loss / max(1, total_tokens)
                    print(f"\n已处理 {sample_count} 个样本, 平均CE={avg_loss:.4f}")
                    
                    # 保存checkpoint
                    self.data_manager.save_checkpoint(idx + 1, self.checkpoint_file, avg_loss)
                    
                    # 如果还有剩余的三元组,更新RL网络
                    if len(self.traj_buffer) > 0:
                        print(f"\n  [Checkpoint] 保存前更新剩余 {len(self.traj_buffer)} 个三元组")
                        self._update_rl_network()
                        self.token_count_since_last_rl_update = 0
                    
                    # 保存模型
                    self.model_manager.save_lora_model(self.layered_loader, avg_loss, stage_group='stage2')
                    self.model_manager.save_rl_model(self.actor_critic, None, 0, avg_loss, stage_group='stage2')
                    
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
        self.model_manager.save_lora_model(self.layered_loader, avg_ce_loss, stage_group='stage2')
        
        # 5. 清理
        if cleanup_after:
            self.data_manager.cleanup_training_data(remove_trajectories=True, remove_path_records=False)
        
        print(f"\nStage2 Epoch 完成！")
        print(f"  - 平均 CE Loss: {avg_ce_loss:.4f}")
        print(f"  - 样本数量: {sample_count}")
        print(f"  - Token总数: {total_tokens}\n")
        
        return {
            'avg_ce_loss': avg_ce_loss,
            'sample_count': sample_count,
            'total_tokens': total_tokens,
            'trajectory_count': len(self.traj_buffer)
        }
