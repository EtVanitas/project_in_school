"""模型管理器 - 管理 LoRA 和 RL 网络的保存、加载和目录管理

职责：
- LoRA 模型的保存和加载（最优 + 最新）
- RL 网络（Actor-Critic）的保存和加载（最优 + 最新）
- 智能加载策略（带回退机制）

目录结构：
- models/training_models/stage1/
  - lora_best.pt / lora_latest.pt
  - rl_best.pt / rl_latest.pt
- models/training_models/stage2/
  - lora_best.pt / lora_latest.pt
  - rl_best.pt / rl_latest.pt
"""

import os
import json
import torch
from datetime import datetime
from config import PathConfig


class ModelManager:
    """模型管理器 - 统一管理 LoRA 和 RL 模型的存取
    
    目录结构：
    - stage1/: LoRA (stage1a) + RL (stage1b)
    - stage2/: LoRA (stage2a) + RL (stage2b)
    """
    
    def __init__(self):
        """初始化模型管理器（直接使用全局PathConfig）"""
        # 使用全局 PathConfig
        self.path_config = PathConfig()
        
        # 确保模型目录存在
        self._ensure_model_dirs()
    
    def _ensure_model_dirs(self):
        """确保模型目录存在"""
        os.makedirs(self.path_config.stage1_model_dir, exist_ok=True)
        os.makedirs(self.path_config.stage2_model_dir, exist_ok=True)
    
    def _get_model_dir(self, stage_group: str) -> str:
        """获取模型目录
        
        Args:
            stage_group: 阶段组 ('stage1' or 'stage2')
            
        Returns:
            模型目录路径
        """
        if stage_group == 'stage1':
            return self.path_config.stage1_model_dir
        elif stage_group == 'stage2':
            return self.path_config.stage2_model_dir
        else:
            raise ValueError(f"Unknown stage group: {stage_group}")
    
    # ==================== LoRA 模型管理 ====================
    
    def save_lora_model(self, layered_loader, loss: float, stage_group: str = 'stage1'):
        """保存 LoRA 模型（同时更新最优和最新）
        
        Args:
            layered_loader: LayeredModelLoader 实例
            loss: 当前 loss
            stage_group: 阶段组 ('stage1' or 'stage2')
        """
        output_dir = self._get_model_dir(stage_group)
        
        # 提取 LoRA 参数
        lora_state_dict = {k: v for k, v in layered_loader.state_dict().items() 
                          if 'lora_' in k}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config = {'loss': round(loss, 4), 'timestamp': timestamp}
        
        # 始终保存最新模型
        latest_path = os.path.join(output_dir, 'lora_latest.pt')
        torch.save(lora_state_dict, latest_path)
        with open(os.path.join(output_dir, 'lora_latest.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 判断是否是最优模型（读取现有最优模型的 loss）
        is_best = False
        best_json_path = os.path.join(output_dir, 'lora_best.json')
        if os.path.exists(best_json_path):
            try:
                with open(best_json_path, 'r', encoding='utf-8') as f:
                    best_config = json.load(f)
                if 'loss' in best_config and loss < best_config['loss']:
                    is_best = True
            except Exception:
                is_best = True  # 如果读取失败，保守地认为是最优
        else:
            is_best = True  # 如果没有最优模型，当前就是最优
        
        # 如果是最优，也保存最优模型
        if is_best:
            best_path = os.path.join(output_dir, 'lora_best.pt')
            torch.save(lora_state_dict, best_path)
            
            best_config = config.copy()
            best_config['is_best'] = True
            with open(os.path.join(output_dir, 'lora_best.json'), 'w', encoding='utf-8') as f:
                json.dump(best_config, f, indent=2, ensure_ascii=False)
            print(f"LoRA 模型已保存：{output_dir} (loss={loss:.4f}, 新最优模型)")
        else:
            print(f"LoRA 模型已保存：{output_dir} (loss={loss:.4f})")
    
    def load_lora_model(self, layered_loader, stage_group: str = 'stage1',
                       use_best: bool = True):
        """加载 LoRA 模型
        
        Args:
            layered_loader: LayeredModelLoader 实例
            stage_group: 阶段组 ('stage1' or 'stage2')
            use_best: 是否使用最优模型（否则使用最新）
        """
        output_dir = self._get_model_dir(stage_group)
        model_name = 'lora_best.pt' if use_best else 'lora_latest.pt'
        model_path = os.path.join(output_dir, model_name)
        
        if not os.path.exists(model_path):
            print(f"未找到 {stage_group} LoRA 模型：{model_path}")
            return False
        
        print(f"加载 {stage_group} LoRA 模型：{model_path}")
        
        # 获取设备（从 layered_loader 的参数中推断）
        device = next(layered_loader.parameters()).device
        state_dict = torch.load(model_path, map_location=device)
        missing_keys, unexpected_keys = layered_loader.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  缺失的键：{len(missing_keys)} 个")
        if unexpected_keys:
            print(f"  意外的键：{len(unexpected_keys)} 个")
        
        print(f"✓ 已加载 {stage_group} LoRA 模型")
        
        # 加载配置信息
        config_path = model_path.replace('.pt', '.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_info = json.load(f)
            if 'loss' in config_info:
                print(f"  - Loss: {config_info['loss']:.4f}")
        
        return True
    
    def try_load_latest_lora_model(self, layered_loader, current_stage: str = 'stage1a'):
        """智能加载最新 LoRA 模型（带回退机制）
        
        Stage1A: 优先加载 stage1 的最新，失败则使用初始权重
        Stage1B: 优先加载 stage1 的最新，失败则从头训练
        Stage2: 优先加载 stage2 的最新，失败则回退到 stage1 的最新
        
        Args:
            layered_loader: LayeredModelLoader 实例
            current_stage: 当前阶段 ('stage1a', 'stage1b', 'stage2a', 'stage2b')
        """
        # 确定阶段组
        if current_stage in ['stage1a', 'stage1b']:
            stage_group = 'stage1'
        elif current_stage in ['stage2a', 'stage2b']:
            stage_group = 'stage2'
        else:
            raise ValueError(f"Unknown stage: {current_stage}")
        
        # 优先加载当前阶段组的最新模型
        if self.load_lora_model(layered_loader, stage_group=stage_group, use_best=False):
            return
        
        # 回退到 stage1（仅当当前是 stage2 时）
        if stage_group == 'stage2':
            if self.load_lora_model(layered_loader, stage_group='stage1', use_best=False):
                print("未找到 stage2 LoRA 模型，使用 stage1 最新模型")
                return
        
        print("未找到任何 LoRA 模型，使用初始权重")
    
    # ==================== RL 模型管理 ====================
    
    def save_rl_model(self, actor_critic, optimizer, epoch: int, loss: float,
                     stage_group: str = 'stage1'):
        """保存 RL 网络（同时更新最优和最新）
        
        Args:
            actor_critic: Actor-Critic 网络
            optimizer: 优化器（可选）
            epoch: 当前 epoch
            loss: 当前 loss
            stage_group: 阶段组 ('stage1' or 'stage2')
        """
        output_dir = self._get_model_dir(stage_group)
        
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'actor_critic_state_dict': actor_critic.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint['timestamp'] = timestamp
        
        # 始终保存最新模型
        latest_path = os.path.join(output_dir, 'rl_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # 判断是否是最优模型（读取现有最优模型的 loss）
        is_best = False
        best_path = os.path.join(output_dir, 'rl_best.pt')
        if os.path.exists(best_path):
            try:
                best_checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
                if 'loss' in best_checkpoint and loss < best_checkpoint['loss']:
                    is_best = True
            except Exception:
                is_best = True  # 如果读取失败，保守地认为是最优
        else:
            is_best = True  # 如果没有最优模型，当前就是最优
        
        # 如果是最优，也保存最优模型
        if is_best:
            torch.save(checkpoint, best_path)
            print(f"RL 模型已保存：{output_dir} (loss={loss:.4f}, 新最优模型)")
        else:
            print(f"RL 模型已保存：{output_dir} (loss={loss:.4f})")
    
    def load_rl_model(self, actor_critic, optimizer=None, stage_group: str = 'stage1',
                     use_best: bool = True):
        """加载 RL 网络
        
        Args:
            actor_critic: Actor-Critic 网络
            optimizer: 优化器（可选）
            stage_group: 阶段组 ('stage1' or 'stage2')
            use_best: 是否使用最优模型
            
        Returns:
            是否成功加载
        """
        output_dir = self._get_model_dir(stage_group)
        model_name = 'rl_best.pt' if use_best else 'rl_latest.pt'
        model_path = os.path.join(output_dir, model_name)
        
        if not os.path.exists(model_path):
            print(f"未找到 {stage_group} RL 模型：{model_path}")
            return False
        
        print(f"加载 {stage_group} RL 模型：{model_path}")
        # 获取模型所在设备
        device = next(actor_critic.parameters()).device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ 已加载 {stage_group} RL 模型")
        if 'loss' in checkpoint:
            print(f"  - Loss: {checkpoint['loss']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  - Epoch: {checkpoint['epoch']}")
        
        return True
    
    def try_load_latest_rl_model(self, actor_critic, optimizer=None, current_stage: str = 'stage1b'):
        """智能加载最新 RL 模型（带回退机制）
        
        Stage1B: 优先加载 stage1 的最新，失败则从头训练
        Stage2: 优先加载 stage2 的最新，失败则回退到 stage1 的最新
        
        Args:
            actor_critic: Actor-Critic 网络
            optimizer: 优化器（可选）
            current_stage: 当前阶段 ('stage1b', 'stage2b')
        """
        # 确定阶段组
        if current_stage == 'stage1b':
            stage_group = 'stage1'
        elif current_stage == 'stage2b':
            stage_group = 'stage2'
        else:
            raise ValueError(f"Unknown stage: {current_stage}")
        
        # 优先加载当前阶段组的最新模型
        if self.load_rl_model(actor_critic, optimizer, stage_group=stage_group, use_best=False):
            return
        
        # 回退到 stage1（仅当当前是 stage2 时）
        if stage_group == 'stage2':
            if self.load_rl_model(actor_critic, optimizer, stage_group='stage1', use_best=False):
                print("未找到 stage2 RL 模型，使用 stage1 最新模型")
                return
        
        print("未找到任何 RL 模型，使用随机初始化")
