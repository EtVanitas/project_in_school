"""模型管理器 - LoRA 模型和 RL 网络的保存和加载（最优 + 最新）

目录结构：
- trainer/models/
  - lora_best.pt / lora_latest.pt
  - rl_best.pt / rl_latest.pt
"""

import os
import json
import torch
from datetime import datetime
from config import PathConfig


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        """初始化"""
        self.path_config = PathConfig()
        self.save_dir = self.path_config.models_dir
        os.makedirs(self.save_dir, exist_ok=True)
       
    def save_lora_model(self, layered_loader, loss: float):
        """保存 LoRA 模型（同时更新最优和最新）"""
        # 提取 LoRA 参数
        lora_state_dict = {k: v for k, v in layered_loader.state_dict().items() 
                          if 'lora_' in k}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config = {'loss': round(loss, 4), 'timestamp': timestamp}
        
        # 保存最新模型
        latest_path = os.path.join(self.save_dir, 'lora_latest.pt')
        torch.save(lora_state_dict, latest_path)
        with open(os.path.join(self.save_dir, 'lora_latest.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 判断是否是最优模型
        is_best = False
        best_json_path = os.path.join(self.save_dir, 'lora_best.json')
        if os.path.exists(best_json_path):
            try:
                with open(best_json_path, 'r', encoding='utf-8') as f:
                    best_config = json.load(f)
                if 'loss' in best_config and loss < best_config['loss']:
                    is_best = True
            except Exception:
                is_best = True
        else:
            is_best = True
        
        # 如果是最优，也保存最优模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'lora_best.pt')
            torch.save(lora_state_dict, best_path)
            
            best_config = config.copy()
            best_config['is_best'] = True
            with open(os.path.join(self.save_dir, 'lora_best.json'), 'w', encoding='utf-8') as f:
                json.dump(best_config, f, indent=2, ensure_ascii=False)
            print(f"LoRA 模型已保存 (loss={loss:.4f}, 新最优模型)")
        else:
            print(f"LoRA 模型已保存 (loss={loss:.4f})")
    
    def load_lora_model(self, layered_loader, use_best: bool = True) -> bool:
        """加载 LoRA 模型"""
        model_name = 'lora_best.pt' if use_best else 'lora_latest.pt'
        model_path = os.path.join(self.save_dir, model_name)
        
        if not os.path.exists(model_path):
            return False
        
        print(f"加载 LoRA 模型：{model_path}")
        
        device = next(layered_loader.parameters()).device
        state_dict = torch.load(model_path, map_location=device)
        missing_keys, unexpected_keys = layered_loader.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  缺失的键：{len(missing_keys)} 个")
        if unexpected_keys:
            print(f"  意外的键：{len(unexpected_keys)} 个")
        
        print(f"✓ 已加载 LoRA 模型")
        
        # 加载配置信息
        config_path = model_path.replace('.pt', '.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_info = json.load(f)
            if 'loss' in config_info:
                print(f"  - Loss: {config_info['loss']:.4f}")
        
        return True
    
    def save_rl_model(self, actor_critic, optimizer, epoch: int, loss: float):
        """保存 RL 网络（同时更新最优和最新）"""
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'actor_critic_state_dict': actor_critic.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint['timestamp'] = timestamp
        
        # 保存最新模型
        latest_path = os.path.join(self.save_dir, 'rl_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # 判断是否是最优模型
        is_best = False
        best_path = os.path.join(self.save_dir, 'rl_best.pt')
        if os.path.exists(best_path):
            try:
                best_checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
                if 'loss' in best_checkpoint and loss < best_checkpoint['loss']:
                    is_best = True
            except Exception:
                is_best = True
        else:
            is_best = True
        
        # 如果是最优，也保存最优模型
        if is_best:
            torch.save(checkpoint, best_path)
            print(f"RL 模型已保存 (loss={loss:.4f}, 新最优模型)")
        else:
            print(f"RL 模型已保存 (loss={loss:.4f})")
    
    def load_rl_model(self, actor_critic, optimizer=None, use_best: bool = True) -> bool:
        """加载 RL 网络"""
        model_name = 'rl_best.pt' if use_best else 'rl_latest.pt'
        model_path = os.path.join(self.save_dir, model_name)
        
        if not os.path.exists(model_path):
            return False
        
        print(f"加载 RL 模型：{model_path}")
        device = next(actor_critic.parameters()).device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ 已加载 RL 模型")
        if 'loss' in checkpoint:
            print(f"  - Loss: {checkpoint['loss']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  - Epoch: {checkpoint['epoch']}")
        
        return True
