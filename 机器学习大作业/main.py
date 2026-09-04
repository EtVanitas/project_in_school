"""
RNN策略引导的动态路径微调 - 主入口

使用方法:
    python main.py              # 显示配置信息
    python main.py --train      # 开始训练
"""

import sys
import argparse
import torch
from config import ModelConfig, LoRAConfig, RLConfig, path_config


def show_config():
    """显示配置信息"""
    print("="*70)
    print("RNN策略引导的动态路径微调 (Actor-Critic)")
    print("="*70)
    
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    rl_config = RLConfig()
    
    print(f"\n模型配置:")
    print(f"  路径：{path_config.qwen_model_path}")
    print(f"  层数：{model_config.num_layers} (前{model_config.fixed_layers}固定 + 后{model_config.dynamic_layers}动态)")
    print(f"  隐藏层：{model_config.hidden_size}")
    print(f"  词表：{model_config.vocab_size:,}")
    
    print(f"\nLoRA 配置:")
    print(f"  秩：{lora_config.r}")
    print(f"  Alpha: {lora_config.alpha}")
    print(f"  Target modules: {lora_config.target_modules}")
    
    print(f"\nRL 配置:")
    print(f"  状态维度: token_probs(399→64投影) + RNN(64) + layer(16,覆盖8-23) + step(1) + context(1) = 82")
    print(f"  架构: LSTM (64->64, 1层单向)")
    print(f"  动作空间: {rl_config.action_dim} (15个跳转目标9-23 + 1个输出)")
    print(f"  Actor头: 82 -> {rl_config.actor_hidden_dim} -> {rl_config.action_dim}")
    print(f"  Critic头: 82 -> {rl_config.critic_hidden_dim} -> 1")
    print(f"  学习率：{rl_config.lr}")
    print(f"  奖励计算：交叉熵损失反向推导，步数惩罚 -{rl_config.step_penalty}/步")
    
    print(f"\n训练配置:")
    print(f"  设备：{'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Epochs: 10 (默认)")
    print(f"  Batch size: 1 (逐样本训练)")
    
    print("\n" + "="*70)
    print("使用方法:")
    print("  python main.py          # 显示配置信息")
    print("  python main.py --train  # 开始训练")
    print("="*70)


def train():
    """训练"""
    print("="*70)
    print("训练模式")
    print("="*70)
    
    try:
        from trainer import Trainer
        
        trainer = Trainer()
        trainer.train_epoch()
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 训练失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="RNN策略引导的动态路径微调")
    parser.add_argument('--train', action='store_true', help='开始训练')
    
    args = parser.parse_args()
    
    if args.train:
        success = train()
        sys.exit(0 if success else 1)
    else:
        show_config()


if __name__ == "__main__":
    main()
