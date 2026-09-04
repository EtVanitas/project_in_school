"""
Qwen3.5-2B 动态路径微调 - 主入口

四阶段训练架构:
- 阶段 0: 文本分块 + 预计算基准（一次性）
- 阶段 1A: 批量路径搜索 + 轨迹收集（批处理模式）
- 阶段 1B: LoRA 微调 + RL 网络训练（策略 + 价值）
- 阶段 2: 策略引导的在线联合训练（DFS + 动态分支）

使用方法:
    python main.py --stage0       # 运行阶段 0（文本分块+基准计算）
    python main.py --stage1a      # 运行阶段 1A（路径搜索）
    python main.py --stage1b      # 运行阶段 1B（LoRA+RL训练）
    python main.py --stage2       # 运行阶段 2（在线联合训练）
    python main.py --all          # 运行完整流程
    python main.py                # 显示配置信息
"""

import sys
import argparse
import torch
import os
from config import ModelConfig, LoRAConfig, RLConfig
from trainer.stage0 import Stage0Trainer
from trainer.stage1a import Stage1ATrainer
from trainer.stage1b import Stage1BTrainer
from trainer.stage2 import Stage2Trainer


def show_config():
    """显示配置信息"""
    print("="*70)
    print("Qwen3.5-2B 动态路径微调架构")
    print("="*70)
    
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    rl_config = RLConfig()
    
    print(f"\n模型配置:")
    print(f"  路径：{model_config.model_path}")
    print(f"  层数：{model_config.num_layers} (前{model_config.fixed_layers}固定 + 后{model_config.dynamic_layers}动态)")
    print(f"  隐藏层：{model_config.hidden_size}")
    print(f"  词表：{model_config.vocab_size:,}")
    print(f"  批次大小：{model_config.batch_size}")
    
    print(f"\nLoRA 配置:")
    print(f"  秩：{lora_config.r}")
    print(f"  Alpha: {lora_config.alpha}")
    
    print(f"\nRL 配置:")
    print(f"  状态维度：vocab_probs({model_config.vocab_size}) -> 256 -> 64 + 17 = 81")
    print(f"  动作空间：{rl_config.action_dim}")
    print(f"  架构：极简版 (无Fusion/MLP,直接拼接)")
    print(f"  Actor 头：81 -> {rl_config.actor_hidden_dim} -> {rl_config.action_dim}")
    print(f"  Critic 头：81 -> {rl_config.critic_hidden_dim} -> 1")
    print(f"  学习率：{rl_config.lr}")
    print(f"  奖励计算：交叉熵损失反向推导，步数惩罚 -{rl_config.step_penalty}/步")
    
    print("\n" + "="*70)
    print("使用方法:")
    print("  python main.py --stage0     # 阶段0：文本分块+基准计算")
    print("  python main.py --stage1a    # 阶段1A：批量路径搜索")
    print("  python main.py --stage1b    # 阶段1B：LoRA+RL训练")
    print("  python main.py --stage2     # 阶段2：在线联合训练")
    print("  python main.py --all        # 完整流程")
    print("="*70)


def test_model_loading():
    """测试模型加载"""
    print("="*70)
    print("测试模式：模型加载")
    print("="*70)
    
    try:
        print("\n加载模型...")
        trainer = Stage1ATrainer()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n设备：{device}")
        
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        print("\n[OK] 模型加载测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def train_stage0():
    """阶段 0：预计算基准（增量式）"""
    print("="*70)
    print("训练模式：阶段 0 - 预计算基准（增量式）")
    print("说明：扫描 texts/len_X/sample_Y.pt，补充缺失的 bases/")
    print("="*70)
    
    # 初始化训练器
    print("\n初始化 Stage0Trainer...")
    trainer = Stage0Trainer()
    
    # 计算并保存基准（自动增量处理）
    print("\n开始增量式计算基准...")
    trainer.compute_and_save_bases(max_length=2048)
    
    print("\n✓ 阶段 0 完成！")


def train_stage1a():
    """阶段 1A: 批量路径搜索 + 轨迹收集"""
    print("="*70)
    print("训练模式：阶段 1A - 批量路径搜索 + 轨迹收集")
    print("说明：批处理模式，每个样本独立找最优path，收集所有路径的三元组")
    print("="*70)
    
    # 初始化训练器
    print("\n初始化 Stage1ATrainer...")
    trainer = Stage1ATrainer()
    
    # 开始训练
    print("\n" + "="*70)
    print("开始路径搜索")
    print("="*70)
    
    try:
        stats = trainer.train_epoch()
        
        print(f"\n训练完成:")
        print(f"  - 平均 Loss：{stats['avg_loss']:.4f}")
        print(f"  - 样本数量：{stats['sample_count']}")
        print(f"  - 批次数：{stats['batch_count']}")
        
        print("\n✓ 阶段 1A 完成！")
        print("  - path_records.json: 记录每个path对应的样本")
        print("  - trajectories/: 所有路径的三元组\n")
        
    except KeyboardInterrupt:
        print("\n\n训练被中断")
    except Exception as e:
        print(f"\n✗ 训练出错：{e}")
        import traceback
        traceback.print_exc()


def train_stage1b():
    """阶段 1B: LoRA 微调 + RL 网络训练"""
    print("="*70)
    print("训练模式：阶段 1B - LoRA 微调 + RL 网络训练")
    print("说明：读取 path_records.json 训练 LoRA，读取 trajectories 训练 Actor-Critic")
    print("="*70)
    
    # 初始化训练器
    print("\n初始化 Stage1BTrainer...")
    trainer = Stage1BTrainer()
    
    # 开始训练
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)
    
    try:
        stats = trainer.train_epoch(cleanup_after=True)
        
        print(f"\n训练完成:")
        print(f"  - LoRA: {stats['lora']}")
        print(f"  - RL: {stats['rl']}")
        
        print("\n✓ 阶段 1B 完成！")
        print("  - 已保存最优和最新的 LoRA 模型")
        print("  - 已保存最优和最新的 RL 模型")
        print("  - 已清理 path_records.json 和 trajectories/\n")
        
    except KeyboardInterrupt:
        print("\n\n训练被中断")
    except Exception as e:
        print(f"\n✗ 训练出错：{e}")
        import traceback
        traceback.print_exc()


def train_stage2():
    """阶段 2: 策略引导的动态路径微调（JSONL版本）"""
    print("="*70)
    print("训练模式：阶段 2 - JSONL Teacher Forcing训练")
    print("说明：使用 Superior-Reasoning-SFT 数据集进行逐个token预测训练")
    print("="*70)
    
    # 初始化训练器
    print("\n初始化 Stage2Trainer...")
    trainer = Stage2Trainer()
    
    # 检查是否存在checkpoint
    checkpoint_file = trainer.checkpoint_file
    start_idx = 0
    if os.path.exists(checkpoint_file):
        start_idx = trainer.data_manager.load_checkpoint(checkpoint_file)
        print(f"从checkpoint恢复: 已处理 {start_idx} 个样本")
    
    # 开始训练
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)
    
    try:
        stats = trainer.train_epoch(start_idx=start_idx, max_samples=None)
        
        if stats.get('skipped'):
            print(f"\n训练跳过: {stats.get('reason', '未知原因')}")
            return
        
        print(f"\n训练完成:")
        print(f"  - 平均 CE Loss: {stats['avg_ce_loss']:.4f}")
        print(f"  - 样本数量: {stats['sample_count']}")
        print(f"  - Token总数: {stats['total_tokens']}")
        
        print("\n阶段 2 完成！")
        print("  - 已保存 Stage2 优化的 LoRA 模型")
        print("  - 已更新 Actor-Critic 策略网络")
        print("  - Checkpoint已保存,可断点续训\n")
        
    except KeyboardInterrupt:
        print("\n\n训练被中断")
    except Exception as e:
        print(f"\n训练出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5-2B 动态路径微调")
    parser.add_argument('--stage0', action='store_true', help='运行阶段 0（预计算基准）')
    parser.add_argument('--stage1a', action='store_true', help='运行阶段 1A（路径搜索+轨迹收集）')
    parser.add_argument('--stage1b', action='store_true', help='运行阶段 1B（LoRA微调+RL训练）')
    parser.add_argument('--stage2', action='store_true', help='运行阶段 2（在线联合训练）')
    parser.add_argument('--all', action='store_true', help='运行完整流程')
    parser.add_argument('--test', action='store_true', help='测试模型加载')
    
    args = parser.parse_args()
    
    if args.stage0:
        train_stage0()
    elif args.stage1a:
        train_stage1a()
    elif args.stage1b:
        train_stage1b()
    elif args.stage2:
        train_stage2()
    elif args.all:
        print(">>> 运行完整流程\n")
        train_stage0()
        print("\n" + "="*70 + "\n")
        train_stage1a()
        print("\n" + "="*70 + "\n")
        train_stage1b()
        print("\n" + "="*70 + "\n")
        train_stage2()
    elif args.test:
        success = test_model_loading()
        sys.exit(0 if success else 1)
    else:
        show_config()
