"""EfficientNet-B0 植物幼苗分类 - 主入口"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config


def main():
    """主函数 - 整合训练和评估流程"""
    parser = argparse.ArgumentParser(description='EfficientNet-B0 植物幼苗分类')
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'eval', 'all'],
                       help='运行模式: train(训练), eval(评估), all(训练+评估)')
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("="*60)
    print("EfficientNet-B0 植物幼苗分类系统")
    print("="*60)
    print(f"运行模式: {args.mode}")
    print(f"设备: {config.DEVICE}")
    print(f"模型: {config.MODEL_NAME}")
    print(f"图像尺寸: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"训练轮数: {config.EPOCHS}")
    print("="*60 + "\n")
    
    # 训练阶段
    if args.mode in ['train', 'all']:
        print("【阶段1】开始训练...\n")
        from train_module import train_model
        model, history, best_acc = train_model()
        print(f"\n✓ 训练完成!最佳验证准确率: {best_acc:.2f}%\n")
        
        if args.mode == 'train':
            return
    
    # 评估阶段
    if args.mode in ['eval', 'all']:
        print("【阶段2】开始评估...\n")
        from eval_module import run_evaluation
        submission = run_evaluation()
        print(f"\n✓ 评估完成!提交文件已保存\n")


if __name__ == "__main__":
    main()
