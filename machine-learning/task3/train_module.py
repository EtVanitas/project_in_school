"""模型训练模块"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config


def create_model(num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED):
    """
    创建EfficientNet-B0模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        
    Returns:
        model: EfficientNet-B0模型
    """
    print(f"创建模型: {config.MODEL_NAME}, 预训练: {pretrained}")
    
    # 加载预训练模型
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # 修改分类器头部为12类
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}, 可训练: {trainable_params:,}\n")
    
    return model


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失，用于防止过拟合"""
    
    def __init__(self, smoothing=config.LABEL_SMOOTHING, class_weights=None):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights
        
    def forward(self, pred, target):
        """
        Args:
            pred: 模型预测 (batch_size, num_classes)
            target: 真实标签 (batch_size,)
        """
        confidence = 1.0 - self.smoothing
        log_probs = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        
        # 如果提供了类别权重，应用权重
        if self.class_weights is not None:
            weights = self.class_weights[target]
            loss = loss * weights
        
        return loss.mean()


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]')
    
    for images, labels in progress_bar:
        # 移动到设备
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计指标
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / total, 100. * correct / total


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    验证模型
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    progress_bar = tqdm(val_loader, desc='Validating')
    
    for images, labels in progress_bar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / total, 100. * correct / total


def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史记录字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 训练曲线已保存到: {save_path}")
    
    plt.show()


def train_model():
    """
    主训练函数
    
    Returns:
        model: 训练好的模型
        history: 训练历史记录
        best_val_acc: 最佳验证准确率
    """
    print("="*60)
    print("开始训练 EfficientNet-B0")
    print("="*60 + "\n")
    
    # 设置设备
    device = torch.device(config.DEVICE)
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("警告: 未检测到GPU，将使用CPU训练\n")
    
    # 加载数据
    try:
        from data_module import create_data_loaders
        train_loader, val_loader, test_loader, class_to_idx, test_paths = create_data_loaders()
    except Exception as e:
        print(f"错误: 数据加载失败 - {e}")
        raise
    
    # 创建模型
    model = create_model().to(device)
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU\n")
        model = nn.DataParallel(model)
    
    # 创建优化器和调度器
    optimizer = optim.SGD(
        model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY, nesterov=True
    )
    
    # Cosine Annealing 学习率调度器（仅在后20个epoch使用）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.LR_SCHEDULER['T_max'],
        eta_min=config.LR_SCHEDULER['eta_min']
    )
    
    print(f"学习率调度: 固定LR (前{config.EPOCHS - config.LR_SCHEDULER['T_max']} epochs) + Cosine (后{config.LR_SCHEDULER['T_max']} epochs)")
    print(f"  - 初始学习率: {config.LEARNING_RATE}")
    print(f"  - 最小学习率: {config.LR_SCHEDULER['eta_min']}")
    print(f"  - Cosine退火开始: epoch {config.EPOCHS - config.LR_SCHEDULER['T_max'] + 1}")
    print(f"  - Cosine周期: {config.LR_SCHEDULER['T_max']} epochs\n")
    
    # 创建带类别权重的损失函数
    if hasattr(config, 'CLASS_WEIGHTS'):
        class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.LABEL_SMOOTHING,
            class_weights=class_weights
        )
        print(f"使用类别权重: {config.CLASS_WEIGHTS}")
        print(f"Black-grass权重: {config.CLASS_WEIGHTS[0]}, Loose Silky-bent权重: {config.CLASS_WEIGHTS[6]}\n")
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.LABEL_SMOOTHING)
    
    # 训练历史记录
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc, patience_counter = 0.0, 0
    
    start_time = time.time()
    
    try:
        for epoch in range(config.EPOCHS):
            epoch_start = time.time()
            
            # 设置学习率
            cosine_start_epoch = config.EPOCHS - config.LR_SCHEDULER['T_max']
            if epoch < cosine_start_epoch:
                # 前30个epoch：固定学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.LEARNING_RATE
            else:
                # 后20个epoch：Cosine退火
                scheduler.step()
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 训练和验证
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            print(f'\nEpoch [{epoch+1}/{config.EPOCHS}] Time: {epoch_time:.1f}s | '
                  f'LR: {current_lr:.6f} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                try:
                    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'class_to_idx': class_to_idx,
                    }, config.MODEL_SAVE_PATH)
                    print(f"✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)\n")
                except Exception as e:
                    print(f"警告: 模型保存失败 - {e}\n")
            else:
                patience_counter += 1
                print(f"早停计数器: {patience_counter}/{config.EARLY_STOP_PATIENCE}\n")
            
            # 早停检查
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\n早停触发!在epoch {epoch+1}停止训练")
                break
    
    except KeyboardInterrupt:
        print("\n\n训练被用户中断!")
        print(f"当前最佳验证准确率: {best_val_acc:.2f}%")
    except Exception as e:
        print(f"\n\n训练过程中发生错误: {e}")
        raise
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print("="*60 + "\n")
    
    # 绘制训练曲线
    try:
        plot_path = config.PLOTS_DIR / "training_history.png"
        plot_training_history(history, save_path=plot_path)
    except Exception as e:
        print(f"警告: 训练曲线绘制失败 - {e}")
    
    return model, history, best_val_acc
