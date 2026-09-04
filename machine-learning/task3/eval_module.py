"""模型评估模块"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config


def load_best_model(device):
    """
    加载最佳模型
    
    Args:
        device: 设备
        
    Returns:
        model: 加载的模型
        checkpoint: 检查点信息
    """
    if not config.MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在: {config.MODEL_SAVE_PATH}\n请先运行训练: python main.py --mode train")
    
    print(f"加载模型: {config.MODEL_SAVE_PATH}")
    try:
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")
    
    # 创建并加载模型
    from train_module import create_model
    model = create_model()
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        raise RuntimeError(f"模型权重加载失败: {e}")
    
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功! Epoch: {checkpoint['epoch']}, 验证准确率: {checkpoint['val_acc']:.2f}%\n")
    
    return model, checkpoint


@torch.no_grad()
def evaluate_model(model, val_loader, device):
    """
    评估模型性能
    
    Returns:
        y_true: 真实标签
        y_pred: 预测标签
    """
    print("评估模型性能...")
    model.eval()
    
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(val_loader, desc='评估中'):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    
    # 计算准确率
    accuracy = 100. * np.sum(y_true == y_pred) / len(y_true)
    print(f"验证集准确率: {accuracy:.2f}%")
    
    # 计算F1分数
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"F1分数 (Macro): {f1_macro:.4f}")
    print(f"F1分数 (Weighted): {f1_weighted:.4f}\n")
    
    # 打印分类报告（包含每个类别的Precision/Recall/F1）
    print("详细分类报告:")
    print("-"*60)
    report = classification_report(y_true, y_pred, target_names=config.CLASSES, digits=4)
    print(report)
    
    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 原始计数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 归一化
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存到: {save_path}")
    
    plt.show()


@torch.no_grad()
def predict_test_set(model, test_loader, test_image_paths, device):
    """
    预测测试集并生成提交文件
    
    Returns:
        submission_df: 提交文件的DataFrame
    """
    print("预测测试集...")
    model.eval()
    
    all_preds, all_filenames = [], []
    
    for images, labels, filenames in tqdm(test_loader, desc='预测中'):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_filenames.extend(filenames)
    
    # 解码标签
    predicted_labels = [config.CLASSES[pred] for pred in all_preds]
    
    # 创建提交文件
    submission_df = pd.DataFrame({'file': all_filenames, 'species': predicted_labels})
    submission_df = submission_df.sort_values('file').reset_index(drop=True)
    
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"✓ 提交文件已保存到: {config.SUBMISSION_PATH}")
    print(f"共预测 {len(submission_df)} 张图片\n")
    
    print("预测类别分布:")
    print(submission_df['species'].value_counts())
    
    return submission_df


def visualize_predictions(model, test_loader, device, num_samples=10):
    """
    可视化预测结果
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        num_samples: 显示的样本数量
    """
    print(f"\n可视化 {num_samples} 个预测结果...")
    model.eval()
    
    # 获取一个batch
    images, labels, filenames = next(iter(test_loader))
    images = images[:num_samples].to(device)
    filenames = filenames[:num_samples]
    
    # 预测
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, predicted = probs.max(1)
    
    # 转换为numpy
    images_cpu = images.cpu()
    predicted_labels = [config.CLASSES[p] for p in predicted.cpu().numpy()]
    confidences_cpu = confidences.cpu().numpy()
    
    # 绘制
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # 反归一化
        img = images_cpu[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"{predicted_labels[i]}\n{confidences_cpu[i]:.2%}", fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = config.PLOTS_DIR / "sample_predictions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 预测示例已保存到: {save_path}\n")
    plt.show()


def run_evaluation():
    """
    运行完整的评估流程
    
    Returns:
        submission_df: 提交文件DataFrame
    """
    print("="*60)
    print("开始评估流程")
    print("="*60 + "\n")
    
    device = torch.device(config.DEVICE)
    print(f"使用设备: {device}\n")
    
    # 加载数据
    from data_module import create_data_loaders
    train_loader, val_loader, test_loader, class_to_idx, test_paths = create_data_loaders()
    
    # 加载模型
    model, checkpoint = load_best_model(device)
    
    # 评估
    y_true, y_pred = evaluate_model(model, val_loader, device)
    
    # 绘制混淆矩阵
    cm_path = config.PLOTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    # 预测测试集
    submission_df = predict_test_set(model, test_loader, test_paths, device)
    
    # 可视化预测结果
    visualize_predictions(model, test_loader, device, num_samples=10)
    
    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)
    
    return submission_df
