"""配置文件 - EfficientNet-B0植物幼苗分类"""

import torch
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "plant-seedlings-classification"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_SAVE_PATH = OUTPUT_DIR / "best_model.pth"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
PLOTS_DIR = OUTPUT_DIR / "plots"

# 创建输出目录
for dir_path in [OUTPUT_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据配置
CLASSES = [
    'Black-grass', 'Charlock', 'Cleavers', 
    'Common Chickweed', 'Common wheat', 'Fat Hen',
    'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
    'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet'
]
NUM_CLASSES = len(CLASSES)

# 图像和批次配置
IMG_SIZE = 224        # EfficientNet-B0输入尺寸
BATCH_SIZE = 32       # 批次大小
NUM_WORKERS = 4       # 数据加载worker数量

# 训练配置
EPOCHS = 30           # 训练轮数
LEARNING_RATE = 1e-3  # 初始学习率
WEIGHT_DECAY = 1e-4   # 权重衰减
MOMENTUM = 0.9        # SGD动量
EARLY_STOP_PATIENCE = 5  # 早停耐心值

# 学习率调度器配置
# 前10个epoch固定学习率，后20个epoch使用余弦退火（先降后升）
LR_SCHEDULER = {
    'type': 'CosineAnnealingLR',  # 简单余弦退火
    'T_max': 20,                  # 余弦周期长度（后20个epoch）
    'eta_min': 1e-5               # 最小学习率
}

# 模型配置
MODEL_NAME = 'efficientnet_b0'
PRETRAINED = True     # 使用ImageNet预训练权重
DROPOUT = 0.3         # Dropout比例
LABEL_SMOOTHING = 0.1 # 标签平滑系数

# 类别权重 (用于处理类别不平衡和难分类别)
# Black-grass (索引0) 和 Loose Silky-bent (索引6) 容易混淆，给Black-grass更高权重
CLASS_WEIGHTS = [
    1.5,  # Black-grass (提高权重，因为最难分类)
    1.0,  # Charlock
    1.0,  # Cleavers
    1.0,  # Common Chickweed
    1.0,  # Common wheat
    1.0,  # Fat Hen
    1.2,  # Loose Silky-bent (稍微提高，因为与Black-grass混淆)
    1.0,  # Maize
    1.0,  # Scentless Mayweed
    1.0,  # Shepherds Purse
    1.0,  # Small-flowered Cranesbill
    1.0,  # Sugar beet
]

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
