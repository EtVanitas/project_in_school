"""
Alice 模型配置文件

"""

from pathlib import Path


class DataConfig:
    """数据处理相关配置"""
    
    # 文本分块配置
    CHUNK_SIZE = 1024  # 文本块大小
    OVERLAP = 128      # 重叠大小
    BATCH_SIZE = 1     # 批次大小
    
    # 掩码率配置（可调整）
    MASK_PROBABILITY = 0.05  # 初始 5%
    
    DATA_DIR = Path("data")
    TRAIN_DATA_PATTERNS = ["*.txt", "*.jsonl"]


class ModelConfig:
    """模型架构相关配置"""
    
    M = 1024
    N = 1024
    P = 512
    Q = 512
    
    NUM_ACTIVATE_CLASSES = 32
    
    # Reason 类型分配
    REASON_TYPE1_INDICES = list(range(4))      # 0-3 (simple)
    REASON_TYPE3_INDICES = list(range(30, 32)) # 30-31 (forget/Type3)
    REASON_TYPE2_INDICES = list(range(4, 30))  # 4-29 (transform)


class RemainConfig:
    """Remain 机制相关配置"""
    
    DECAY_RATE = 0.3  # 指数衰减率 λ=0.3，e^(-0.3)≈0.7，每轮衰减
    # decay_factor = e^(-0.3) ≈ 0.7408


class FlowConfig:
    """推理流程相关配置"""
    MAX_ITERATIONS = 10
    
    # 激活阈值
    HIGH_THRESHOLD = 0.6
    LOW_THRESHOLD = 0.4
    ACTIVATED_POOL_MAX_SIZE = 30
    
    # 输出列表上限（防止显存爆炸）
    OUTPUT_LIST_MAX_SIZE = 10
    
    # 长期记忆配置
    ENABLE_LONG_TERM_MEMORY = False
    LONG_TERM_TOPK = 5
    LONG_TERM_BATCH_SIZE = 100  # 每 100 条保存为一个文件
    LONG_TERM_STORAGE_DIR = "long_term_memories"  # 存储目录
    LONG_TERM_MIN_ACTIVATION = 0.9  # 长期记忆最小激活值阈值
    LONG_TERM_MAX_TRAJECTORIES = 100  # 滑动窗口：只保留最新 100 条 [x2, remain] 对
    
    # 训练优化配置
    ENABLE_GRADIENT_CHECKPOINTING = False  # 已禁用：树状剪枝导致梯度路径不一致
    ENABLE_MIXED_PRECISION = True  # 混合精度训练（安全）


class OptimizerConfig:
    """优化器相关配置"""
    
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.1
    BETAS = (0.9, 0.999)
    MAX_GRAD_NORM = 1.0


class Config:
    """统一配置类"""
    
    data = DataConfig()
    model = ModelConfig()
    remain = RemainConfig()
    flow = FlowConfig()
    optimizer = OptimizerConfig()
    
    train = type('TrainConfig', (), {
        'DEVICE': 'cuda',
        'LOG_INTERVAL': 10,
        'SAVE_INTERVAL': 1,  # 每轮都保存
        'EPOCHS_PER_CHUNK': 10,  # 每个文本块的训练轮数
    })()
