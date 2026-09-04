"""
Alice 模型配置文件

"""

import torch
from pathlib import Path


class DataConfig:
    """数据处理相关配置"""
    
    CHUNK_SIZE = 1024
    OVERLAP = 128
    BATCH_SIZE = 4
    
    MASK_PROBABILITY = 0.15
    RANDOM_REPLACE_PROB = 0.1
    KEEP_ORIGINAL_PROB = 0.1
    
    PAD_TOKEN_ID = 0
    UNK_TOKEN_ID = 100
    CLS_TOKEN_ID = 101
    SEP_TOKEN_ID = 102
    MASK_TOKEN_ID = 103
    
    DATA_DIR = Path("data")
    TRAIN_DATA_PATTERNS = ["*.txt", "*.jsonl"]


class ModelConfig:
    """模型架构相关配置"""
    
    M = 1024
    N = 1024
    P = 512
    Q = 512
    
    NUM_ACTIVATE_CLASSES = 72
    
    # ========== 激活阈值（三阶段）==========
    # 阶段 1: 早期 (0-33%) - 宽松筛选，高低阈值差距大
    # 阶段 2: 中期 (34-66%) - 中等筛选，高低阈值差距中等
    # 阶段 3: 后期 (67-100%) - 严格筛选，高低阈值差距小
    
    ACTIVATE_THRESHOLDS_STAGE1 = torch.tensor([2] * 72)
    ACTIVATE_THRESHOLDS_STAGE2 = torch.tensor([3] * 72)
    ACTIVATE_THRESHOLDS_STAGE3 = torch.tensor([4] * 72)
    
    # Remain 辅助阈值（三阶段）
    ACTIVATE_THRESHOLD_LOW_STAGE1 = 0.2   # 差距 0.3 (宽松)
    ACTIVATE_THRESHOLD_LOW_STAGE2 = 0.5   # 差距 0.2 (中等)
    ACTIVATE_THRESHOLD_LOW_STAGE3 = 0.8   # 差距 0.1 (严格)
    
    # 当前使用阶段（运行时设置）
    CURRENT_STAGE = 1  # 1, 2, or 3
    
    @classmethod
    def get_threshold_for_stage(cls, stage: int) -> tuple:
        """
        获取指定阶段的阈值（已注册为 buffer，可直接用于模型）
        
        Args:
            stage: 阶段编号 (1/2/3)
        
        Returns:
            (r_high_tensor, r_low_tensor) - 都是张量
        """
        if stage == 1:
            return (
                cls.ACTIVATE_THRESHOLDS_STAGE1.clone(), torch.tensor(cls.ACTIVATE_THRESHOLD_LOW_STAGE1)
            )
        elif stage == 2:
            return (
                cls.ACTIVATE_THRESHOLDS_STAGE2.clone(), torch.tensor(cls.ACTIVATE_THRESHOLD_LOW_STAGE2)
            )
        else:
            return (
                cls.ACTIVATE_THRESHOLDS_STAGE3.clone(), torch.tensor(cls.ACTIVATE_THRESHOLD_LOW_STAGE3)
            )
    
    # Reason 类型分配
    REASON_TYPE1_INDICES = list(range(8))      # 0-7
    REASON_TYPE3_INDICES = list(range(68, 72)) # 68-71
    REASON_TYPE2_INDICES = list(range(8, 68))  # 8-67


class RemainConfig:
    """Remain 机制相关配置"""
    
    DECAY_RATE = 0.3  # 指数衰减率 λ=0.3
    # decay_factor = e^(-0.3) ≈ 0.7408


class FlowConfig:
    """推理流程相关配置"""
    
    MAX_ITERATIONS = 10             # 最大迭代轮次
    
    # 池子容量
    MAX_ACTIVATED_X = 5             # activated_pool 容量
    MAX_UNACTIVATED_X = 5           # pending_pool 容量
    MAX_OUTPUT_PER_LABEL = 1        # 每个 label 的输出表容量上限
    
    ACTIVATE_USAGE_PENALTY = 0.01   # 激活次数软限制惩罚系数
    
    # ========== 长期记忆配置 ==========
    USE_LONG_TERM_MEMORY = False     # 是否启用长期记忆
    LONG_TERM_MEMORY_PATH = None     # 预训练长期记忆路径
    
    # TrajectoryRecorder 超参数
    LONG_TERM_MEMORY_TOPK = 1              # 查询时返回的 topk 值
    LONG_TERM_MEMORY_MAX_SIZE = 10         # 记忆库最大容量
    LONG_TERM_MEMORY_MIN_USAGE = 3         # 最小使用次数阈值
    LONG_TERM_MEMORY_USAGE_BONUS = 0.1     # 使用次数加权系数


class LossConfig:
    """损失函数相关配置"""
    
    W_OUT_INIT = 1.0
    W_OUT_FINAL = 0.5
    
    @classmethod
    def get_weights(cls, progress_ratio):
        """获取当前权重（线性插值）"""
        w_out = cls.W_OUT_INIT + (cls.W_OUT_FINAL - cls.W_OUT_INIT) * progress_ratio
        return w_out


class OptimizerConfig:
    """优化器相关配置"""
    
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.1
    BETAS = (0.9, 0.999)
    MAX_GRAD_NORM = 1.0


class SpecialOutputConfig:
    """特殊输出配置"""
    
    INDICES = list(range(68, 72))  # Type3 Reason (forget 类型)


class Config:
    """统一配置类"""
    
    data = DataConfig()
    model = ModelConfig()
    remain = RemainConfig()
    flow = FlowConfig()
    loss = LossConfig()
    optimizer = OptimizerConfig()
    special_output = SpecialOutputConfig()
    
    train = type('TrainConfig', (), {
        'DEVICE': 'cuda',
        'NUM_EPOCHS': 1,  # ✅ 测试用 1 轮
        'LOG_INTERVAL': 10,
        'SAVE_INTERVAL': 1,  # ✅ 每轮都保存
    })()
