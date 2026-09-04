"""
项目配置文件

统一管理所有超参数、路径配置和训练设置
"""

from pathlib import Path


# ========== 数据相关配置 ==========

class DataConfig:
    """数据处理相关配置"""
    
    # 文本块配置
    CHUNK_SIZE = 1024              # 每个文本块的 token 数
    OVERLAP = 128                  # 块间重叠的 token 数
    BATCH_SIZE = 4                 # 每批处理的文本块数
    
    # 数据加载
    NUM_WORKERS = 4                # 数据加载进程数
    PIN_MEMORY = True              # 锁定内存加速 CPU→GPU 传输
    PREFETCH_FACTOR = 2            # 预取因子
    
    # MLM 配置
    MASK_PROBABILITY = 0.15        # 被 mask 的概率
    RANDOM_REPLACE_PROB = 0.1      # 随机替换概率
    KEEP_ORIGINAL_PROB = 0.1       # 保持不变概率
    
    # 特殊 tokens (BERT-base-chinese)
    PAD_TOKEN_ID = 0
    UNK_TOKEN_ID = 100
    CLS_TOKEN_ID = 101
    SEP_TOKEN_ID = 102
    MASK_TOKEN_ID = 103
    
    # 数据路径
    DATA_DIR = Path("data")
    TRAIN_DATA_PATTERNS = ["*.txt", "*.jsonl"]


# ========== 模型架构配置 ==========

class ModelConfig:
    """模型架构相关配置"""
    
    # 输入尺寸
    M = 1024                       # 输入矩阵行数（也是词嵌入维度）
    N = 1024                       # 输入矩阵列数
    
    # Reason 中间维度
    P = 512                        # Reason 中间变换行数
    Q = 512                        # Reason 中间变换列数
    
    # 模块数量
    NUM_ACTIVATE_CLASSES = 72      # Activate-Reason 对数量
    NUM_MANAGER_CLASSES = 8        # 辅助管理类数量
    NUM_SPECIAL_OUTPUT_CLASSES = 10  # 直接输出的类数量
    
    # 阈值配置（每个类独立）
    ACTIVATE_THRESHOLDS = [0.5] * 72  # 激活类阈值列表（向后兼容）
    MANAGER_THRESHOLDS = [0.3] * 8    # Manager 阈值列表（通常比 Activate 小）
    
    # 双重阈值配置
    ACTIVATE_THRESHOLD_HIGH = 0.7     # 直接激活阈值
    ACTIVATE_THRESHOLD_LOW = 0.5      # remain 辅助阈值
    
    # STE (Straight-Through Estimator) 配置
    USE_STE = False                   # 是否启用 STE 训练
    STE_K = 10.0                      # STE 斜率
    EPOCH_THRESHOLD = 10              # 早期/后期的 epoch 阈值
    
    # TopK 松弛化配置
    MANAGER_TOP_K = 3                 # Manager 选择的 top-k 数量
    ACTIVATE_TOP_K = 5                # Activate 选择的 top-k 数量
    
    # Reason 类型分配
    # 第一类 (Pass-through): 0-7
    REASON_TYPE1_INDICES = list(range(8))  # [0,1,2,3,4,5,6,7]
    # 第三类 (With Forget): 8, 16, 24, 32
    REASON_TYPE3_INDICES = [8, 16, 24, 32]  # 4 个带遗忘门的
    # 第二类 (Standard): 其他所有（在 __init__ 中计算）
    
    @classmethod
    def get_reason_type2_indices(cls):
        """获取第二类 Reason 索引"""
        all_indices = set(range(72))
        type1_set = set(cls.REASON_TYPE1_INDICES)
        type3_set = set(cls.REASON_TYPE3_INDICES)
        return sorted(list(all_indices - type1_set - type3_set))
    
    # Manager 管理范围（交叉分配）
    # 每个 Manager 管理：1 个第一类 + 10 个（第二类 + 第三类）
    # 8 组围成圈，每两组交叉 2 个
    MANAGER_RANGES = [
        [0] + list(range(8, 16)),     # M0: Pass_0 + [8-15]
        [1] + list(range(14, 24)),    # M1: Pass_1 + [14-23], 交叉 14-15
        [2] + list(range(22, 32)),    # M2: Pass_2 + [22-31], 交叉 22-23
        [3] + list(range(30, 40)),    # M3: Pass_3 + [30-39], 交叉 30-31
        [4] + list(range(38, 48)),    # M4: Pass_4 + [38-47], 交叉 38-39
        [5] + list(range(46, 56)),    # M5: Pass_5 + [46-55], 交叉 46-47
        [6] + list(range(54, 64)),    # M6: Pass_6 + [54-63], 交叉 54-55
        [7] + list(range(62, 72)),    # M7: Pass_7 + [62-71], 交叉 62-63
    ]


# ========== 训练流程控制配置 ==========

class TrainingFlowConfig:
    """训练流程控制相关配置"""
    
    # 迭代限制
    MAX_ITERATIONS = 100           # MainModel 最大迭代轮次
    MAX_X_COUNT = 50               # 每轮 X 总数上限
    MAX_ACTIVATED_X = 40           # 激活的 X 数量
    MAX_UNACTIVATED_X = 10         # 未激活的 X 数量
    MAX_AGE_WITHOUT_ACTIVATION = 3  # X 最大年龄（未被激活的存活轮次）
    MAX_ACTIVATE_USAGE = 10        # 每个 Activate 最大使用次数
    
    # 衰减系数
    DECAY_RATE_A = 0.3             # Remain 衰减系数（可随训练阶段调整）


# ========== 损失函数配置 ==========

class LossConfig:
    """损失函数相关配置"""
    
    # 三阶段损失权重（根据训练进度调整）
    # 阶段 1: 早期（1-33%），阶段 2: 中期（34-66%），阶段 3: 后期（67-100%）
    
    # 输出损失权重 L_output
    OUTPUT_LOSS_WEIGHTS = [1.0, 1.0, 1.0]  # 始终为 1
    
    # 推理次数惩罚权重 L_steps
    STEPS_PENALTY_WEIGHTS = [0.1, 0.01, 0.05]
    
    # 激活分布熵权重 L_balance
    BALANCE_ENTROPY_WEIGHTS = [0.01, 0.05, 0.1]
    
    @classmethod
    def get_weights(cls, progress_ratio: float) -> tuple:
        """
        根据训练进度获取当前权重
        
        Args:
            progress_ratio: 训练进度 (0.0 ~ 1.0)
            
        Returns:
            (output_weight, steps_weight, balance_weight)
        """
        if progress_ratio < 0.33:
            stage = 0
        elif progress_ratio < 0.66:
            stage = 1
        else:
            stage = 2
        
        return (
            cls.OUTPUT_LOSS_WEIGHTS[stage],
            cls.STEPS_PENALTY_WEIGHTS[stage],
            cls.BALANCE_ENTROPY_WEIGHTS[stage]
        )


# ========== 优化器配置 ==========

class OptimizerConfig:
    """优化器相关配置"""
    
    LEARNING_RATE = 1e-4           # 初始学习率
    WEIGHT_DECAY = 0.01            # 权重衰减
    BETAS = (0.9, 0.999)          # AdamW 的 beta 参数
    EPS = 1e-8                     # AdamW 的 epsilon
    
    # 学习率调度（暂不实现，后续讨论）
    # LR_SCHEDULER_TYPE = "cosine"
    # WARMUP_RATIO = 0.1


# ========== 训练循环配置 ==========

class TrainLoopConfig:
    """训练循环相关配置"""
    
    NUM_EPOCHS = 10                # 训练轮数
    GRAD_CLIP_NORM = 1.0           # 梯度裁剪范数
    LOG_INTERVAL = 10              # 日志打印间隔（batch 数）
    SAVE_INTERVAL = 1              # checkpoint 保存间隔（epoch 数）
    
    # 设备
    DEVICE = "cuda"                # 训练设备
    USE_AMP = False                # 是否使用混合精度训练（后续实现）
    USE_GRADIENT_CHECKPOINTING = False  # 是否使用梯度检查点（后续实现）
    
    # 检查点路径
    CHECKPOINT_DIR = Path("checkpoints")
    RESUME_FROM = None             # 恢复训练的 checkpoint 路径


# ========== 特殊输出类配置 ==========

class SpecialOutputConfig:
    """特殊输出类索引配置"""
    
    # 默认均匀分布选择 10 个类
    # 可根据实验结果调整
    INDICES = list(range(0, 72, 7))[:10]  # [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]


# ========== 完整配置集合 ==========

class Config:
    """完整的配置集合"""
    
    data = DataConfig()
    model = ModelConfig()
    flow = TrainingFlowConfig()
    loss = LossConfig()
    optimizer = OptimizerConfig()
    train = TrainLoopConfig()
    special_output = SpecialOutputConfig()
    
    @classmethod
    def print_summary(cls):
        """打印配置摘要"""
        print("="*60)
        print("配置摘要")
        print("="*60)
        
        print(f"\n数据配置:")
        print(f"  文本块大小：{cls.data.CHUNK_SIZE}")
        print(f"  批次大小：{cls.data.BATCH_SIZE}")
        print(f"  重叠量：{cls.data.OVERLAP}")
        
        print(f"\n模型配置:")
        print(f"  输入尺寸：{cls.model.M}×{cls.model.N}")
        print(f"  Activate-Reason 对数量：{cls.model.NUM_ACTIVATE_CLASSES}")
        print(f"  特殊输出类数量：{cls.model.NUM_SPECIAL_OUTPUT_CLASSES}")
        
        print(f"\n流程控制:")
        print(f"  最大迭代次数：{cls.flow.MAX_ITERATIONS}")
        print(f"  每轮 X 上限：{cls.flow.MAX_X_COUNT}")
        print(f"  衰减系数：{cls.flow.DECAY_RATE_A}")
        
        print(f"\n损失权重（三阶段）:")
        print(f"  输出损失：{cls.loss.OUTPUT_LOSS_WEIGHTS}")
        print(f"  步数惩罚：{cls.loss.STEPS_PENALTY_WEIGHTS}")
        print(f"  平衡熵：{cls.loss.BALANCE_ENTROPY_WEIGHTS}")
        
        print(f"\n优化器:")
        print(f"  学习率：{cls.optimizer.LEARNING_RATE}")
        print(f"  权重衰减：{cls.optimizer.WEIGHT_DECAY}")
        
        print(f"\n训练:")
        print(f"  训练轮数：{cls.train.NUM_EPOCHS}")
        print(f"  设备：{cls.train.DEVICE}")
        
        print("="*60)


if __name__ == "__main__":
    Config.print_summary()
