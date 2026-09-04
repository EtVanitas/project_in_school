#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大语言模型完整框架
包含数据处理、模型架构、训练流程、微调和强化学习等全过程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import json
import os
from transformers import AutoTokenizer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 配置管理模块 ====================
class ModelConfig:
    """模型配置管理器"""
    
    def __init__(self, vocab_size: int = 21128, text_seq_len: int = 512, 
                 use_gradient_checkpointing: bool = True):
        self.vocab_size = vocab_size
        self.text_seq_len = text_seq_len
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'vocab_size': self.vocab_size,
            'text_seq_len': self.text_seq_len,
            'use_gradient_checkpointing': self.use_gradient_checkpointing
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """从字典创建配置对象"""
        return cls(
            vocab_size=config_dict.get('vocab_size', 21128),
            text_seq_len=config_dict.get('text_seq_len', 512),
            use_gradient_checkpointing=config_dict.get('use_gradient_checkpointing', True)
        )

# ==================== 数据处理模块 ====================
class TextChunkProcessor(Dataset):
    """文本块处理器 - 将文本分词后切割成固定长度的token块"""
    
    def __init__(self, texts: List[str], tokenizer, chunk_size: int = 512):
        """
        初始化文本块处理器
        Args:
            texts: 文本列表
            tokenizer: 分词器 (使用bert-base-chinese)
            chunk_size: token块大小，默认512
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # 预处理所有文本，生成token块
        self.chunks = self._process_texts_to_chunks(texts)
        logger.info(f"总共生成 {len(self.chunks)} 个文本块，每个块大小: {chunk_size}")
    
    def _process_texts_to_chunks(self, texts: List[str]) -> List[List[int]]:
        """将文本列表处理成token块"""
        all_chunks = []
        
        for text in texts:
            if not text.strip():
                continue
                
            # 分词编码
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,  # 不添加特殊token，保持纯净的token序列
                return_attention_mask=False
            )
            tokens = encoding['input_ids']
            
            # 切割成固定大小的块
            for i in range(0, len(tokens), self.chunk_size):
                chunk = tokens[i:i + self.chunk_size]
                
                # 如果块不够长，进行填充
                if len(chunk) < self.chunk_size:
                    chunk = chunk + [self.pad_token_id] * (self.chunk_size - len(chunk))
                
                all_chunks.append(chunk)
        
        return all_chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        """获取单个文本块"""
        chunk = self.chunks[idx]
        
        # 转换为tensor
        input_ids = torch.tensor(chunk, dtype=torch.long)
        
        return {
            'input_ids': input_ids
        }

class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, tokenizer_name: str = 'bert-base-chinese'):
        """
        初始化文本预处理器
        Args:
            tokenizer_name: 预训练分词器名称，默认使用bert-base-chinese
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"成功加载分词器: {tokenizer_name}")
        except Exception as e:
            logger.error(f"加载分词器失败: {e}")
            raise e
    
    def load_jsonl_data(self, file_path: str) -> List[str]:
        """
        加载JSONL格式数据
        Args:
            file_path: JSONL文件路径
        Returns:
            文本列表
        """
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        # 支持多种字段名
                        text_fields = ['text', 'content', 'sentence', 'prompt', 'input']
                        text = None
                        
                        # 查找文本字段
                        for field in text_fields:
                            if field in data and data[field]:
                                text = str(data[field])
                                break
                        
                        # 如果没找到标准字段，尝试使用第一个字符串值
                        if text is None:
                            for value in data.values():
                                if isinstance(value, str) and value.strip():
                                    text = value.strip()
                                    break
                        
                        if text:
                            texts.append(text)
                        else:
                            logger.warning(f"第{line_num}行未找到有效文本字段")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析错误: {e}")
                    except Exception as e:
                        logger.warning(f"第{line_num}行处理错误: {e}")
        except Exception as e:
            logger.error(f"加载JSONL数据失败: {e}")
        
        logger.info(f"成功加载 {len(texts)} 条数据")
        return texts
    
    def create_text_dataloader(self, texts: List[str], chunk_size: int = 512,
                              batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        """
        创建文本数据加载器 - 将文本分词后切割成固定长度的token块
        Args:
            texts: 文本列表
            chunk_size: token块大小，默认512
            batch_size: 批次大小，默认1（逐个块处理）
            shuffle: 是否打乱数据，默认False（保持顺序）
        Returns:
            DataLoader对象
        """
        dataset = TextChunkProcessor(texts, self.tokenizer, chunk_size=chunk_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ==================== 图像处理模块 ====================
class ImagePreprocessor:
    """图像预处理器基类（预留）"""
    
    def __init__(self):
        """初始化图像预处理器"""
        logger.info("图像预处理器已初始化（预留接口）")
        # TODO: 实现具体的图像预处理逻辑
    
    def load_images(self, image_paths: List[str]) -> List:
        """
        加载图像数据（预留接口）
        Args:
            image_paths: 图像文件路径列表
        Returns:
            图像数据列表
        """
        # TODO: 实现图像加载逻辑
        raise NotImplementedError("图像加载功能待实现")
    
    def create_image_dataloader(self, images: List, batch_size: int = 1, 
                               shuffle: bool = False) -> DataLoader:
        """
        创建图像数据加载器（预留接口）
        Args:
            images: 图像数据列表
            batch_size: 批次大小
            shuffle: 是否打乱数据
        Returns:
            DataLoader对象
        """
        # TODO: 实现图像数据加载器
        raise NotImplementedError("图像数据加载器待实现")

class ImageChunkProcessor(Dataset):
    """图像块处理器（预留）"""
    
    def __init__(self, images: List, chunk_size: tuple = (224, 224)):
        """
        初始化图像块处理器
        Args:
            images: 图像数据列表
            chunk_size: 图像块大小 (height, width)
        """
        self.images = images
        self.chunk_size = chunk_size
        # TODO: 实现图像块处理逻辑
        logger.info(f"图像块处理器已初始化，块大小: {chunk_size}（预留接口）")
    
    def __len__(self):
        # TODO: 实现长度计算
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取单个图像块（预留）"""
        # TODO: 实现图像块获取逻辑
        raise NotImplementedError("图像块处理待实现")

# ==================== 自定义矩阵神经网络模块 ====================
class CustomMatrixBlock(nn.Module):
    """自定义矩阵神经网络块 - 实现 7次矩阵相乘 + 7次非线性处理"""
    
    def __init__(self, 
                 x_rows: int,           # X矩阵行数
                 x_cols: int,           # X矩阵列数
                 in_features: int,      # 输入特征维度
                 out_features: int,     # 输出特征维度
                 hidden_a: int,         # A矩阵中间维度
                 hidden_b: int,         # B矩阵中间维度
                 hidden_c: int,         # C矩阵中间维度
                 hidden_d: int,         # D矩阵中间维度
                 position_sequence: str = "0000000"):   # 7位统一控制序列
        """
        初始化自定义矩阵块
        Args:
            x_rows: X矩阵行数
            x_cols: X矩阵列数
            in_features: 输入特征维度
            out_features: 输出特征维度
            hidden_a/b/c/d: 各矩阵的中间隐藏维度
            position_sequence: 7位数字字符串，统一控制非线性处理
                             0: 不处理, 1: SILU处理, 2: GELU处理
        """
        super().__init__()
        
        # 验证输入参数
        assert len(position_sequence) == 7, "position_sequence必须是7位数字"
        
        self.in_features = in_features
        self.out_features = out_features
        self.x_rows = x_rows
        self.x_cols = x_cols
        
        # 解析控制参数
        self.position_controls = [int(x) for x in position_sequence]
        
        # 中间维度（用于矩阵分解）
        self.hidden_a = hidden_a
        self.hidden_b = hidden_b
        self.hidden_c = hidden_c
        self.hidden_d = hidden_d
        
        # 创建分解矩阵参数 (严格按照指定维度)
        # A = A1 × A2: [in_features, x_rows] = [in_features, hidden_a] × [hidden_a, x_rows]
        self.A1 = nn.Parameter(torch.randn(in_features, hidden_a))
        self.A2 = nn.Parameter(torch.randn(hidden_a, x_rows))
        
        # B = B1 × B2: [x_cols, x_cols] = [x_cols, hidden_b] × [hidden_b, x_cols]
        self.B1 = nn.Parameter(torch.randn(x_cols, hidden_b))
        self.B2 = nn.Parameter(torch.randn(hidden_b, x_cols))
        
        # C = C1 × C2: [x_rows, x_rows] = [x_rows, hidden_c] × [hidden_c, x_rows]
        self.C1 = nn.Parameter(torch.randn(x_rows, hidden_c))
        self.C2 = nn.Parameter(torch.randn(hidden_c, x_rows))
        
        # D = D1 × D2: [x_cols, out_features] = [x_cols, hidden_d] × [hidden_d, out_features]
        self.D1 = nn.Parameter(torch.randn(x_cols, hidden_d))
        self.D2 = nn.Parameter(torch.randn(hidden_d, out_features))
        
        # 激活函数
        self.silu = nn.SiLU()
        self.gelu = nn.GELU()
        
        # 归一化层
        self.rms_norm_after_AX = nn.RMSNorm(x_cols)      # AX后的RMSNorm
        self.rms_norm_after_AXBXTC = nn.RMSNorm(x_rows)      # AXBXTC后的RMSNorm
        
        # 参数初始化
        self._init_parameters()
    
    def _init_parameters(self):
        """参数初始化"""
        for param in [self.A1, self.A2, self.B1, self.B2, 
                     self.C1, self.C2, self.D1, self.D2]:
            nn.init.xavier_normal_(param)
    
    def _apply_nonlinear(self, tensor: torch.Tensor, control_value: int) -> torch.Tensor:
        """
        应用非线性处理（统一控制逻辑）
        Args:
            tensor: 输入张量
            control_value: 控制值 (0:不处理, 1:SILU处理, 2:GELU处理)
        Returns:
            处理后的张量
        """
        if control_value == 0:  # 不进行非线性处理
            return tensor
        elif control_value == 1:  # 使用SILU函数处理
            return self.silu(tensor)
        elif control_value == 2:  # 使用GELU函数处理
            return self.gelu(tensor)
        else:
            # 对于其他值，保持原样（容错处理）
            return tensor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播实现 AXBX^TCXD 运算
        计算流程: A × X × B × X^T × C × X × D
        Args:
            x: 输入张量 [x_rows, x_cols]
        Returns:
            输出张量 [in_features, out_features]
        """
        
        # 构造大矩阵 (按照正确维度)
        A = torch.matmul(self.A1, self.A2)  # [in_features, x_rows]
        B = torch.matmul(self.B1, self.B2)  # [x_cols, x_cols]
        C = torch.matmul(self.C1, self.C2)  # [x_rows, x_rows]
        D = torch.matmul(self.D1, self.D2)  # [x_cols, out_features]
        
        # 开始AXBX^TCXD计算流程
        
        # 步骤1: 对A进行非线性处理 (0-不处理，1-SILU，2-GELU)
        A_processed = self._apply_nonlinear(A, self.position_controls[0])
        
        # 步骤2: A × X (第一次矩阵相乘) - [in_features, x_rows] × [x_rows, x_cols]
        result_AX = torch.matmul(A_processed, x)  # [in_features, x_cols]
        result_AX = self.rms_norm_after_AX(result_AX)
        
        # 步骤3: 对AX进行非线性处理
        result_AX = self._apply_nonlinear(result_AX, self.position_controls[1])
        
        # 步骤4: (AX) × B (第二次矩阵相乘) - [in_features, x_cols] × [x_cols, x_cols]
        result_AXB = torch.matmul(result_AX, B)  # [in_features, x_cols]
        
        # 步骤5: 对(AX)B进行非线性处理
        result_AXB = self._apply_nonlinear(result_AXB, self.position_controls[2])
        
        # 步骤6: ((AX)B) × X^T (第三次矩阵相乘) - [in_features, x_cols] × [x_cols, x_rows]
        X_T = x.transpose(-2, -1)  # [x_cols, x_rows]
        result_AXBXt = torch.matmul(result_AXB, X_T)  # [in_features, x_rows]
        
        # 步骤7: 对((AX)B)X^T进行非线性处理
        result_AXBXt = self._apply_nonlinear(result_AXBXt, self.position_controls[3])
        
        # 步骤8: (((AX)B)X^T) × C (第四次矩阵相乘) - [in_features, x_rows] × [x_rows, x_rows]
        result_AXBXtC = torch.matmul(result_AXBXt, C)  # [in_features, x_rows]
        
        # 步骤9: 对(((AX)B)X^T)C进行非线性处理
        result_AXBXtC = self._apply_nonlinear(result_AXBXtC, self.position_controls[4])
        result_AXBXtC = self.rms_norm_after_AXBXTC(result_AXBXtC)

        # 步骤10: ((((AX)B)X^T)C) × X (第五次矩阵相乘) - [in_features, x_rows] × [x_rows, x_cols]
        result_AXBXtCX = torch.matmul(result_AXBXtC, x)  # [in_features, x_cols]
        
        # 步骤11: 对((((AX)B)X^T)C)X进行非线性处理
        result_AXBXtCX = self._apply_nonlinear(result_AXBXtCX, self.position_controls[5])
        
        # 步骤12: (((((AX)B)X^T)C)X) × D (第六次矩阵相乘) - [in_features, x_cols] × [x_cols, out_features]
        final_result = torch.matmul(result_AXBXtCX, D)  # [in_features, out_features]
        
        # 步骤13: 对最终结果进行非线性处理
        final_result = self._apply_nonlinear(final_result, self.position_controls[6])
        
        # 直接返回最终结果
        return final_result  # [in_features, out_features]

# ==================== 主模型模块 ====================
class MainModel(nn.Module):
    """主模型类 - 文本处理核心架构，包含20个CustomMatrixBlock层"""
    
    def __init__(self, vocab_size: int = 21128, text_seq_len: int = 512, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.config = ModelConfig(vocab_size, text_seq_len, use_gradient_checkpointing)
        self.vocab_size = self.config.vocab_size
        self.text_seq_len = self.config.text_seq_len
        self.use_gradient_checkpointing = self.config.use_gradient_checkpointing
        
        # 词嵌入层
        self.text_embedding = nn.Embedding(vocab_size, 1024)
        
        # RMSNorm层（共19个，对应19次运算）
        self.rms_norm_layers = nn.ModuleList([
            nn.RMSNorm(1024),   # block_0输入
            nn.RMSNorm(2048),   # blocks_1-6输入
            nn.RMSNorm(2048),   # block_17_1输入
            nn.RMSNorm(2048),   # block_17_2输入
            nn.RMSNorm(2048),   # blocks_7-12输入
            nn.RMSNorm(4096),   # block_18输入（记忆矩阵）
            nn.RMSNorm(2048),   # blocks_13-16输入
            nn.RMSNorm(2048),   # block_19输入
            nn.RMSNorm(1024),   # logits计算前
            nn.RMSNorm(2048),   # blocks_1残差连接后
            nn.RMSNorm(2048),   # blocks_2残差连接后
            nn.RMSNorm(2048),   # blocks_3残差连接后
            nn.RMSNorm(2048),   # blocks_4残差连接后
            nn.RMSNorm(2048),   # blocks_5残差连接后
            nn.RMSNorm(2048),   # blocks_6残差连接后
            nn.RMSNorm(2048),   # blocks_7-12残差连接后
            nn.RMSNorm(2048),   # blocks_13残差连接后
            nn.RMSNorm(2048),   # blocks_14残差连接后
            nn.RMSNorm(2048),   # blocks_15残差连接后
        ])
        
        # CustomMatrixBlocks配置
        self.block_0 = CustomMatrixBlock(
            x_rows=512, x_cols=1024, in_features=1024, out_features=2048,
            hidden_a=2048, hidden_b=2048, hidden_c=1024, hidden_d=2048,
            position_sequence="2111112"
        )
        
        self.blocks_1_to_16 = nn.ModuleList([
            CustomMatrixBlock(1024, 2048, 1024, 2048, 512, 1024, 512, 1024, "2111112"),
            CustomMatrixBlock(1024, 2048, 1024, 2048, 1024, 2048, 1024, 2048, "2111112"),
            CustomMatrixBlock(1024, 2048, 1024, 2048, 1536, 3072, 1536, 3072, "2111112"),
            CustomMatrixBlock(1024, 2048, 1024, 2048, 2048, 4096, 2048, 4096, "2111112"),
        ]*4)
        
        self.block_17_1 = CustomMatrixBlock(
            x_rows=1024, x_cols=2048, in_features=4096, out_features=4096,
            hidden_a=2048, hidden_b=4096, hidden_c=2048, hidden_d=2048,
            position_sequence="2111112"
        )
        self.block_17_2 = CustomMatrixBlock(
            x_rows=1024, x_cols=2048, in_features=1024, out_features=2048,
            hidden_a=2048, hidden_b=4096, hidden_c=2048, hidden_d=4096,
            position_sequence="1121211"
        )
        
        self.block_18 = CustomMatrixBlock(
            x_rows=4096, x_cols=4096, in_features=1024, out_features=2048,
            hidden_a=2048, hidden_b=2048, hidden_c=2048, hidden_d=2048,
            position_sequence="2111112"
        )
        
        self.block_19 = CustomMatrixBlock(
            x_rows=1024, x_cols=2048, in_features=512, out_features=1024,
            hidden_a=2048, hidden_b=4096, hidden_c=2048, hidden_d=2048,
            position_sequence="2112112"
        )
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """参数初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用较小的增益避免梯度爆炸
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # 词嵌入使用更小的标准差
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Parameter):
                nn.init.xavier_uniform_(module, gain=0.1)
    
    def _checkpointed_blocks_7_to_12(self, x):
        """
        梯度检查点包装的blocks_7-12处理
        Args:
            x: 输入张量 [1024, 2048]
        Returns:
            处理后的张量 [1024, 2048]
        """
        # blocks_7-12: 残差连接处理 (索引6-11对应block_7-block_12)
        for j in range(6, 12):
            residual = x.clone()
            x = self.blocks_1_to_16[j](x)
            x = x + residual  # 残差连接
        return x
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        主模型前向传播
        Args:
            input_ids: 输入token IDs [batch_size, seq_len] 或 [seq_len]
            labels: 标签用于计算loss [batch_size, seq_len] 或 [seq_len]，可选
        Returns:
            如果提供labels: 返回(logits, loss)
            如果不提供labels: 返回logits
        """
        # 确保输入是正确的形状
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
        
        batch_size, seq_len = input_ids.shape
        
        # 1. 词嵌入: [batch_size, 512] -> [batch_size, 512, 1024]
        embedded = self.text_embedding(input_ids)  # [batch_size, 512, 1024]
        
        # 处理每个样本
        logits_list = []
        losses = []
        
        for i in range(batch_size):
            x = embedded[i]  # [512, 1024]
            
            # 2. block_0: [512, 1024] -> [1024, 2048]
            x = self.rms_norm_layers[0](x)  # RMSNorm before block_0
            x = self.block_0(x)  # [1024, 2048]
            
            # 3. blocks_1-6: 残差连接处理 ([1024, 2048] -> [1024, 2048])
            x = self.rms_norm_layers[1](x)  # RMSNorm before blocks_1-6
            for j in range(6):
                residual = x.clone()
                x = self.blocks_1_to_16[j](x)
                x = x + residual  # 残差连接
                # 每个残差连接后添加RMSNorm
                if j < 5:  # blocks_1-5后添加（共5个）
                    x = self.rms_norm_layers[9+j](x)
            
            # 4. block_17: 记忆机制处理
            # block_17_1: 生成记忆矩阵 [1024, 2048] -> [4096, 4096]
            x_norm1 = self.rms_norm_layers[2](x)  # RMSNorm before block_17_1
            memory = self.block_17_1(x_norm1)
            
            # block_17_2: 遗忘处理 [1024, 2048] -> [1024, 2048]
            x_norm2 = self.rms_norm_layers[3](x)  # RMSNorm before block_17_2
            forget_output = self.block_17_2(x_norm2)
            x = x + forget_output  # 与原始矩阵相加
            
            # 5. blocks_7-12: 残差连接处理 (使用梯度检查点)
            x = self.rms_norm_layers[4](x)  # RMSNorm before blocks_7-12
            if self.use_gradient_checkpointing:
                x = checkpoint(self._checkpointed_blocks_7_to_12, x, use_reentrant=False)
            else:
                # 不使用梯度检查点的传统处理
                for j in range(6, 12):
                    residual = x.clone()
                    x = self.blocks_1_to_16[j](x)
                    x = x + residual  # 残差连接
            
            # 6. block_18: 记忆与x融合 [4096, 4096] -> [1024, 2048]
            memory_norm = self.rms_norm_layers[5](memory)  # RMSNorm before block_18
            memory_processed = self.block_18(memory_norm)
            x = x + memory_processed  # 与x相加
            
            # 7. blocks_13-16: 残差连接处理
            x = self.rms_norm_layers[6](x)  # RMSNorm before blocks_13-16
            for j in range(12, 16):
                residual = x.clone()
                x = self.blocks_1_to_16[j](x)
                x = x + residual  # 残差连接
                # blocks_13-15后添加RMSNorm（共3个）
                if j < 15:
                    x = self.rms_norm_layers[16+(j-12)](x)
            
            # 8. block_19: 输出层处理 [1024, 2048] -> [512, 1024]
            x = self.rms_norm_layers[7](x)  # RMSNorm before block_19
            x = self.block_19(x)  # [512, 1024]
            
            # 9. 与词嵌入层转置相乘得到logits [512, 1024] × [1024, vocab_size] -> [512, vocab_size]
            x = self.rms_norm_layers[8](x)  # RMSNorm before logits computation
            embedding_weight = self.text_embedding.weight  # [vocab_size, 1024]
            logits = torch.matmul(x, embedding_weight.transpose(0, 1))  # [512, vocab_size]
            logits_list.append(logits)
            
            # 如果提供了labels，计算loss
            if labels is not None:
                if labels.dim() == 1:
                    sample_labels = labels.unsqueeze(0)  # [1, seq_len]
                else:
                    sample_labels = labels[i]  # [seq_len]
                
                # 计算交叉熵损失
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
                # 检查标签有效性
                valid_labels = (sample_labels != -100) & (sample_labels >= 0) & (sample_labels < logits.size(-1))
                if valid_labels.any():
                    # 只计算有效标签的损失
                    valid_logits = logits[valid_labels]
                    valid_sample_labels = sample_labels[valid_labels]
                    if valid_logits.numel() > 0 and valid_sample_labels.numel() > 0:
                        loss = loss_fn(valid_logits, valid_sample_labels)
                        # 检查NaN
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"检测到无效损失值: {loss}")
                            loss = torch.tensor(10.0, device=logits.device, requires_grad=True)
                    else:
                        loss = torch.tensor(10.0, device=logits.device, requires_grad=True)
                else:
                    loss = torch.tensor(10.0, device=logits.device, requires_grad=True)
                losses.append(loss)
        
        # 合并结果
        final_logits = torch.stack(logits_list, dim=0)  # [batch_size, 512, vocab_size]
        
        if labels is not None:
            final_loss = torch.stack(losses).mean()  # 平均批次损失
            return final_logits, final_loss
        
        return final_logits

# ==================== 训练器模块 ====================
class Trainer:
    """模型训练器（带学习率预热和梯度截断）"""
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4, accumulation_steps: int = 4,
                 warmup_steps: int = 1000, max_grad_norm: float = 0.5):
        """
        初始化训练器
        Args:
            model: 要训练的模型
            learning_rate: 基础学习率
            accumulation_steps: 梯度累积步数，默认4步
            warmup_steps: 学习率预热步数，默认1000步
            max_grad_norm: 最大梯度范数，用于梯度截断，默认0.5
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.accumulation_steps = accumulation_steps
        self.accumulation_count = 0
        self.accumulated_loss = 0.0
        
        # 学习率预热参数
        self.warmup_steps = warmup_steps
        self.base_learning_rate = learning_rate
        self.current_step = 0
        
        # 梯度截断参数
        self.max_grad_norm = max_grad_norm
        
        # 使用PyTorch官方Muon优化器
        # 查找≥2D参数（隐藏层）用Muon优化
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        # 查找其他参数用AdamW优化
        adamw_params = [p for p in model.parameters() if p.ndim < 2]
        
        # Muon优化器参数设置（参考官方文档推荐值）
        self.optimizer = optim.Muon(
            muon_params, 
            lr=0.02,  # 学习率
            weight_decay=0.1,  # 权重衰减
            momentum=0.95,  # 动量因子
            nesterov=True,  # 启用Nesterov动量
            ns_steps=5,  # Newton-Schulz迭代步数
            adjust_lr_fn="match_rms_adamw"  # 匹配AdamW的RMS
        )
        
        # 辅助优化器用于其他参数
        self.aux_optimizer = optim.AdamW(
            adamw_params, 
            lr=3e-4, 
            betas=(0.90, 0.95), 
            weight_decay=0.01
        )
        
        self.optimizers = [self.optimizer, self.aux_optimizer]
        logger.info("使用PyTorch官方Muon优化器")
        
        # 使用自定义学习率调度（包含预热）
        self._setup_lr_scheduler()
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"学习率预热步数: {warmup_steps}")
        logger.info(f"最大梯度范数: {max_grad_norm}")
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        Args:
            logits: 模型输出 [batch_size, seq_len, vocab_size]
            labels: 真实标签 [batch_size, seq_len]
        Returns:
            损失值
        """
        # 检查NaN值
        if torch.isnan(logits).any():
            logger.warning("检测到logits中的NaN值")
            # 返回一个大的有限损失值而不是NaN
            return torch.tensor(float('inf'), device=logits.device, requires_grad=True)
        
        # 忽略padding位置的损失
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        # 重塑张量以适应CrossEntropyLoss
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        # 检查标签有效性
        valid_labels = (labels != -100) & (labels >= 0) & (labels < logits.size(-1))
        if not valid_labels.any():
            logger.warning("没有有效的标签用于计算损失")
            return torch.tensor(10.0, device=logits.device, requires_grad=True)
        
        loss = loss_fn(logits, labels)
        
        # 检查损失是否为NaN或无穷大
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"计算出无效损失值: {loss}")
            return torch.tensor(10.0, device=logits.device, requires_grad=True)
        
        return loss
    
    def _setup_lr_scheduler(self):
        """设置学习率调度器（包含预热）"""
        # 不使用传统的调度器，而是手动控制学习率
        pass
    
    def _get_warmup_lr(self) -> float:
        """计算当前步骤的学习率（包含预热）"""
        if self.current_step < self.warmup_steps:
            # 线性预热
            return self.base_learning_rate * (self.current_step / self.warmup_steps)
        else:
            # 预热后使用余弦衰减
            decay_steps = self.current_step - self.warmup_steps
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(decay_steps / 10000.0) * torch.pi))
            return max(self.base_learning_rate * cosine_decay, 1e-6)
    
    def _update_learning_rate(self):
        """更新所有优化器的学习率"""
        current_lr = self._get_warmup_lr()
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # 每100步记录一次学习率
        if self.current_step % 100 == 0:
            logger.info(f"步骤 {self.current_step}, 学习率: {current_lr:.6f}")
    
    def _clip_gradients(self):
        """梯度截断"""
        # 对所有参数进行梯度截断
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        
        # 检查是否有梯度爆炸
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > self.max_grad_norm * 2:  # 如果梯度范数过大
            logger.warning(f"梯度范数过大: {total_norm:.4f}, 已截断到 {self.max_grad_norm}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        单步训练（Mask LM模式）
        Args:
            batch: 批次数据 {'input_ids': [batch_size, seq_len]}
        Returns:
            当前步的损失值
        """
        # 移动数据到设备
        original_input_ids = batch['input_ids'].to(self.device)  # [batch_size, seq_len]
        batch_size, seq_len = original_input_ids.shape
        
        # 为每个样本创建mask输入和标签
        masked_inputs = []
        label_sequences = []
        
        # 直接在Trainer中实现mask处理逻辑
        for i in range(batch_size):
            input_ids = original_input_ids[i]
            device = input_ids.device
            seq_len = input_ids.shape[0]
            
            # 创建mask位置
            mask_positions = torch.rand(seq_len, device=device) < 0.1  # 10% mask概率
            
            # 80%替换为[MASK]
            mask_for_mask = torch.rand(seq_len, device=device) < 0.8
            mask_for_mask = mask_for_mask & mask_positions
            
            # 10%替换为随机token
            mask_for_random = torch.rand(seq_len, device=device) < 0.5  # 0.1 / (1-0.8) = 0.5
            mask_for_random = mask_for_random & mask_positions & (~mask_for_mask)
            
            # 复制原始输入
            masked_input = input_ids.clone()
            # 标签就是原始输入
            labels = input_ids.clone()
            
            # 应用mask策略
            masked_input[mask_for_mask] = 103  # [MASK] token ID
            
            # 随机替换
            if mask_for_random.any():
                random_tokens = torch.randint(0, self.model.vocab_size, (mask_for_random.sum().item(),), device=device)
                masked_input[mask_for_random] = random_tokens
            
            masked_inputs.append(masked_input)
            label_sequences.append(labels)
        
        # 堆叠为批次
        masked_batch = torch.stack(masked_inputs, dim=0)  # [batch_size, seq_len]
        labels_batch = torch.stack(label_sequences, dim=0)  # [batch_size, seq_len]
        
        # 重塑为文本块格式 [batch_size, seq_len] -> [batch_size, seq_len]
        # 注意：这里的格式已经是正确的
        text_blocks = masked_batch  # [batch_size, 512]
        labels_for_model = labels_batch  # [batch_size, 512]
        
        # 前向传播（同时计算logits和loss）
        logits, loss = self.model(text_blocks, labels_for_model)
        
        # 梯度累积处理
        # 除以累积步数来平均梯度
        loss_scaled = loss / self.accumulation_steps
        
        # 检查损失是否为NaN
        if torch.isnan(loss_scaled):
            logger.warning("检测到NaN损失，跳过此批次")
            return float('inf')
        
        loss_scaled.backward()
        
        self.accumulation_count += 1
        self.accumulated_loss += loss.item()
        
        # 每累积指定步数后更新参数
        if self.accumulation_count % self.accumulation_steps == 0:
            # 梯度截断
            self._clip_gradients()
            
            # 更新学习率
            self._update_learning_rate()
            
            # 更新参数
            for opt in self.optimizers:
                opt.step()
                opt.zero_grad()
            
            # 更新步骤计数
            self.current_step += 1
            
            # 重置累积计数
            self.accumulation_count = 0
            self.accumulated_loss = 0.0
        
        # 如果不是累积步的最后一步，不立即更新参数
        # 梯度会在计算图中累积，直到达到累积步数
        
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        训练一个epoch
        Args:
            dataloader: 数据加载器
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            if num_batches % 100 == 0:
                logger.info(f"批次 {num_batches}, 损失: {loss:.4f}")
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        评估模型
        Args:
            dataloader: 验证数据加载器
        Returns:
            验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids)
                loss = self.compute_loss(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')

# ==================== 有监督微调模块 ====================
class SupervisedFineTuner:
    """有监督微调类"""
    
    def __init__(self, model: nn.Module, instruction_template: str = "请回答以下问题：{}"):
        """
        初始化微调器
        Args:
            model: 预训练模型
            instruction_template: 指令模板
        """
        self.model = model
        self.instruction_template = instruction_template
    
    def format_instruction(self, instruction: str, input_text: str = "", 
                          output_text: str = "") -> str:
        """
        格式化指令数据
        Args:
            instruction: 指令
            input_text: 输入文本
            output_text: 输出文本
        Returns:
            格式化的文本
        """
        prompt = self.instruction_template.format(instruction)
        if input_text:
            prompt += f"\n输入：{input_text}"
        if output_text:
            prompt += f"\n输出：{output_text}"
        return prompt
    
    def fine_tune(self, train_data: List[Dict], val_data: List[Dict], 
                  epochs: int = 3, batch_size: int = 4):
        """
        执行有监督微调
        Args:
            train_data: 训练数据 [{'instruction': '', 'input': '', 'output': ''}]
            val_data: 验证数据
            epochs: 训练轮数
            batch_size: 批次大小
        """
        logger.info("开始有监督微调...")
        
        # 准备微调数据
        train_texts = [self.format_instruction(**item) for item in train_data]
        val_texts = [self.format_instruction(**item) for item in val_data]
        
        # 创建数据加载器
        processor = TextPreprocessor()
        train_loader = processor.create_text_dataloader(train_texts, batch_size=batch_size)
        val_loader = processor.create_text_dataloader(val_texts, batch_size=batch_size, shuffle=False)
        
        # 创建训练器
        trainer = Trainer(self.model)
        
        # 训练循环
        for epoch in range(epochs):
            logger.info(f"微调 Epoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss = trainer.train_epoch(train_loader)
            logger.info(f"训练损失: {train_loss:.4f}")
            
            # 验证
            val_loss = trainer.evaluate(val_loader)
            logger.info(f"验证损失: {val_loss:.4f}")

# ==================== 强化学习模块 ====================
class RLTrainer:
    """强化学习训练器（基于PPO）"""
    
    def __init__(self, model: nn.Module, reward_model: nn.Module):
        """
        初始化RL训练器
        Args:
            model: 策略模型
            reward_model: 奖励模型
        """
        self.model = model
        self.reward_model = reward_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # PPO超参数
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.lam = 0.95
    
    def compute_rewards(self, responses: List[str]) -> torch.Tensor:
        """
        计算奖励分数
        Args:
            responses: 模型生成的回复
        Returns:
            奖励张量
        """
        # 这里应该使用专门的奖励模型
        # 简化实现：随机奖励作为示例
        rewards = torch.randn(len(responses)).to(self.device)
        return rewards
    
    def ppo_step(self, prompts: List[str], responses: List[str], 
                 old_logprobs: torch.Tensor, advantages: torch.Tensor) -> float:
        """
        PPO更新步骤
        Args:
            prompts: 提示文本
            responses: 回复文本
            old_logprobs: 旧的对数概率
            advantages: 优势函数值
        Returns:
            PPO损失
        """
        # 重新计算新的对数概率
        # 这里需要实现具体的PPO算法逻辑
        
        # 简化的PPO损失计算
        ratio = torch.exp(torch.randn_like(old_logprobs))  # 新旧概率比
        clipped_ratio = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        return policy_loss.item()

# ==================== GPU工具函数 ====================
def get_device():
    """获取训练设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.warning("使用CPU训练")
    return device

# ==================== 主程序入口 ====================
def main():
    """主函数 - 完整的预训练流程"""
    
    # 1. 设备设置
    logger.info("=== 设备设置 ===")
    device = get_device()
    
    # 2. 配置设置
    logger.info("=== 预训练配置 ===")
    vocab_size = 21128
    text_seq_len = 512
    batch_size = 1
    chunk_size = 512
    epochs = 10
    learning_rate = 1e-4  # 降低学习率避免NaN
    use_gradient_checkpointing = True
    
    logger.info(f"词汇表大小: {vocab_size}")
    logger.info(f"序列长度: {text_seq_len}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"梯度检查点: {'启用' if use_gradient_checkpointing else '禁用'}")
    
    # 3. 模型初始化
    logger.info("=== 步骤1: 模型初始化 ===")
    model = MainModel(
        vocab_size=vocab_size, 
        text_seq_len=text_seq_len,
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    model = model.to(device)
    logger.info(f"模型已移动到设备: {device}")
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 数据准备
    logger.info("=== 步骤2: 数据准备 ===")
    processor = TextPreprocessor()
    
    # 加载训练数据
    data_path = "D:/Download/000_00001.jsonl"
    train_texts = []
    try:
        train_texts = processor.load_jsonl_data(data_path)
        logger.info(f"成功加载 {len(train_texts)} 条训练文本")
    except Exception as e:
        logger.warning(f"无法加载训练数据: {e}")
        train_texts = ["这是示例文本"] * 100
        logger.info("使用示例数据进行测试")
    
    # 创建数据加载器
    train_loader = processor.create_text_dataloader(
        train_texts, 
        batch_size=batch_size, 
        chunk_size=chunk_size
    )
    
    # 5. 训练器初始化
    logger.info("=== 步骤3: 训练器初始化 ===")
    trainer = Trainer(model, learning_rate=learning_rate)
    trainer.device = device  # 确保使用正确的设备
    
    # 6. 预训练循环
    logger.info("=== 步骤4: 开始预训练 ===")
    best_loss = float('inf')
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        # 训练
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                loss = trainer.train_step(batch)
                total_loss += loss
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    avg_loss = total_loss / num_batches
                    # 计算实际处理的文本块数量（batch_idx + 1是因为索引从0开始）
                    text_blocks_processed = (batch_idx + 1) * batch_size
                    logger.info(f"  Batch {batch_idx} (已处理{text_blocks_processed}个512大小文本块), Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}")
                    
                # 显存清理
                if batch_idx % 50 == 0:
                    text_blocks_processed = (batch_idx + 1) * batch_size
                    logger.info(f"  清理显存 (已处理{text_blocks_processed}个文本块)")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    text_blocks_processed = batch_idx * batch_size
                    logger.warning(f"显存不足，跳过batch {batch_idx} (已处理{text_blocks_processed}个文本块): {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except Exception as e:
                logger.error(f"Batch {batch_idx} 训练出错: {e}")
                continue
        
        # 计算epoch平均损失
        epoch_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        logger.info(f"Epoch {epoch + 1} 完成, 平均损失: {epoch_loss:.4f}")
        
        # 保存检查点
        model_config = ModelConfig(vocab_size, text_seq_len, use_gradient_checkpointing)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            logger.info(f"新最佳损失: {best_loss:.4f}")
            
            # 保存最佳模型
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'aux_optimizer_state_dict': trainer.aux_optimizer.state_dict(),
                'best_loss': best_loss,
                'config': model_config.to_dict(),
                'training_info': {
                    'epochs_trained': epoch + 1,
                    'training_type': 'mask_lm_pretrain'
                }
            }
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"保存最佳模型检查点: {checkpoint_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'aux_optimizer_state_dict': trainer.aux_optimizer.state_dict(),
                'current_loss': epoch_loss,
                'config': model_config.to_dict()
            }
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"保存定期检查点: {checkpoint_path}")
        
        # 重置数据加载器
        train_loader = processor.create_text_dataloader(
            train_texts, 
            batch_size=batch_size, 
            chunk_size=chunk_size
        )
    
    logger.info("预训练完成！")
    
    # 7. 模型测试
    logger.info("=== 步骤5: 模型测试 ===")
    model.eval()
    with torch.no_grad():
        # 使用第一条文本进行测试
        test_text = train_texts[0] if train_texts else "测试文本"
        test_text_blocks = processor.get_text_blocks_tensor([test_text], chunk_size=chunk_size)
        test_text_blocks = test_text_blocks.to(device)
        
        # 推理
        logits = model(test_text_blocks)
        logger.info(f"推理输出形状: {logits.shape}")
        
        # 生成一些预测
        if logits.dim() == 3:  # [batch_size, seq_len, vocab_size]
            predictions = torch.argmax(logits[0, :10], dim=-1)  # 第一个样本的前10个token预测
        else:  # [seq_len, vocab_size]
            predictions = torch.argmax(logits[:10], dim=-1)  # 前10个token预测
        logger.info(f"前10个预测token: {predictions.tolist()}")
    
    # 7. 最终模型保存
    logger.info("=== 步骤6: 保存最终模型 ===")
    final_model_path = "alice_final_pretrained.pth"
    model_config = ModelConfig(vocab_size, text_seq_len, use_gradient_checkpointing)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_name': processor.tokenizer.name_or_path,
        'config': model_config.to_dict(),
        'training_info': {
            'total_epochs': epochs,
            'best_loss': best_loss,
            'training_type': 'mask_lm_pretrain_complete'
        }
    }, final_model_path)
    
    logger.info(f"最终预训练模型已保存到 {final_model_path}")
    logger.info("预训练流程完成！")

if __name__ == "__main__":
    main()