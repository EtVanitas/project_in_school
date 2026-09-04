"""
Alice 模型预训练主文件

基于向量化 MainModel + MUON 优化器 + Remain 记忆机制

特性:
1. 支持多 label 批处理输入
2. MLM 掩码语言建模损失
3. PyTorch 官方 MUON 优化器
4. Remain 轨迹自动记录与保存
5. 小批次训练（batch_size=2）避免显存溢出
6. CPU/GPU 自动降级支持
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List

# 导入配置
from config import Config

# 导入模型
from alice_main import MainModel
from text_embedding_lite import TextEmbeddingProcessor, CustomEmbedding


class PretrainTrainer:
    
    def __init__(self, config=Config):
        self.config = config
        # 设备管理：支持 CPU/GPU 自动降级
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("初始化 Alice 预训练模型...")
        
        # 词嵌入层
        self.embedding = CustomEmbedding(
            vocab_size=21128,
            embedding_dim=config.model.M
        ).to(self.device)
        
        # Alice 主体模型（设备由 MainModel 内部管理）
        self.model = MainModel()
        
        # 输出投影层 Linear(m, vocab_size)
        self.output_projection = nn.Linear(
            config.model.M, 
            21128  # BERT-base-chinese 词表大小
        ).to(self.device)
        
        # MUON 优化器（PyTorch 官方实现）
        # Muon 只支持 2D 参数，需要排除 bias（1D 参数）
        from torch.optim import Muon
        
        # 收集所有 2D 参数（权重矩阵）
        params_2d = []
        for param in list(self.model.parameters()) + list(self.embedding.parameters()) + list(self.output_projection.parameters()):
            if param.dim() == 2:
                params_2d.append(param)
        
        # 收集所有 1D 参数（bias）
        params_1d = []
        for param in list(self.model.parameters()) + list(self.embedding.parameters()) + list(self.output_projection.parameters()):
            if param.dim() == 1:
                params_1d.append(param)
        
        # 使用 Muon 优化 2D 参数
        self.optimizer = Muon(
            params_2d,
            lr=0.01,
            momentum=0.95,
            weight_decay=0.1
        )
        
        # 使用 AdamW 优化 bias 参数
        self.bias_optimizer = torch.optim.AdamW(
            params_1d,
            lr=0.001,
            weight_decay=0.1
        ) if params_1d else None
        
        print(f"使用 PyTorch 官方 MUON 优化器（2D 参数）+ AdamW（{len(params_1d) if params_1d else 0} 个 bias 参数）")
        
        # MLM 损失函数
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 统计信息
        self.stats = {
            'total_steps': 0,
            'total_loss': 0.0,
            'mlm_loss': 0.0,
            'best_val_loss': float('inf'),
        }
        
        total_params = (
            sum(p.numel() for p in self.model.parameters()) +
            sum(p.numel() for p in self.embedding.parameters()) +
            sum(p.numel() for p in self.output_projection.parameters())
        )
        print(f"总参数量：{total_params:,}")
        print(f"优化器：{type(self.optimizer).__name__} (lr={self.optimizer.defaults['lr']})")
        print(f"设备：{self.device}")
    
    def compute_mlm_loss(self, output_lists: Dict[int, List[torch.Tensor]], 
                         labels: torch.Tensor) -> tuple[torch.Tensor, Dict]:
        """
        计算 MLM 损失
        
        Args:
            output_lists: {label: [out1, out2, ...]} 
                          每个 out: (m, n)
            labels: (batch, seq_len) 真实标签，seq_len = n
            
        Returns:
            total_loss, loss_dict
        """
        batch_size = labels.shape[0]
        seq_len = labels.shape[1]  # 应该等于 n=1024
        
        all_logits = []
        all_labels = []
        
        # 处理每个 label 的输出
        for label_idx in sorted(output_lists.keys()):
            outputs_for_label = output_lists[label_idx]
            
            if len(outputs_for_label) == 0:
                continue
            
            # 批处理：堆叠所有输出
            # outputs_for_label: [(m, n), (m, n), ...]
            # stacked: (num_outputs, m, n)
            stacked = torch.stack(outputs_for_label, dim=0).to(self.device)
            num_outputs = stacked.size(0)
            
            # 投影到词表空间
            # Linear(m, vocab_size): (num_outputs, m, n) → (num_outputs, n, vocab_size)
            projected = self.output_projection(stacked)
            
            # 调整维度用于计算损失
            # projected: (num_outputs, n, vocab_size)
            # 展平为 (num_outputs * n, vocab_size)
            all_logits.append(projected.view(-1, projected.size(-1)))
            
            # 准备对应的 labels
            if num_outputs == batch_size:
                # 完美匹配：每个输出对应一个样本
                all_labels.append(labels.view(-1))
            else:
                # 不匹配：使用第一个样本的 labels（临时方案）
                first_label = labels[0:1]  # (1, seq_len)
                expanded = first_label.repeat(num_outputs, 1)  # (num_outputs, seq_len)
                all_labels.append(expanded.view(-1))
            
            # 及时清理临时变量
            del stacked, projected
        
        # 合并所有 logits 和 labels
        if len(all_logits) == 0:
            return torch.tensor(0.0, device=self.device), {}
        
        combined_logits = torch.cat(all_logits, dim=0)  # (total_tokens, vocab_size)
        combined_labels = torch.cat(all_labels, dim=0)   # (total_tokens,)
        
        # 计算交叉熵损失
        mlm_loss = self.mlm_criterion(combined_logits, combined_labels)
        
        loss_dict = {
            'mlm_loss': mlm_loss.item(),
            'total_tokens': combined_logits.shape[0],
        }
        
        # 清理临时变量
        del all_logits, all_labels, combined_logits, combined_labels
        
        return mlm_loss, loss_dict
    
    def train_epoch(self, dataloader, epoch: int) -> tuple[float, bool]:
        """训练一个 epoch
        
        Args:
            dataloader: 数据加载器，返回 batch_dict
                        batch_dict 包含:
                        - 'embedded': (batch, m, n) 嵌入后的输入
                        - 'masked_input': (batch, n) MLM 掩码后的输入 IDs
                        - 'labels': (batch, n) 真实标签
                        - 'token_ids': (batch, n) 原始 token IDs
            epoch: 当前 epoch 编号
        
        Returns:
            avg_loss: 平均损失
            success: 是否成功完成（False 表示显存过高退出）
        """
        self.model.train()
        self.embedding.train()
        self.output_projection.train()
        
        epoch_loss = 0.0
        epoch_mlm_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch_dict in enumerate(progress_bar):
            # ========== 数据准备 ==========
            embedded = batch_dict['embedded'].to(self.device)      # (batch, m, n)
            labels = batch_dict['labels'].to(self.device)          # (batch, n)
            
            # 前向传播
            self.optimizer.zero_grad()
            if self.bias_optimizer is not None:
                self.bias_optimizer.zero_grad()
            
            try:
                output_lists, stats = self.model(embedded, epoch=epoch)
                
                # 计算损失
                mlm_loss, loss_dict = self.compute_mlm_loss(output_lists, labels)
                
                # 反向传播
                mlm_loss.backward()
                
                # 梯度裁剪（仅 MainModel）
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    Config.optimizer.MAX_GRAD_NORM
                )
                
                # 更新参数（双优化器）
                self.optimizer.step()
                if self.bias_optimizer is not None:
                    self.bias_optimizer.step()
                
                # 统计信息
                batch_loss = mlm_loss.item()
                epoch_loss += batch_loss
                epoch_mlm_loss += loss_dict.get('mlm_loss', 0.0)
                num_batches += 1
                
                # 进度条更新
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{epoch_loss / max(1, num_batches):.4f}',
                })
                
                self.stats['total_steps'] += 1
                self.stats['total_loss'] += batch_loss
                self.stats['mlm_loss'] += loss_dict.get('mlm_loss', 0.0)
                
                # ========== 显存清理（关键修复）==========
                # 1. 释放大对象
                del output_lists, stats, mlm_loss, loss_dict
                del embedded, labels
                
                # 2. 清理 CUDA 缓存（GPU 环境）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                    # 3. 监控显存
                    memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                    memory_max = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                    memory_usage = memory_allocated / memory_max * 100
                    
                    if memory_usage > 90:
                        print(f"\n警告：显存占用过高 ({memory_usage:.1f}%), 自动退出")
                        print(f"   已用：{memory_allocated:.2f} GB / {memory_max:.2f} GB")
                        return epoch_loss / max(1, num_batches), False
            
            except Exception as e:
                print(f"[Error] Batch {batch_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                # 异常时也要清理
                del embedded, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # Epoch 统计
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"\nEpoch {epoch} 完成:")
        print(f"  平均损失：{avg_loss:.4f}")
        print(f"  总步数：{self.stats['total_steps']}")
        print(f"  累计损失：{self.stats['total_loss']:.4f}")
        
        # 显存清理（GPU 环境）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            print(f"\n[显存清理] 当前已用：{allocated:.2f} GB")
        
        return avg_loss, True
    
    def train(self, train_dataloader, num_epochs=None):
        """完整训练流程
        
        Args:
            train_dataloader: 训练数据加载器
            num_epochs: 训练轮数
        """
        if num_epochs is None:
            num_epochs = self.config.train.NUM_EPOCHS
        
        start_epoch = 0
        
        print(f"开始训练，共 {num_epochs} 个 epoch...")
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}\n")
            
            # 训练
            train_loss, success = self.train_epoch(
                train_dataloader, 
                epoch
            )
            
            # 检查是否因显存过高退出
            if not success:
                print(f"\nEpoch {epoch} 因显存过高提前终止")
                print("建议：减少 batch_size 或 max_chunks 后重新训练")
                break
        
        print("\n" + "="*60)
        print("训练完成！")
        print(f"最终训练损失：{self.stats['total_loss'] / max(1, self.stats['total_steps']):.4f}")
        print("="*60)
    
    def reset(self):
        """重置模型状态（用于重新训练）"""
        # 重置 RemainManager
        if hasattr(self.model, 'remain_manager'):
            self.model.remain_manager.clear()
        
        # 重置统计信息
        self.stats = {
            'total_steps': 0,
            'total_loss': 0.0,
            'mlm_loss': 0.0,
        }
        
        print("已重置模型状态")


def create_train_dataloader(file_paths: List[str], batch_size: int = 2, 
                           max_chunks: int = None):
    """
    创建训练数据加载器（适配 text_embedding_lite 接口）
    
    Args:
        file_paths: 文本文件路径列表
        batch_size: 批次大小（推荐 2，避免显存溢出）
        max_chunks: 最大 chunks 数（None 表示全部加载）
        
    Returns:
        TextEmbeddingProcessor 实例
    """
    processor = TextEmbeddingProcessor(
        file_paths=file_paths,
        chunk_size=1024,
        overlap=128,
        batch_size=batch_size,
        num_workers=2,
        max_chunks=max_chunks
    )
    
    return processor


def main():
    """主函数 - 快速开始训练"""
    print("="*60)
    print("Alice 预训练系统 v9.0")
    print("="*60)
    
    # 初始化配置
    config = Config()
    
    # 创建训练器
    trainer = PretrainTrainer(config)
    
    # 准备数据
    print("\n准备数据...")
    
    # 训练数据
    train_files = ['test_data/test_0.txt', 'test_data/test_1.txt']
    train_loader = create_train_dataloader(
        file_paths=train_files,
        batch_size=2,
        max_chunks=None
    )
    
    print(f"训练集：{len(train_files)} 个文件")
    print(f"批次大小：2（小批次训练，避免显存溢出）")
    
    # 开始训练
    print("\n开始训练...\n")
    
    trainer.train(
        train_dataloader=train_loader,
        num_epochs=config.train.NUM_EPOCHS
    )
    
    print("\n预训练系统训练完成！")


if __name__ == "__main__":
    main()
