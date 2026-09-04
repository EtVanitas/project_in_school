"""
Alice 模型指令微调主文件

特性:
- 多轮自回归生成训练
- Teacher Forcing（使用真实答案作为上下文）
- 逐块预测答案
- 只计算答案块的损失
注意：需要预训练权重作为初始化
"""

import traceback

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List
import json
from pathlib import Path
from torch.optim import Muon

from config import Config
from alice_main import MainModel
from text_pretrain_data import CustomEmbedding
from finetune_data import InstructionTuningDataLoader


class InstructionFinetuneTrainer:
    """指令微调训练器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 词嵌入层
        self.embedding = CustomEmbedding(
            vocab_size=21128,
            embedding_dim=Config.model.M
        ).to(self.device)
        
        # Alice 主体模型
        self.model = MainModel()
        
        # 输出投影层
        self.output_projection = nn.Linear(Config.model.M, 21128).to(self.device)
        
        # 初始化优化器
        self._init_optimizers()
        
        # MLM 损失函数
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 统计信息
        self.stats = {
            'total_steps': 0,
            'total_loss': 0.0,
        }
        
        # 打印配置
        total_params = sum(p.numel() for p in self.model.parameters()) + \
                      sum(p.numel() for p in self.embedding.parameters()) + \
                      sum(p.numel() for p in self.output_projection.parameters())
        print(f"总参数量：{total_params:,}")
        print(f"设备：{self.device}")
    
    def _init_optimizers(self):
        """初始化双优化器（Muon + AdamW）"""
        # 确保所有参数都需要梯度
        model_params = list(self.model.parameters())
        embedding_params = list(self.embedding.parameters())
        projection_params = list(self.output_projection.parameters())
        
        print(f"Model 参数：{len(model_params)}")
        print(f"Embedding 参数：{len(embedding_params)}")
        print(f"Projection 参数：{len(projection_params)}")
        
        for param in model_params + embedding_params + projection_params:
            param.requires_grad = True
        
        # 收集所有参数
        all_params = model_params + embedding_params + projection_params
        
        params_2d = [p for p in all_params if p.dim() == 2]
        params_1d = [p for p in all_params if p.dim() == 1]
        
        print(f"收集到 {len(params_2d)} 个 2D 参数，{len(params_1d)} 个 1D 参数")
        
        # Muon 优化 2D 参数（权重矩阵）
        self.optimizer = Muon(params_2d, lr=Config.optimizer.LEARNING_RATE, momentum=0.95, weight_decay=Config.optimizer.WEIGHT_DECAY)
        
        # AdamW 优化 1D 参数（bias）
        self.bias_optimizer = torch.optim.AdamW(params_1d, lr=Config.optimizer.LEARNING_RATE_BIAS, weight_decay=Config.optimizer.WEIGHT_DECAY) if params_1d else None
        
        print(f"优化器：Muon ({len(params_2d)} 个 2D 参数) + AdamW ({len(params_1d)} 个 bias)")
    
    def compute_mlm_loss(self, output_lists: List[torch.Tensor], labels_list: List[torch.Tensor]) -> tuple[torch.Tensor, Dict]:
        """
        计算 MLM 损失（支持多块）
        
        Args:
            output_lists: List[Tensor[m, n]] - 所有 Type3 Reason 的输出
            labels_list: List[Tensor] - 每块的标签
        
        Returns:
            mlm_loss, loss_dict
        """
        all_logits = []
        all_labels = []
        
        # 只计算答案块的 loss（问题块的 labels 都是 -100）
        for output, labels in zip(output_lists, labels_list):
            # 检查这个块是否有有效标签
            if (labels != -100).sum() == 0:
                continue  # 跳过问题块
            
            # 投影到词表空间
            projected = self.output_projection(output.to(self.device))
            
            all_logits.append(projected.view(-1, projected.size(-1)))
            all_labels.append(labels.view(-1).to(self.device))
        
        if not all_logits:
            return torch.tensor(0.0, device=self.device), {}
        
        combined_logits = torch.cat(all_logits, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        
        mlm_loss = self.mlm_criterion(combined_logits, combined_labels)
        
        loss_dict = {'mlm_loss': mlm_loss.item(), 'total_tokens': combined_logits.shape[0]}
        
        del all_logits, all_labels, combined_logits, combined_labels
        return mlm_loss, loss_dict
    
    def train_epoch(self, dataloader, epoch: int) -> tuple[float, bool]:
        """训练一个 epoch"""
        self.model.train()
        self.embedding.train()
        self.output_projection.train()
        
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        # 显存监控
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            print(f"\n[显存] Epoch {epoch} 开始：{initial_memory:.2f} GB")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # batch_data 包含：
            # - chunks: List[Tensor] - 所有文本块
            # - labels: List[Tensor] - 每块的标签
            # - pred_chunk_idx: int - 当前要预测的块索引
            
            chunks = batch_data['chunks']
            labels = batch_data['labels']
            
            # 将所有块拼成完整的序列
            all_tokens = torch.cat(chunks, dim=0).unsqueeze(0).to(self.device)
            embedded = self.embedding(all_tokens)
            
            self.optimizer.zero_grad()
            if self.bias_optimizer is not None:
                self.bias_optimizer.zero_grad()
            
            try:
                # 前向传播
                output_lists = self.model(embedded, epoch=epoch)
                
                # 检查是否有输出
                if len(output_lists) == 0:
                    print(f"[警告] Batch {batch_idx}: 无 Type3 Reason 输出，跳过")
                    continue
                
                # 计算损失（只计算答案块）
                mlm_loss, loss_dict = self.compute_mlm_loss(output_lists, labels)
                
                # 反向传播
                mlm_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.optimizer.MAX_GRAD_NORM)
                
                # 更新参数
                self.optimizer.step()
                if self.bias_optimizer is not None:
                    self.bias_optimizer.step()
                
                # 统计
                batch_loss = mlm_loss.item()
                epoch_loss += batch_loss
                num_batches += 1
                
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{epoch_loss / max(1, num_batches):.4f}',
                })
                
                # 每 10 个 batch 打印一次详细统计
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(f"  [Batch {batch_idx+1}] loss={batch_loss:.4f}, avg_loss={epoch_loss/max(1,num_batches):.4f}")
                
                self.stats['total_steps'] += 1
                self.stats['total_loss'] += batch_loss
                
                # 清理
                del output_lists, mlm_loss, loss_dict, embedded, all_tokens
                
                # GPU 显存管理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_usage = torch.cuda.memory_allocated(self.device) / \
                                  torch.cuda.get_device_properties(self.device).total_memory * 100
                    
                    if memory_usage > 90:
                        print(f"\n警告：显存占用过高 ({memory_usage:.1f}%), 自动退出")
                        return epoch_loss / max(1, num_batches), False
            
            except Exception as e:
                print(f"\n[Error] Batch {batch_idx} failed: {e}")
                print(f"  错误类型：{type(e).__name__}")
                traceback.print_exc()
                # 异常时也要清理
                del embedded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"\nEpoch {epoch} 完成：平均损失 {avg_loss:.4f}")
        
        # 打印显存使用情况
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3
            print(f"\n[显存统计] Epoch {epoch}:")
            print(f"  初始：{initial_memory:.2f} GB")
            print(f"  最终：{final_memory:.2f} GB")
            print(f"  峰值：{peak_memory:.2f} GB")
            print(f"  净变化：{final_memory - initial_memory:+.2f} GB")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[显存清理] 当前已用：{torch.cuda.memory_allocated(self.device)/1024**3:.2f} GB")
        
        return avg_loss, True
    
    def train(self, dataloader, num_epochs=5):
        """完整训练流程"""
        print(f"开始指令微调训练，共 {num_epochs} 个 epoch...")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs}\n{'='*60}\n")
            
            train_loss, success = self.train_epoch(dataloader, epoch)
            
            if not success:
                print(f"\nEpoch {epoch} 因显存过高提前终止")
                break
        
        print(f"\n{'='*60}\n训练完成！最终损失：{self.stats['total_loss']/max(1, self.stats['total_steps']):.4f}\n{'='*60}")
    
    def reset(self):
        """重置模型状态"""
        if hasattr(self.model, 'remain_manager'):
            self.model.remain_manager.clear()
        
        self.stats = {'total_steps': 0, 'total_loss': 0.0, 'mlm_loss': 0.0}
        print("已重置模型状态")


def create_finetune_dataloader(file_paths: List[str], batch_size: int = None):
    """创建指令微调数据加载器"""
    if batch_size is None:
        batch_size = Config.data.BATCH_SIZE
    
    return InstructionTuningDataLoader(
        file_paths=file_paths,
        batch_size=batch_size,
        chunk_size=Config.data.CHUNK_SIZE,
        max_question_chunks=5,
        max_answer_chunks=10
    )


def main():
    """主函数"""
    print("="*60)
    print("Alice 指令微调系统 v9.0")
    print("="*60)
    
    trainer = InstructionFinetuneTrainer()
    
    # 准备数据
    train_files = ['test_data/instructions.jsonl']
    train_loader = create_finetune_dataloader(file_paths=train_files, batch_size=2)
    
    print(f"\n训练集：{len(train_files)} 个文件")
    print(f"批次大小：2")
    
    # 开始训练
    trainer.train(dataloader=train_loader, num_epochs=5)
    print("\n指令微调训练完成！")


if __name__ == "__main__":
    main()
