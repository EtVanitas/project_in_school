"""Alice 模型预训练主文件（异步流水线版本）

特性:
- MLM 掩码语言建模损失
- MUON 优化器（2D 参数）+ AdamW（bias）
- Remain 轨迹自动记录与保存
- 混合精度训练 (AMP) 节省显存
- 异步流水线加速（GPU/CPU 并行）

使用方式:
    # 同步版本（默认）
    python train_pretrain.py
    
    # 异步版本（性能优化）
    python train_pretrain_async.py
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List
from torch.cuda.amp import autocast, GradScaler

from config import Config

# 异步版本组件
from alice_main_async import MainModelAsync
from text_embedding_lite import TextEmbeddingProcessor, CustomEmbedding


class PretrainTrainerAsync:
    """预训练管理器（异步版本）"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 词嵌入层
        self.embedding = CustomEmbedding(
            vocab_size=21128,
            embedding_dim=Config.model.M
        ).to(self.device)
        
        # Alice 主体模型（异步版本）
        self.model = MainModelAsync()
        
        # 输出投影层
        self.output_projection = nn.Linear(Config.model.M, 21128).to(self.device)
        
        # 初始化优化器
        self._init_optimizers()
        
        # MLM 损失函数
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 混合精度训练 scaler
        self.use_mixed_precision = Config.flow.ENABLE_MIXED_PRECISION
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            print("[配置] 已启用混合精度训练")
        else:
            self.scaler = None
            if self.use_mixed_precision:
                print("[警告] 混合精度训练需要 GPU，已自动禁用")
        
        # 统计信息
        self.stats = {
            'total_steps': 0,
            'total_loss': 0.0,
            'mlm_loss': 0.0,
        }
        
        # 打印配置
        total_params = sum(p.numel() for p in self.model.parameters()) + \
                      sum(p.numel() for p in self.embedding.parameters()) + \
                      sum(p.numel() for p in self.output_projection.parameters())
        print(f"总参数量：{total_params:,}")
        print(f"设备：{self.device}")
    
    def _init_optimizers(self):
        """初始化双优化器（Muon + AdamW）"""
        from torch.optim import Muon
        
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
        
        # 检查 output_projection 的参数是否在列表中
        proj_weight_included = any(p is self.output_projection.weight for p in params_2d)
        proj_bias_included = any(p is self.output_projection.bias for p in params_1d) if hasattr(self.output_projection, 'bias') else False
        
        print(f"Projection weight 在优化器中：{proj_weight_included}")
        print(f"Projection bias 在优化器中：{proj_bias_included}")
        
        # Muon 优化 2D 参数（权重矩阵）
        self.optimizer = Muon(params_2d, lr=0.01, momentum=0.95, weight_decay=0.1)
        
        # AdamW 优化 1D 参数（bias）
        self.bias_optimizer = torch.optim.AdamW(params_1d, lr=0.001, weight_decay=0.1) if params_1d else None
        
        print(f"优化器：Muon ({len(params_2d)} 个 2D 参数) + AdamW ({len(params_1d)} 个 bias)")
    
    def compute_mlm_loss(self, output_lists: List[torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, Dict]:
        """计算 MLM 损失"""
        all_logits = []
        all_labels = []
        
        for output in output_lists:
            # 投影到词表空间：(m, n) → (n, vocab_size)
            output_on_device = output.to(self.device)
            projected = self.output_projection(output_on_device)
            
            all_logits.append(projected.view(-1, projected.size(-1)))
            
            label_tensor = labels[0].view(-1)
            all_labels.append(label_tensor)
        
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
        
        for batch_idx, batch_dict in enumerate(progress_bar):
            embedded = batch_dict['embedded'].to(self.device)
            labels = batch_dict['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            if self.bias_optimizer is not None:
                self.bias_optimizer.zero_grad()
            
            try:
                # 混合精度前向传播
                if self.use_mixed_precision and self.scaler is not None:
                    with autocast():
                        output_lists = self.model(embedded, epoch=epoch)
                else:
                    output_lists = self.model(embedded, epoch=epoch)
                
                # 检查是否有输出
                if len(output_lists) == 0:
                    print(f"[警告] Batch {batch_idx}: 无 Type3 Reason 输出，跳过")
                    continue
                
                mlm_loss, loss_dict = self.compute_mlm_loss(output_lists, labels)
                
                # 反向传播（混合精度）
                if self.use_mixed_precision and self.scaler is not None:
                    # ✅ 正确使用 scaler.scale(loss)
                    scaled_loss = self.scaler.scale(mlm_loss)
                    scaled_loss.backward()
                    
                    # ✅ unscale_() 还原梯度用于裁剪（必须在 step 之前）
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.optimizer.MAX_GRAD_NORM)
                    
                    if self.bias_optimizer is not None:
                        self.scaler.unscale_(self.bias_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for group in self.bias_optimizer.param_groups for p in group['params']], 
                            Config.optimizer.MAX_GRAD_NORM
                        )
                    
                    # ✅ step() 更新参数
                    self.scaler.step(self.optimizer)
                    if self.bias_optimizer is not None:
                        self.scaler.step(self.bias_optimizer)
                    
                    # ✅ update() 更新缩放因子
                    self.scaler.update()
                else:
                    # 普通精度反向传播
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
                
                self.stats['total_steps'] += 1
                self.stats['total_loss'] += batch_loss
                self.stats['mlm_loss'] += loss_dict.get('mlm_loss', 0.0)
                
                # 清理
                del output_lists, mlm_loss, loss_dict, embedded, labels
                
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
                print(f"  embedded: shape={embedded.shape}, requires_grad={embedded.requires_grad}")
                print(f"  labels: shape={labels.shape}, requires_grad={labels.requires_grad}")
                if len(output_lists) > 0:
                    print(f"  output_lists[0]: shape={output_lists[0].shape}, requires_grad={output_lists[0].requires_grad}")
                import traceback
                traceback.print_exc()
                # 异常时也要清理
                del embedded, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"\nEpoch {epoch} 完成：平均损失 {avg_loss:.4f}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[显存清理] 当前已用：{torch.cuda.memory_allocated(self.device)/1024**3:.2f} GB")
        
        return avg_loss, True
    
    def train(self, train_dataloader, num_epochs=1):
        """完整训练流程"""
        print(f"开始训练，共 {num_epochs} 个 epoch...")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs}\n{'='*60}\n")
            
            train_loss, success = self.train_epoch(train_dataloader, epoch)
            
            if not success:
                print(f"\nEpoch {epoch} 因显存过高提前终止")
                break
        
        print(f"\n{'='*60}\n训练完成！最终损失：{self.stats['total_loss']/max(1, self.stats['total_steps']):.4f}\n{'='*60}")
    
    def reset(self):
        """重置模型状态"""
        if hasattr(self.model, 'remain_manager'):
            self.model.remain_manager.clear()
        if hasattr(self.model, 'long_term_memory'):
            self.model.long_term_memory.wait_all_prefetch()
        
        self.stats = {'total_steps': 0, 'total_loss': 0.0, 'mlm_loss': 0.0}
        print("已重置模型状态")
    
    def shutdown(self):
        """关闭异步组件"""
        if hasattr(self.model, 'shutdown'):
            self.model.shutdown()


def create_train_dataloader(file_paths: List[str], batch_size: int = 2, max_chunks: int = None):
    """创建训练数据加载器"""
    return TextEmbeddingProcessor(
        file_paths=file_paths,
        chunk_size=1024,
        overlap=128,
        batch_size=batch_size,
        num_workers=2,
        max_chunks=max_chunks
    )


def main():
    """主函数"""
    print("="*60)
    print("Alice 预训练系统 v9.0（异步流水线版）")
    print("="*60)
    
    # 打印优化配置
    print(f"\n优化配置:")
    print(f"  梯度检查点：{Config.flow.ENABLE_GRADIENT_CHECKPOINTING} (已禁用：树状剪枝)")
    print(f"  混合精度训练：{Config.flow.ENABLE_MIXED_PRECISION}")
    print(f"  异步流水线：True")
    print(f"  最大迭代次数：{Config.flow.MAX_ITERATIONS}")
    print(f"  激活池容量：{Config.flow.ACTIVATED_POOL_MAX_SIZE}")
    print()
    
    trainer = PretrainTrainerAsync()
    
    # 准备数据
    train_files = ['test_data/test_0.txt', 'test_data/test_1.txt']
    train_loader = create_train_dataloader(file_paths=train_files, batch_size=2)
    
    print(f"\n训练集：{len(train_files)} 个文件")
    print(f"批次大小：2")
    
    try:
        # 开始训练
        trainer.train(train_dataloader=train_loader, num_epochs=1)
        print("\n预训练系统训练完成！")
    finally:
        # 确保异步组件正确关闭
        trainer.shutdown()


if __name__ == "__main__":
    main()
