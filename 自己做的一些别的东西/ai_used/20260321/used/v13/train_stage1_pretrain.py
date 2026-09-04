"""Alice 预训练主程序"""

import gc
import math
import sys
import time
import traceback

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim import Muon

from config import Config
from alice_main import MainModel
from text_pretrain_data import PretrainDataProcessor, CustomEmbedding


class PretrainTrainer:
    """预训练管理器"""
    
    def __init__(self):
        self.device = torch.device(Config.train.DEVICE)
        self.log_interval = Config.train.LOG_INTERVAL
        self.save_interval = Config.train.SAVE_INTERVAL
        self.vocab_size = 21128
        
        self.embedding = CustomEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=Config.model.N
        ).to(self.device)
        
        self.model = MainModel()
        self.output_projection = nn.Linear(Config.model.N, self.vocab_size).to(self.device)
        
        self._init_optimizers()
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 混合精度训练
        self.use_mixed_precision = Config.flow.ENABLE_MIXED_PRECISION
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler(device=self.device.type)
            print("[配置] 已启用混合精度训练")
        else:
            self.scaler = None
        
        self.stats = {'total_steps': 0, 'total_loss': 0.0, 'mlm_loss': 0.0}
        self.best_loss = float('inf')
        self.recent_epochs = []
        self.keep_last_n = 5
        
        # Loss 历史记录（用于绘图）
        self.loss_history = []  # 每个 batch 的 loss
        self.avg_loss_history = []  # 累积平均 loss
        
        total_params = sum(p.numel() for p in self.model.parameters()) + \
                      sum(p.numel() for p in self.embedding.parameters()) + \
                      sum(p.numel() for p in self.output_projection.parameters())
        print(f"总参数量：{total_params:,}")
        print(f"设备：{self.device}")
    
    def _init_optimizers(self):
        params = list(self.model.parameters()) + list(self.embedding.parameters()) + list(self.output_projection.parameters())
        for param in params:
            param.requires_grad = True
        
        params_2d = [p for p in params if p.dim() == 2]
        params_1d = [p for p in params if p.dim() == 1]
        
        print(f"Model 参数：{len(list(self.model.parameters()))}")
        print(f"Embedding 参数：{len(list(self.embedding.parameters()))}")
        print(f"Projection 参数：{len(list(self.output_projection.parameters()))}")
        print(f"收集到 {len(params_2d)} 个 2D 参数，{len(params_1d)} 个 1D 参数")
        
        self.optimizer = Muon(params_2d, 
                              lr=Config.optimizer.LEARNING_RATE,
                              momentum=0.95, 
                              weight_decay=Config.optimizer.WEIGHT_DECAY)
        self.bias_optimizer = torch.optim.AdamW(params_1d, 
                                                lr=Config.optimizer.LEARNING_RATE, 
                                                weight_decay=Config.optimizer.WEIGHT_DECAY) if params_1d else None
        print(f"优化器：Muon ({len(params_2d)} 个 2D 参数，lr={Config.optimizer.LEARNING_RATE}) + AdamW ({len(params_1d)} 个 bias, lr={Config.optimizer.LEARNING_RATE})")
    
    def compute_mlm_loss(self, output_lists, labels):
        """计算 MLM 损失"""
        if len(output_lists) == 0:
            return torch.tensor(0.0, device=self.device), {}
        
        target_labels = labels[0].view(-1)
        accumulated_logits = None
        
        for output in output_lists:
            # 检查 output 是否有效
            if not torch.isfinite(output).all():
                print(f"[警告] output 包含 NaN/Inf，跳过此输出")
                continue
            
            projected = self.output_projection(output.to(self.device))
            
            if accumulated_logits is None:
                accumulated_logits = projected
            else:
                accumulated_logits = accumulated_logits + projected
        
        if accumulated_logits is None:
            return torch.tensor(0.0, device=self.device), {}
        
        logits_flat = accumulated_logits.view(-1, self.vocab_size)
        labels_flat = target_labels.view(-1)
        
        # 检查有效标签数量
        valid_tokens = (labels_flat != -100).sum().item()
        if valid_tokens == 0:
            print(f"[警告] 没有有效标签，返回 0 loss")
            return torch.tensor(0.0, device=self.device), {'mlm_loss': 0.0, 'total_tokens': 0}
        
        mlm_loss = self.mlm_criterion(logits_flat, labels_flat)
        
        loss_dict = {
            'mlm_loss': mlm_loss.item(),
            'total_tokens': valid_tokens
        }
        
        del accumulated_logits, logits_flat, labels_flat
        return mlm_loss, loss_dict
    
    def train_epoch(self, dataloader, epoch):
        """训练一个 epoch"""
        epoch_start = time.time()
        
        self.model.train()
        self.embedding.train()
        self.output_projection.train()
        
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            print(f"\n[显存] Epoch {epoch} 开始：{initial_memory:.2f} GB")
        
        for batch_idx, batch_dict in enumerate(progress_bar):
            embedded = batch_dict['embedded'].to(self.device)
            labels = batch_dict['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            if self.bias_optimizer is not None:
                self.bias_optimizer.zero_grad()
            
            try:
                if self.use_mixed_precision and self.scaler is not None:
                    with autocast(device_type=str(self.device)):
                        output_lists = self.model(embedded, epoch=epoch)
                else:
                    output_lists = self.model(embedded, epoch=epoch)
                
                if len(output_lists) == 0:
                    print(f"[警告] Batch {batch_idx}: 无 Type3 Reason 输出，跳过")
                    continue
                
                mlm_loss, loss_dict = self.compute_mlm_loss(output_lists, labels)
                
                if self.use_mixed_precision and self.scaler is not None:
                    scaled_loss = self.scaler.scale(mlm_loss)
                    scaled_loss.backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.optimizer.MAX_GRAD_NORM)
                    if self.bias_optimizer is not None:
                        self.scaler.unscale_(self.bias_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for group in self.bias_optimizer.param_groups for p in group['params']], 
                            Config.optimizer.MAX_GRAD_NORM
                        )
                    self.scaler.step(self.optimizer)
                    if self.bias_optimizer is not None:
                        self.scaler.step(self.bias_optimizer)
                    self.scaler.update()
                else:
                    mlm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.optimizer.MAX_GRAD_NORM)
                    self.optimizer.step()
                    if self.bias_optimizer is not None:
                        self.bias_optimizer.step()
                
                batch_loss = mlm_loss.item()
                
                # 检测 NaN，跳过无效的 batch
                if not math.isfinite(batch_loss):
                    print(f"\n[警告] Batch {batch_idx+1} loss=NaN，跳过此 batch")
                    del output_lists, mlm_loss, loss_dict, embedded, labels
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                epoch_loss += batch_loss
                num_batches += 1
                
                # 记录 loss 历史
                self.loss_history.append(batch_loss)
                
                progress_bar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg_loss': f'{epoch_loss / max(1, num_batches):.4f}'})
                
                if (batch_idx + 1) % self.log_interval == 0 or batch_idx == 0:
                    avg_loss = sum(self.loss_history) / len(self.loss_history)
                    print(f"  [Batch {batch_idx+1}] loss={batch_loss:.4f}, avg_loss={avg_loss:.4f}")
                
                self.stats['total_steps'] += 1
                self.stats['total_loss'] += batch_loss
                self.stats['mlm_loss'] += loss_dict.get('mlm_loss', 0.0)
                
                del output_lists, mlm_loss, loss_dict, embedded, labels
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_usage = torch.cuda.memory_allocated(self.device) / torch.cuda.get_device_properties(self.device).total_memory * 100
                    if memory_usage > 90:
                        print(f"\n警告：显存占用过高 ({memory_usage:.1f}%), 自动退出")
                        return epoch_loss / max(1, num_batches), False
            
            except Exception as e:
                print(f"\n[Error] Batch {batch_idx} failed: {e}")
                traceback.print_exc()
                del embedded, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        avg_loss = epoch_loss / max(1, num_batches)
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch} 完成：平均损失 {avg_loss:.4f}")
        print(f"[时间] Epoch {epoch} 耗时：{epoch_time/60:.2f} 分钟 ({epoch_time:.1f} 秒)")
        print(f"  平均每 batch: {epoch_time/max(1,num_batches):.2f} 秒")
        
        is_best = avg_loss < self.best_loss
        if is_best:
            self.best_loss = avg_loss
            print(f"\n刷新最佳损失：{avg_loss:.4f}")
        
        # 绘制 loss 曲线
        self._plot_loss_curve(epoch)
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3
            print(f"\n[显存统计] Epoch {epoch}:")
            print(f"  初始：{initial_memory:.2f} GB")
            print(f"  最终：{final_memory:.2f} GB")
            print(f"  峰值：{peak_memory:.2f} GB")
            print(f"  净变化：{final_memory - initial_memory:+.2f} GB")
            torch.cuda.empty_cache()
            print(f"[显存清理] 当前已用：{torch.cuda.memory_allocated(self.device)/1024**3:.2f} GB")
        
        return avg_loss, True
    
    def _plot_loss_curve(self, epoch):
        """绘制 loss 曲线图"""
        if not self.loss_history:
            return
        
        plt.figure(figsize=(12, 5))
        
        # 子图 1：每个 batch 的 loss
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, 'b-', label='Batch Loss', alpha=0.7, linewidth=1)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title(f'Epoch {epoch} - Batch Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 2：累积平均 loss
        avg_losses = [sum(self.loss_history[:i+1])/(i+1) for i in range(len(self.loss_history))]
        plt.subplot(1, 2, 2)
        plt.plot(avg_losses, 'r-', label='Average Loss', linewidth=2)
        plt.xlabel('Batch')
        plt.ylabel('Average Loss')
        plt.title(f'Epoch {epoch} - Average Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_dir = Path('plots')
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f'loss_epoch_{epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss 曲线已保存到 {save_path}")
        
        plt.close()
    
    def train(self, dataloader, num_epochs=1):
        """完整训练流程（多轮次）"""
        total_start = time.time()
        
        print(f"开始训练，共 {num_epochs} 个 epoch...")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs}\n{'='*60}\n")
            train_loss, success = self.train_epoch(dataloader, epoch)
            
            print(f"\n保存 Epoch {epoch} 检查点...")
            self.save_checkpoint(epoch, is_best=(train_loss == self.best_loss))
            
            if not success:
                print(f"\nEpoch {epoch} 因显存过高提前终止")
                break
        
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"训练完成！最终损失：{self.stats['total_loss']/max(1, self.stats['total_steps']):.4f}")
        print(f"总耗时：{total_time/3600:.2f} 小时 (平均每 epoch: {total_time/num_epochs/60:.2f} 分钟)")
        print(f"{'='*60}")
    
    def train_streaming(self, file_paths, num_epochs_per_file=None, batch_size=1, start_file_idx=0, max_chunks=1):
        """流式训练（逐个文件加载）"""
        total_start = time.time()
        
        # 使用配置的 EPOCHS_PER_CHUNK
        if num_epochs_per_file is None:
            num_epochs_per_file = Config.train.EPOCHS_PER_CHUNK
        
        print(f"\n{'='*60}")
        print(f"流式训练模式")
        print(f"文件数量：{len(file_paths)}")
        print(f"起始索引：{start_file_idx} / {len(file_paths)}")
        print(f"每文件训练：{num_epochs_per_file} 轮")
        print(f"批次大小：{batch_size}")
        print(f"每文件文本块数：{max_chunks}")
        print(f"{'='*60}\n")
        
        global_epoch = 0
        progress_file = Path('checkpoints/training_progress.txt')
        
        for file_idx in range(start_file_idx, len(file_paths)):
            file_path = file_paths[file_idx]
            
            print(f"\n{'='*60}")
            print(f"处理文件 [{file_idx+1}/{len(file_paths)}]: {Path(file_path).name}")
            print(f"{'='*60}\n")
            
            try:
                dataloader = create_train_dataloader(
                    file_paths=[file_path],
                    batch_size=batch_size,
                    max_chunks=max_chunks
                )
                
                print(f"数据加载完成")
                
                for epoch_in_file in range(num_epochs_per_file):
                    print(f"\n  文件内 Epoch {epoch_in_file+1}/{num_epochs_per_file}")
                    
                    train_loss, success = self.train_epoch(dataloader, global_epoch)
                    
                    print(f"\n  保存检查点...")
                    self.save_checkpoint(global_epoch, is_best=(train_loss == self.best_loss))
                    
                    if not success:
                        print(f"\n  Epoch {global_epoch} 因显存过高提前终止")
                        break
                    
                    global_epoch += 1
                
                # ========== 构建长期记忆索引 ==========
                if Config.flow.ENABLE_LONG_TERM_MEMORY:
                    print(f"\n  构建长期记忆索引...")
                    try:
                        self.model.long_term_memory.build_index()
                        print(f"  长期记忆索引已构建并加载")
                    except Exception as e:
                        print(f"  构建索引失败：{e}")
                        traceback.print_exc()
                # ===========================================
                
                del dataloader
                gc.collect()
                
                # 保存进度
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_file, 'w') as f:
                    f.write(f'next_file_idx={file_idx + 1}\n')
                    f.write(f'total_files={len(file_paths)}\n')
                    f.write(f'global_epoch={global_epoch}\n')
                    f.write(f'best_loss={self.best_loss:.4f}\n')
                
                print(f"\n文件 {Path(file_path).name} 处理完成")
                print(f"进度已保存：{progress_file}")
                
            except Exception as e:
                print(f"\n文件 {file_path} 处理失败：{e}")
                traceback.print_exc()
                with open(progress_file, 'w') as f:
                    f.write(f'next_file_idx={file_idx}\n')
                    f.write(f'error={str(e)}\n')
                continue
        
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"流式训练完成！")
        print(f"处理文件数：{len(file_paths) - start_file_idx}")
        print(f"总 epoch 数：{global_epoch}")
        print(f"最终损失：{self.stats['total_loss']/max(1, self.stats['total_steps']):.4f}")
        print(f"总耗时：{total_time/3600:.2f} 小时")
        print(f"{'='*60}")
    
    def reset(self):
        if hasattr(self.model, 'remain_manager'):
            self.model.remain_manager.clear()
        self.stats = {'total_steps': 0, 'total_loss': 0.0, 'mlm_loss': 0.0}
        print("已重置模型状态")
    
    def save_checkpoint(self, epoch, checkpoint_dir='checkpoints', is_best=False):
        """保存检查点（保留最优 + 最后 N 轮）"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        save_path = checkpoint_dir / f'pretrain_epoch_{epoch}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'embedding_state_dict': self.embedding.state_dict(),
            'projection_state_dict': self.output_projection.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'bias_optimizer_state_dict': self.bias_optimizer.state_dict() if self.bias_optimizer else None,
            'stats': self.stats,
            'vocab_size': self.vocab_size,
            'best_loss': self.best_loss,
        }
        
        torch.save(checkpoint, save_path)
        print(f"检查点已保存到 {save_path}")
        
        # 如果是最优模型，单独保存
        if is_best:
            best_path = checkpoint_dir / 'pretrain_best.pt'
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存到 {best_path} (loss={self.best_loss:.4f})")
        
        # 管理最近 N 轮的检查点
        self.recent_epochs.append(epoch)
        
        # 清理超出数量的旧检查点
        while len(self.recent_epochs) > self.keep_last_n:
            old_epoch = self.recent_epochs.pop(0)
            old_path = checkpoint_dir / f'pretrain_epoch_{old_epoch}.pt'
            # 检查这个旧检查点是否就是最优模型（通过比较 loss）
            should_delete = True
            
            if old_path.exists():
                # 读取旧检查点的 loss
                try:
                    old_checkpoint = torch.load(old_path, map_location='cpu')
                    old_loss = old_checkpoint.get('best_loss', float('inf'))
                    
                    # 如果旧检查点的 loss 等于当前最佳 loss，说明它是最优模型，不删除
                    if abs(old_loss - self.best_loss) < 1e-6:
                        should_delete = False
                        print(f"  跳过删除 pretrain_epoch_{old_epoch}.pt (是最优模型)")
                except Exception as e:
                    print(f"  无法读取旧检查点：{e}，仍然删除")
                
                if should_delete:
                    old_path.unlink()
                    print(f"已清理旧检查点：pretrain_epoch_{old_epoch}.pt")


def create_train_dataloader(file_paths, batch_size=None, max_chunks=None):
    """创建训练数据加载器"""
    if batch_size is None:
        batch_size = Config.data.BATCH_SIZE
    
    return PretrainDataProcessor(
        file_paths=file_paths,
        chunk_size=Config.data.CHUNK_SIZE,
        overlap=Config.data.OVERLAP,
        batch_size=batch_size,
        num_workers=2,
        max_chunks=max_chunks
    )


def main():
    """主函数 - 自动检测是否续训"""
    print("="*60)
    print("Alice 预训练系统 v9.0")
    print("="*60)
    
    # 检查是否有检查点文件
    checkpoint_path = Path('checkpoints/pretrain_best.pt')
    progress_file = Path('checkpoints/training_progress.txt')
    
    if checkpoint_path.exists() and progress_file.exists():
        print(f"\n发现检查点文件：{checkpoint_path}")
        print(f"发现进度文件：{progress_file}")
        print(f"\n自动进入续训模式...")
        print()
        main_resume()
        return
    
    # 否则从头开始训练
    print(f"\n未找到检查点，从头开始训练...")
    print()
    
    print(f"\n优化配置:")
    print(f"  梯度检查点：{Config.flow.ENABLE_GRADIENT_CHECKPOINTING}")
    print(f"  混合精度训练：{Config.flow.ENABLE_MIXED_PRECISION}")
    print(f"  最大迭代次数：{Config.flow.MAX_ITERATIONS}")
    print(f"  激活池容量：{Config.flow.ACTIVATED_POOL_MAX_SIZE}")
    print()
    
    trainer = PretrainTrainer()
    
    # 只使用 AA 目录下的第一个文件
    data_dir = Path('wiki_zh_2019/wiki_zh/AA')
    test_file = list(sorted(data_dir.glob('wiki_*')))[0]
    
    print(f"测试文件：{test_file}")
    print(f"预计文本块数：~422 个")
    print()
    
    print(f"开始训练...")
    print(f"  每文件训练轮数：1")
    print(f"  批次大小：1")
    print(f"  每个文件文本块数：全部 (约 422 个)")
    print(f"  预计耗时：~35 分钟")
    print()
    
    trainer.train_streaming(
        file_paths=[str(test_file)],
        num_epochs_per_file=Config.train.EPOCHS_PER_CHUNK,
        batch_size=None,  # 使用配置的默认值
        max_chunks=10  # 使用全部文本块
    )
    
    print("\n测试完成！")


def main_resume():
    """断点续训 - 从中断处继续训练（自动读取进度）"""
    print("="*60)
    print("Alice 预训练系统 v9.0 - 断点续训")
    print("="*60)
    
    # 1. 检查进度文件和检查点
    progress_file = Path('checkpoints/training_progress.txt')
    checkpoint_path = Path('checkpoints/pretrain_best.pt')
    
    if not progress_file.exists():
        print("警告：未找到进度文件，将从头开始训练")
        return main()
    
    if not checkpoint_path.exists():
        print("警告：未找到检查点文件，将从头开始训练")
        return main()
    
    # 2. 读取进度
    progress = {}
    with open(progress_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                progress[key] = value
    
    start_file_idx = int(progress.get('next_file_idx', 0))
    total_files = int(progress.get('total_files', 0))
    prev_best_loss = float(progress.get('best_loss', 999))
    
    print(f"\n训练进度:")
    print(f"  已处理：{start_file_idx} / {total_files} 个文件")
    print(f"  剩余：{total_files - start_file_idx} 个文件")
    print(f"  当前最佳损失：{prev_best_loss:.4f}")
    print()
    
    # 3. 加载模型检查点
    checkpoint_path = Path('checkpoints/pretrain_best.pt')
    
    if checkpoint_path.exists():
        print(f"加载检查点：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
    else:
        print("警告：未找到模型检查点，将从头开始训练")
        checkpoint = None
    
    # 4. 创建训练器
    trainer = PretrainTrainer()
    
    if checkpoint:
        # 恢复模型状态
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.embedding.load_state_dict(checkpoint['embedding_state_dict'])
        trainer.output_projection.load_state_dict(checkpoint['projection_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['bias_optimizer_state_dict']:
            trainer.bias_optimizer.load_state_dict(checkpoint['bias_optimizer_state_dict'])
        
        trainer.best_loss = checkpoint['best_loss']
        trainer.stats = checkpoint['stats']
        
        print(f"模型权重已恢复")
        print(f"优化器状态已恢复")
        print(f"最佳损失：{trainer.best_loss:.4f}")
        print(f"已训练步数：{trainer.stats['total_steps']}")
    
    print()
    
    # 5. 准备文件列表
    data_dir = Path('wiki_zh_2019/wiki_zh')
    
    all_files = []
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            files = list(subdir.glob('wiki_*'))
            all_files.extend(sorted(files))
    
    remaining_files = [str(f) for f in all_files[start_file_idx:]]
    
    print(f"\n数据集信息:")
    print(f"  总文件数：{len(all_files)}")
    print(f"  剩余文件数：{len(remaining_files)}")
    print()
    
    # 6. 继续训练
    trainer.train_streaming(
        file_paths=remaining_files,
        num_epochs_per_file=Config.train.EPOCHS_PER_CHUNK,
        batch_size=None,  # 使用配置的默认值
        start_file_idx=0,  # 因为已经是剩余文件了
        max_chunks=10  # 每个文件处理所有文本块
    )
    
    print("\n续训完成！")


if __name__ == "__main__":
    # 自动模式：检测检查点并决定是否续训
    main()
