"""
增强版预训练主文件

整合文本预处理和增强版 Alice 模型（带 Manager 辅助管理）
修复损失计算：输出投影到 vocab_size 后求交叉熵

功能：
1. 数据加载和 MLM 掩码生成
2. 前向传播（增强版 Alice 模型）
3. 损失计算（投影层 + MLM 交叉熵 + 步数惩罚 + 激活分布熵）
4. 反向传播和优化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import time

# 导入配置
from config import Config

# 导入增强版模块
from alice_main_enhanced import MainModelEnhanced
from text_embedding_lite import TextEmbeddingProcessor, CustomEmbedding


class EnhancedPretrainTrainer:
    """增强版预训练训练器"""
    
    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.train.DEVICE)
        
        # ========== 初始化组件 ==========
        print("初始化模型和数据处理器...")
        
        # 词嵌入层
        self.embedding = CustomEmbedding(
            vocab_size=21128,
            embedding_dim=config.model.M
        ).to(self.device)
        
        # Alice 主体模型
        self.model = MainModelEnhanced().to(self.device)
        
        # 输出投影层 Linear(1024, vocab_size)
        self.output_projection = nn.Linear(
            config.model.M, 
            21128  # BERT-base-chinese 词表大小
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + 
            list(self.embedding.parameters()) +
            list(self.output_projection.parameters()),
            lr=config.optimizer.LEARNING_RATE,
            weight_decay=config.optimizer.WEIGHT_DECAY,
            betas=config.optimizer.BETAS
        )
        
        # MLM 损失函数
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 统计信息
        self.stats = {
            'total_steps': 0,
            'total_loss': 0.0,
            'output_loss': 0.0,
            'steps_penalty': 0.0,
            'balance_entropy': 0.0
        }
        
        total_params = (
            sum(p.numel() for p in self.model.parameters()) +
            sum(p.numel() for p in self.embedding.parameters()) +
            sum(p.numel() for p in self.output_projection.parameters())
        )
        print(f"总参数量：{total_params:,}")
        print(f"设备：{self.device}")
    
    def compute_total_loss(self, outputs, labels, stats_from_model):
        """
        计算总损失
        
        L = w_out * L_output + w_steps * L_steps + w_bal * L_balance
        
        Args:
            outputs: List[Tensor], MainModel 的输出列表，每个元素 (batch, m, n)
            labels: Tensor, 真实标签 (batch, seq_len)
            stats_from_model: Dict, MainModel 返回的统计信息
            
        Returns:
            total_loss, loss_dict
        """
        batch_size = labels.shape[0]
        progress_ratio = self.stats['total_steps'] / max(1, self.config.train.NUM_EPOCHS * 1000)
        w_out, w_steps, w_bal = self.config.loss.get_weights(progress_ratio)
        
        # ========== 1. 输出损失 L_output (Scheme A: 各自投影后相加) ==========
        # 每个输出独立投影到 vocab_size，然后相加 logits，最后计算交叉熵
        all_logits = []
        
        for out in outputs:
            if out.dim() == 2:
                out = out.unsqueeze(0)  # (1, m, n)
            
            # 投影到 vocab_size
            logits_i = self.output_projection(out)  # (batch, m, vocab_size)
            all_logits.append(logits_i)
        
        if len(all_logits) > 0:
            # 相加 logits
            logits_sum = torch.stack(all_logits).sum(dim=0)  # (batch, m, vocab_size)
            
            # 在 m 维度上平均池化
            logits_pooled = logits_sum.mean(dim=1)  # (batch, vocab_size)
            
            # 与 labels 计算交叉熵
            labels_pooled = labels[:, 0]  # 取第一个 token 的标签
            output_loss = self.mlm_criterion(logits_pooled, labels_pooled)
        else:
            # 没有输出时，用零占位
            output_loss = torch.zeros(1, device=self.device).mean()
        
        # ========== 2. 推理次数惩罚 L_steps ==========
        iteration_count = stats_from_model.get('iterations', 0)
        target_iterations = 10  # 理想迭代次数
        steps_penalty = torch.abs(torch.tensor(iteration_count - target_iterations, 
                                               device=self.device)).float()
        
        # ========== 3. 激活分布熵 L_balance ==========
        activate_usage = stats_from_model.get('activate_usage', [1] * 72)
        usage_tensor = torch.tensor(activate_usage, dtype=torch.float32, device=self.device)
        usage_prob = usage_tensor / (usage_tensor.sum() + 1e-6)
        balance_entropy = -(usage_prob * torch.log(usage_prob + 1e-6)).sum()
        
        # ========== 4. 总损失 ==========
        total_loss = w_out * output_loss + w_steps * steps_penalty + w_bal * balance_entropy
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'output_loss': output_loss.item(),
            'steps_penalty': steps_penalty.item(),
            'balance_entropy': balance_entropy.item(),
            'weights': {
                'w_out': w_out,
                'w_steps': w_steps,
                'w_bal': w_bal
            }
        }
        
        return total_loss, loss_dict
    
    def train_step(self, batch_data):
        """单个训练步骤"""
        self.optimizer.zero_grad()
        
        # 1. 数据准备
        tokens = batch_data['input_ids'].to(self.device)  # (batch, seq_len)
        masked_tokens, labels = batch_data['mlm_processed']
        masked_tokens = masked_tokens.to(self.device)
        labels = labels.to(self.device)
        
        # 2. 词嵌入
        x = self.embedding(masked_tokens)  # (batch, seq_len, 1024)
        
        # 由于 Alice 接受 (m, n) = (1024, 1024)，需要调整形状
        # 假设 seq_len=1024，则 x: (batch, 1024, 1024)
        if x.size(1) != self.config.model.M or x.size(2) != self.config.model.N:
            # 填充或截断
            x = x[:, :self.config.model.M, :self.config.model.N]
        
        # 3. 前向传播
        outputs, model_stats = self.model(x)
        
        # 4. 损失计算
        total_loss, loss_dict = self.compute_total_loss(outputs, labels, model_stats)
        
        # 5. 反向传播
        total_loss.backward()
        
        # 6. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.train.GRAD_CLIP_NORM
        )
        
        # 7. 更新参数
        self.optimizer.step()
        
        # 8. 更新统计
        self.stats['total_steps'] += 1
        self.stats['total_loss'] += loss_dict['total_loss']
        self.stats['output_loss'] += loss_dict['output_loss']
        self.stats['steps_penalty'] += loss_dict['steps_penalty']
        self.stats['balance_entropy'] += loss_dict['balance_entropy']
        
        return loss_dict
    
    def train_one_epoch(self, dataloader, epoch):
        """训练一个 epoch"""
        self.model.train()
        self.embedding.train()
        self.output_projection.train()
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        epoch_stats = {
            'total_loss': 0.0,
            'output_loss': 0.0,
            'steps_penalty': 0.0,
            'balance_entropy': 0.0,
            'num_batches': 0
        }
        
        for batch_idx, batch in enumerate(pbar):
            # 训练一步
            loss_dict = self.train_step(batch)
            
            # 更新 epoch 统计
            for key in ['total_loss', 'output_loss', 'steps_penalty', 'balance_entropy']:
                epoch_stats[key] += loss_dict[key]
            epoch_stats['num_batches'] += 1
            
            # 日志
            if batch_idx % self.config.train.LOG_INTERVAL == 0:
                avg_loss = epoch_stats['total_loss'] / max(1, epoch_stats['num_batches'])
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'out': f'{loss_dict["output_loss"]:.4f}',
                    'step': f'{loss_dict["steps_penalty"]:.2f}',
                    'bal': f'{loss_dict["balance_entropy"]:.2f}'
                })
        
        # 计算 epoch 平均
        for key in ['total_loss', 'output_loss', 'steps_penalty', 'balance_entropy']:
            epoch_stats[key] /= max(1, epoch_stats['num_batches'])
        
        return epoch_stats
    
    def train(self, dataloader):
        """完整训练流程"""
        print("\n开始训练...")
        start_time = time.time()
        
        for epoch in range(self.config.train.NUM_EPOCHS):
            print(f"\n{'='*60}")
            print(f'Epoch {epoch+1}/{self.config.train.NUM_EPOCHS}')
            print(f'{'='*60}')
            
            epoch_stats = self.train_one_epoch(dataloader, epoch)
            
            # ========== Epoch 结束：更新长期记忆 ==========
            print(f"\n更新长期记忆...")
            self.model.remain_manager.update_epoch_end()
            
            # 打印 epoch 总结
            print(f"\nEpoch {epoch+1} 总结:")
            print(f"  平均总损失：{epoch_stats['total_loss']:.4f}")
            print(f"  平均输出损失：{epoch_stats['output_loss']:.4f}")
            print(f"  平均步数惩罚：{epoch_stats['steps_penalty']:.2f}")
            print(f"  平均平衡熵：{epoch_stats['balance_entropy']:.2f}")
            
            # 保存 checkpoint
            if (epoch + 1) % self.config.train.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch)
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时：{total_time/3600:.2f} 小时")
    
    def save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint_path = self.config.train.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch+1}.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'embedding_state_dict': self.embedding.state_dict(),
            'projection_state_dict': self.output_projection.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
            'config': {
                'M': self.config.model.M,
                'N': self.config.model.N,
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint 已保存至：{checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.embedding.load_state_dict(checkpoint['embedding_state_dict'])
        self.output_projection.load_state_dict(checkpoint['projection_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint['stats']
        
        print(f"Checkpoint 已加载：{checkpoint_path}")


def main():
    """主函数"""
    # 初始化训练器
    trainer = EnhancedPretrainTrainer()
    
    # 准备数据（示例）
    processor = TextEmbeddingProcessor()
    
    # 创建测试数据
    from text_embedding_lite import create_test_data
    create_test_data(num_samples=100)
    
    # 加载数据集
    dataset = processor.load_dataset('data')
    dataloader = processor.create_dataloader(dataset)
    
    # 开始训练
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
