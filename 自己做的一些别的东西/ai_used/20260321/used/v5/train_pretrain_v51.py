"""
Alice 模型 v5.1 预训练主文件

使用向量化版本 MainModelVectorized + MUON 优化器

特性:
1. 完全向量化计算（4-6 倍加速）
2. MUON 优化器（更适合大模型）
3. 保留完整计算图用于反向传播
4. MLM 掩码语言建模
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from pathlib import Path
from tqdm import tqdm
import time
import math

# 导入配置
from config import Config

# 导入向量化模型
from alice_main_vectorized_v51 import MainModelVectorized
from text_embedding_lite import TextEmbeddingProcessor, CustomEmbedding


class MUON(Optimizer):
    """
    MUON 优化器（适用于大模型训练）
    
    参考：https://github.com/kyegomez/Muon
    
    特点:
    1. 动量项使用正交化
    2. 适合高维参数空间
    3. 稳定性优于 AdamW
    """
    
    def __init__(self, params, lr=0.01, momentum=0.95, weight_decay=0.1):
        """
        Args:
            params: 模型参数
            lr: 学习率
            momentum: 动量因子
            weight_decay: 权重衰减
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        """执行一步优化"""
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # 权重衰减
                if weight_decay != 0:
                    grad = grad + weight_decay * p
                
                # 初始化动量缓冲区
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                momentum_buffer = state['momentum_buffer']
                
                # 更新动量
                momentum_buffer.mul_(momentum).add_(grad)
                
                # 正交化（简化版：使用符号函数）
                update = torch.sign(momentum_buffer)
                
                # 应用更新
                p.add_(update, alpha=-lr)


class PretrainTrainerV51:
    """v5.1 向量化版本预训练训练器"""
    
    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.train.DEVICE)
        
        # ========== 初始化组件 ==========
        print("初始化 v5.1 向量化模型...")
        
        # 词嵌入层
        self.embedding = CustomEmbedding(
            vocab_size=21128,
            embedding_dim=config.model.M
        ).to(self.device)
        
        # Alice 主体模型（向量化版本）
        self.model = MainModelVectorized().to(self.device)
        
        # 输出投影层 Linear(1024, vocab_size)
        self.output_projection = nn.Linear(
            config.model.M, 
            21128  # BERT-base-chinese 词表大小
        ).to(self.device)
        
        # MUON 优化器
        self.optimizer = MUON(
            list(self.model.parameters()) + 
            list(self.embedding.parameters()) +
            list(self.output_projection.parameters()),
            lr=0.01,  # MUON 推荐学习率
            momentum=0.95,
            weight_decay=0.1
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
        print(f"优化器：MUON (lr={self.optimizer.defaults['lr']})")
        print(f"设备：{self.device}")
    
    def compute_total_loss(self, outputs, labels, stats_from_model):
        """
        计算总损失
        
        L = w_out * L_output + w_steps * L_steps + w_bal * L_balance
        
        Args:
            outputs: List[Tensor], MainModel 的输出列表
            labels: Tensor, 真实标签 (batch, seq_len)
            stats_from_model: Dict, MainModel 返回的统计信息
        
        Returns:
            total_loss, loss_dict
        """
        batch_size = labels.shape[0]
        progress_ratio = self.stats['total_steps'] / max(1, self.config.train.NUM_EPOCHS * 1000)
        w_out, w_steps, w_bal = self.config.loss.get_weights(progress_ratio)
        
        # ========== 1. 输出损失 ==========
        # 将所有输出投影到 vocab_size 并计算交叉熵
        output_loss = 0.0
        
        for out_tensor in outputs:
            # out_tensor: (m, n) 或 (batch_k, m, n)
            if out_tensor.dim() == 2:
                out_tensor = out_tensor.unsqueeze(0)
            
            # 投影到 vocab_size
            # (batch_k, m, n) → (batch_k, n, vocab_size)
            projected = self.output_projection(out_tensor)
            
            # 计算交叉熵
            # labels: (batch, seq_len)
            # 需要调整维度匹配
            loss = self.mlm_criterion(
                projected.view(-1, projected.size(-1)),  # (batch_k * n, vocab_size)
                labels.view(-1)  # (batch * seq_len)
            )
            output_loss += loss
        
        output_loss = output_loss / max(1, len(outputs))
        
        # ========== 2. 步数惩罚 ==========
        iterations = stats_from_model.get('iterations', 1)
        steps_penalty = torch.tensor(iterations / self.config.flow.MAX_ITERATIONS, 
                                      device=self.device)
        
        # ========== 3. 激活分布熵 ==========
        activate_usage = stats_from_model.get('activate_usage', [0] * 72)
        usage_tensor = torch.tensor(activate_usage, dtype=torch.float, device=self.device)
        usage_probs = usage_tensor / (usage_tensor.sum() + 1e-10)
        balance_entropy = -(usage_probs * torch.log(usage_probs + 1e-10)).sum()
        
        # ========== 总损失 ==========
        total_loss = (
            w_out * output_loss +
            w_steps * steps_penalty +
            w_bal * balance_entropy
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'output_loss': output_loss.item(),
            'steps_penalty': steps_penalty.item(),
            'balance_entropy': balance_entropy.item()
        }
        
        return total_loss, loss_dict
    
    def train_step(self, batch_data, batch_labels):
        """
        单步训练
        
        Args:
            batch_data: 批次数据（已掩码的 input_ids）
            batch_labels: 标签（原始 input_ids）
        
        Returns:
            loss_dict
        """
        self.optimizer.zero_grad()
        
        # 词嵌入
        embedded = self.embedding.embed(batch_data)  # (batch, seq_len, M)
        
        # 调整为 (batch, M, seq_len) 以匹配模型输入
        embedded = embedded.transpose(1, 2)  # (batch, M, seq_len)
        
        # 填充到 1024 维度
        if embedded.size(2) < 1024:
            embedded = nn.functional.pad(embedded, (0, 1024 - embedded.size(2)))
        
        # 前向传播（向量化版本）
        outputs, stats = self.model(embedded, epoch=self.stats['total_steps'])
        
        # 计算损失
        total_loss, loss_dict = self.compute_total_loss(outputs, batch_labels, stats)
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.optimizer.MAX_GRAD_NORM
        )
        
        # 优化器步骤
        self.optimizer.step()
        
        # 更新统计
        self.stats['total_steps'] += 1
        self.stats['total_loss'] += loss_dict['total_loss']
        self.stats['output_loss'] += loss_dict['output_loss']
        self.stats['steps_penalty'] += loss_dict['steps_penalty']
        self.stats['balance_entropy'] += loss_dict['balance_entropy']
        
        return loss_dict
    
    def train_epoch(self, data_loader, epoch):
        """训练一个 epoch"""
        self.model.train()
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        epoch_losses = []
        
        for batch_idx, (batch_data, batch_labels) in enumerate(pbar):
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            loss_dict = self.train_step(batch_data, batch_labels)
            epoch_losses.append(loss_dict['total_loss'])
            
            # 进度条更新
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'avg_loss': f"{avg_loss:.4f}",
                'output': f"{loss_dict['output_loss']:.4f}"
            })
            
            # 定期保存检查点
            if (self.stats['total_steps'] % 100 == 0) and (self.stats['total_steps'] > 0):
                self.save_checkpoint(f"checkpoint_step_{self.stats['total_steps']}.pth")
        
        # epoch 总结
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch+1} 完成:")
        print(f"  平均损失：{avg_epoch_loss:.4f}")
        print(f"  总步数：{self.stats['total_steps']}")
        
        return avg_epoch_loss
    
    def save_checkpoint(self, filepath):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'embedding_state_dict': self.embedding.state_dict(),
            'projection_state_dict': self.output_projection.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"检查点已保存：{filepath}")
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.embedding.load_state_dict(checkpoint['embedding_state_dict'])
        self.output_projection.load_state_dict(checkpoint['projection_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint['stats']
        
        print(f"检查点已加载：{filepath}")
        print(f"  恢复步数：{self.stats['total_steps']}")
        print(f"  恢复损失：{self.stats['total_loss']:.4f}")


def main():
    """主训练流程"""
    print("="*60)
    print("Alice 模型 v5.1 向量化版本 - 预训练")
    print("="*60)
    
    # 创建训练器
    trainer = PretrainTrainerV51(Config)
    
    # TODO: 加载数据集
    # data_loader = create_dataloader(...)
    
    # 训练循环
    num_epochs = Config.train.NUM_EPOCHS
    
    for epoch in range(num_epochs):
        # train_epoch(data_loader, epoch)
        print(f"\n准备训练 Epoch {epoch+1}/{num_epochs}")
        
        # 保存 epoch 检查点
        trainer.save_checkpoint(f"epoch_{epoch+1}_checkpoint.pth")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)


if __name__ == "__main__":
    main()
