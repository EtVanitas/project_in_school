"""Stage0：基准数据生成（批量处理版）"""

import os
import sys
import gc
import torch
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig


class Stage0Trainer:
    """阶段 0：基准数据生成器（批量处理版）"""
    
    def __init__(self, batch_size: int = 32):
        # 直接使用全局配置
        self.model_config = ModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size  # 批处理大小
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 加载模型和 tokenizer
        print(f"Stage0Trainer: 加载完整模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path)
        
        # 使用 device_map="balanced" 来更好地管理显存
        self.full_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_path,
            dtype=torch.bfloat16,
            device_map="balanced"
        )
        self.full_model.eval()
        
        # 存储目录（使用 path_config）
        self.text_dir = self.model_config.path_config.text_dir
        self.base_dir = self.model_config.path_config.base_dir
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.base_dir, exist_ok=True)
    
    def compute_and_save_bases(self, max_length: int = 2048):
        """批量计算并保存基准数据"""
        if not os.path.exists(self.text_dir):
            print(f"Error: 文本目录不存在：{self.text_dir}")
            return
        
        length_folders = [f for f in os.listdir(self.text_dir) if f.startswith('len_')]
        
        if not length_folders:
            print(f"Error: 未找到长度文件夹")
            return
        
        print(f"开始处理 {len(length_folders)} 个长度文件夹...")
        print(f"基础批处理大小：{self.batch_size}")
        
        total_new = 0
        processed_count = 0
        
        for folder_name in sorted(length_folders):
            seq_len = int(folder_name.replace('len_', ''))
            if seq_len > max_length:
                continue
            
            # [动态 batch_size 调整] 短序列可以使用更大的批次
            # 如果序列长度 < 200，batch_size * 8（因为显存占用与序列长度成正比）
            if seq_len < 200:
                adjusted_batch_size = self.batch_size * 8
                print(f"\nlen_{seq_len}: 检测到短序列，batch_size 调整为 {adjusted_batch_size} (原 {self.batch_size})")
            else:
                adjusted_batch_size = self.batch_size
            
            new_count = self._process_length_group_batch(seq_len, adjusted_batch_size)
            total_new += new_count
            processed_count += 1
            
            # 每个长度组处理后清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        print(f"完成！本次新增基准：{total_new} 个")
    
    def _process_length_group_batch(self, seq_len: int, batch_size: int = None) -> int:
        """批量处理一个长度组的新增文本
        
        Args:
            seq_len: 序列长度
            batch_size: 批处理大小（如果为 None，使用 self.batch_size）
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        text_len_dir = os.path.join(self.text_dir, f'len_{seq_len}')
        if not os.path.exists(text_len_dir):
            return 0
        
        base_len_dir = os.path.join(self.base_dir, f'len_{seq_len}')
        os.makedirs(base_len_dir, exist_ok=True)
        
        # 扫描文件
        text_files = sorted([f for f in os.listdir(text_len_dir) if f.endswith('.pt')])
        existing_bases = set([f.replace('.pt', '') for f in os.listdir(base_len_dir) if f.endswith('.pt')])
        
        # 找出需要处理的文件
        need_process = []
        for text_file in text_files:
            sample_name = text_file.replace('.pt', '')
            if sample_name not in existing_bases:
                need_process.append(text_file)
        
        if not need_process:
            return 0
        
        print(f"处理 len_{seq_len}: 需新增 {len(need_process)}/{len(text_files)} 个")
        
        # 批量处理
        total_processed = 0
        
        for batch_start in range(0, len(need_process), batch_size):
            batch_files = need_process[batch_start:batch_start + batch_size]
            
            # 1. 加载批次数据
            batch_input_ids = []
            batch_sample_names = []
            
            for text_file in batch_files:
                text_path = os.path.join(text_len_dir, text_file)
                sample_id = text_file.replace('sample_', '').replace('.pt', '')
                
                input_ids = torch.load(text_path, weights_only=True)  # [seq_len]
                batch_input_ids.append(input_ids)
                batch_sample_names.append(f'sample_{sample_id}')
            
            # 2. 批量前向传播
            batch_bases = self._compute_batch_bases(batch_input_ids)
            
            # 3. 批量保存
            for sample_name, base_data in zip(batch_sample_names, batch_bases):
                base_file = os.path.join(base_len_dir, f"{sample_name}.pt")
                torch.save(base_data, base_file)
                
                # 立即释放，避免内存累积
                del base_data
            
            total_processed += len(batch_files)
            
            # 4. 清理显存
            del batch_input_ids, batch_bases
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # 进度显示
            print(f"  已处理 {total_processed}/{len(need_process)} 个样本")
        
        return total_processed
    
    @torch.no_grad()
    def _compute_batch_bases(self, batch_input_ids: List[torch.Tensor]) -> List[Dict]:
        """批量计算基准数据"""
        if not batch_input_ids:
            return []
        
        # 堆叠成 batch
        batch_tensor = torch.stack(batch_input_ids).to(self.device)  # [batch_size, seq_len]
        
        # 批量前向传播
        outputs = self.full_model(input_ids=batch_tensor)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 提取每个样本的最后一个 token
        last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # 计算 top-k
        top_k = 100
        top_logits, top_indices = torch.topk(last_logits, k=top_k, dim=-1)
        
        # 转换为列表格式
        batch_bases = []
        for i in range(len(batch_input_ids)):
            batch_bases.append({
                'top_logits': top_logits[i].cpu(),
                'top_indices': top_indices[i].cpu(),
            })
        
        del outputs, logits, last_logits, top_logits, top_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return batch_bases


def main():
    """主函数"""
    trainer = Stage0Trainer(batch_size=8)
    trainer.compute_and_save_bases(max_length=2048)


if __name__ == '__main__':
    main()
