"""数据加载器
提供JSONL数据集加载、tokenization和checkpoint管理
"""

import os
import json
from typing import List, Dict, Tuple
import torch
from config import PathConfig


class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.path_config = PathConfig()
    
    def load_jsonl_dataset(self) -> List[Dict]:
        """加载JSONL数据集"""
        jsonl_path = self.path_config.jsonl_dataset_path
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL文件不存在: {jsonl_path}")
        
        dataset = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if 'input' in data and 'output' in data:
                        dataset.append({
                            'input': data['input'],
                            'output': data['output']
                        })
                except Exception as e:
                    print(f"警告: 跳过第{line_num}行,解析失败: {e}")
                    continue
        
        print(f"成功加载 {len(dataset)} 条样本从 {jsonl_path}")
        return dataset
    
    def tokenize_input_output_pair(self, input_text: str, output_text: str, 
                                    tokenizer, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """将input和output转换为token序列"""
        # 分别编码input和output,不添加特殊token
        input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False).squeeze(0)
        output_ids = tokenizer.encode(output_text, return_tensors='pt', add_special_tokens=False).squeeze(0)
        
        # 移动到设备
        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)
        total_seq_len = len(input_ids) + len(output_ids)
        
        return input_ids, output_ids, total_seq_len
    
    # ==================== Checkpoint 管理 ====================
    
    def save_checkpoint(self, processed_count: int, avg_loss: float = None):
        """保存处理进度到checkpoint文件"""
        from datetime import datetime
        checkpoint_file = os.path.join(self.path_config.checkpoint_dir, 'checkpoint.json')
        
        checkpoint_data = {
            'processed_count': processed_count,
            'timestamp': datetime.now().isoformat(),
        }
        
        if avg_loss is not None:
            checkpoint_data['avg_loss'] = avg_loss
        
        # 确保目录存在
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        print(f"Checkpoint已保存: 已处理 {processed_count} 个样本")
    
    def load_checkpoint(self) -> int:
        """加载checkpoint,返回已处理的样本数"""
        checkpoint_file = os.path.join(self.path_config.checkpoint_dir, 'checkpoint.json')
        if not os.path.exists(checkpoint_file):
            print(f"未找到checkpoint文件,从头开始训练")
            return 0
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            processed_count = checkpoint_data.get('processed_count', 0)
            timestamp = checkpoint_data.get('timestamp', '未知时间')
            avg_loss = checkpoint_data.get('avg_loss', None)
            
            loss_str = f", 平均loss={avg_loss:.4f}" if avg_loss is not None else ""
            print(f"从checkpoint恢复: 已处理 {processed_count} 个样本 (时间: {timestamp}{loss_str})")
            
            return processed_count
        except Exception as e:
            print(f"加载checkpoint失败: {e},从头开始训练")
            return 0
