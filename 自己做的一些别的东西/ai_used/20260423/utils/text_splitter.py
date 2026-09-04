"""
文本分块工具 - 随机长度切分（基于 token）
支持纯文本和 JSON 格式（自动提取 text 字段）
直接保存 token IDs，避免重复 tokenize
"""

import os
import json
import random
import torch
from transformers import AutoTokenizer
from config import ModelConfig


class TextSplitter:
    """文本分块器"""
    
    def __init__(self):
        # 直接使用全局配置
        self.model_config = ModelConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path)
        self.text_dir = self.model_config.path_config.text_dir
        os.makedirs(self.text_dir, exist_ok=True)
    
    def process(self, data_file: str, max_length: int = 2048):
        """处理数据集：随机长度切分，充分利用所有 token"""
        print(f"TextSplitter: 处理 {data_file}...")
        
        # 加载并提取文本
        texts = []
        json_count = 0
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 尝试解析 JSON
                try:
                    data = json.loads(line)
                    # 如果是 JSON 且有 text 字段，提取它
                    if isinstance(data, dict) and 'text' in data:
                        text = data['text']
                        json_count += 1
                    else:
                        text = line
                except json.JSONDecodeError:
                    # 不是 JSON，直接使用整行
                    text = line
                
                if text:
                    texts.append(text)
        
        print(f"已加载 {len(texts)} 个文本（JSON 格式：{json_count} 个）")
        
        # 完整分块策略：每个文本切分成多个块，充分利用所有 token
        total_saved = 0
        
        for idx, text in enumerate(texts):
            # Tokenize 整个文本
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            actual_len = len(tokens)
            
            if actual_len == 0:
                continue
            
            # 滑动窗口分块：每次随机选择长度，从上次结束位置继续
            start_pos = 0
            while start_pos < actual_len:
                
                # 先从 1-max_length 随机采样一个目标长度
                target_length = random.randint(1, max_length)
                
                # 使用切片自动处理边界：如果剩余不够，会截断到实际长度
                end_pos = min(actual_len, start_pos + target_length)
                chunk_tokens = tokens[start_pos:end_pos]
                use_length = len(chunk_tokens)  # 实际使用的长度
                
                # 保存当前块
                len_dir = os.path.join(self.text_dir, f'len_{use_length}')
                os.makedirs(len_dir, exist_ok=True)
                existing_count = len([f for f in os.listdir(len_dir) if f.endswith('.pt')])
                
                # 保存为 .pt 文件
                file_path = os.path.join(len_dir, f'sample_{existing_count}.pt')
                torch.save(torch.tensor(chunk_tokens, dtype=torch.long), file_path)
                
                total_saved += 1
                
                # 移动到下一个位置
                start_pos += use_length
            
            if (idx + 1) % 100 == 0:
                print(f"  进度：{idx + 1}/{len(texts)}")
        
        print(f"完成！共保存 {total_saved} 个文本块到 {self.text_dir}")

