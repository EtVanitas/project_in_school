"""
文本预处理和词嵌入模块（精简版）

功能：
1. BERT-base-chinese 分词器加载
2. 自定义可训练 Embedding 层（1024 维）
3. 智能文本块切分（优先标点符号处断开）
4. MLM 掩码生成
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import re


class CustomEmbedding(nn.Module):
    """自定义可训练的 Embedding 层（1024 维）"""
    
    def __init__(self, vocab_size=21128, embedding_dim=1024, padding_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class TextChunkDataset(Dataset):
    """文本块数据集 - 读取、分词、智能切分"""
    
    def __init__(self, file_paths: List[str], chunk_size=1024, overlap=128, 
                 max_chunks: Optional[int] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # 加载分词器
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.sep_token_id
        
        # 处理文本
        self.chunks = self._process_files(file_paths)
        if max_chunks:
            self.chunks = self.chunks[:max_chunks]
    
    def _process_files(self, file_paths: List[str]) -> List[List[int]]:
        """读取文件并处理为文本块"""
        texts = []
        for fp in file_paths:
            fp = Path(fp)
            if not fp.exists():
                continue
            
            with open(fp, 'r', encoding='utf-8') as f:
                if fp.suffix == '.txt':
                    texts.append(f.read().strip())
                elif fp.suffix == '.jsonl':
                    for line in f:
                        try:
                            data = json.loads(line)
                            for field in ['content', 'text', 'body']:
                                if field in data:
                                    texts.append(str(data[field]))
                                    break
                        except:
                            continue
        
        if not texts:
            raise ValueError("未读取到文本")
        
        # 分词
        full_text = " ".join(texts)
        sentences = re.split(r'([。！？；!?;]\s*)', full_text)
        
        all_tokens = []
        for i in range(0, len(sentences), 100):
            batch = sentences[i:i+100]
            encodings = self.tokenizer(batch, add_special_tokens=False,
                                      return_attention_mask=False,
                                      return_token_type_ids=False)
            for ids in encodings['input_ids']:
                all_tokens.extend(ids)
                all_tokens.append(self.sep_id)
        
        # 智能切分
        return self._smart_chunk(all_tokens)
    
    def _smart_chunk(self, tokens: List[int]) -> List[List[int]]:
        """智能切分，优先在标点处断开"""
        chunks = []
        boundary_tokens = set()
        for char in '。！？；!?;,，':
            encoded = self.tokenizer.encode(char, add_special_tokens=False)
            if len(encoded) == 1:
                boundary_tokens.add(encoded[0])
        
        step = self.chunk_size - self.overlap
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            
            if end >= len(tokens):
                chunk = tokens[start:] + [self.pad_id] * (self.chunk_size - len(tokens[start:]))
                chunks.append(chunk)
                break
            
            # 寻找边界 token
            best_end = end
            search_start = max(start, end - self.overlap)
            for i in range(end - 1, search_start - 1, -1):
                if tokens[i] in boundary_tokens:
                    best_end = i + 1
                    break
            
            chunk = tokens[start:best_end]
            if len(chunk) < self.chunk_size:
                chunk += [self.pad_id] * (self.chunk_size - len(chunk))
            
            chunks.append(chunk)
            start = best_end - self.overlap if self.overlap > 0 else end
        
        return chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)


class MLMPreprocessor:
    """MLM 掩码预处理器"""
    
    def __init__(self, tokenizer, mask_prob=0.15, random_replace_prob=0.1, 
                 keep_prob=0.1):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.random_replace_prob = random_replace_prob
        self.keep_prob = keep_prob
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)
        
        self.special_tokens = {
            tokenizer.pad_token_id, tokenizer.cls_token_id,
            tokenizer.sep_token_id, tokenizer.mask_token_id,
            tokenizer.unk_token_id
        }
    
    def __call__(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        labels = input_ids.clone()
        masked_input = input_ids.clone()
        
        prob_matrix = torch.full((batch_size, seq_len), self.mask_prob)
        for sid in self.special_tokens:
            prob_matrix[input_ids == sid] = 0.0
        
        mask = torch.bernoulli(prob_matrix).bool()
        
        for b in range(batch_size):
            for s in range(seq_len):
                if mask[b, s]:
                    rand = torch.rand(1).item()
                    if rand < self.keep_prob:
                        pass
                    elif rand < self.keep_prob + self.random_replace_prob:
                        masked_input[b, s] = torch.randint(0, self.vocab_size, (1,)).item()
                    else:
                        masked_input[b, s] = self.mask_id
        
        labels[~mask] = -100
        return masked_input, labels


class TextEmbeddingProcessor:
    """统一的文本处理接口（推荐使用）"""
    
    def __init__(self, file_paths: List[str], chunk_size=1024, overlap=128,
                 batch_size=4, num_workers=4, mlm_config=None):
        self.batch_size = batch_size
        
        self.dataset = TextChunkDataset(file_paths, chunk_size, overlap)
        self.embedding = CustomEmbedding(
            vocab_size=len(self.dataset.tokenizer),
            embedding_dim=1024,
            padding_idx=self.dataset.pad_id
        )
        
        if mlm_config is None:
            mlm_config = {
                'mask_prob': 0.15,
                'random_replace_prob': 0.1,
                'keep_prob': 0.1
            }
        self.mlm_preprocessor = MLMPreprocessor(self.dataset.tokenizer, **mlm_config)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=(num_workers > 0)
        )
    
    def process_batch(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理批次数据"""
        masked_input, labels = self.mlm_preprocessor(token_ids)
        embedded = self.embedding(token_ids)
        return embedded, masked_input, labels
    
    def __iter__(self):
        for token_ids in self.dataloader:
            embedded, masked_input, labels = self.process_batch(token_ids)
            yield {
                'embedded': embedded,
                'masked_input': masked_input,
                'labels': labels,
                'token_ids': token_ids
            }
    
    def __len__(self):
        return len(self.dataloader)


def create_test_data(output_dir="test_data", num_files=2):
    """创建测试数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    sample_texts = [
        "人工智能是研究使计算机模拟人类智能活动的技术科学。",
        "机器学习是人工智能的核心，使计算机能够从数据中学习。",
        "深度学习模仿人脑的工作方式来处理数据和识别模式。",
    ]
    
    for i in range(num_files):
        txt_path = output_dir / f"test_{i}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for _ in range(100):
                f.write(sample_texts[i % len(sample_texts)] + "\n")
    
    print(f"测试数据已生成到 {output_dir}")
    return [str(output_dir / f"test_{i}.txt") for i in range(num_files)]
