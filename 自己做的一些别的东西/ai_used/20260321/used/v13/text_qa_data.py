"""
指令微调数据加载器

提供两种数据加载器：
1. SimpleTextLoader - 简单文本补全（预训练，复用 pretrain_data.py）
2. QALoader - 问答/指令微调
"""

import random
import re

import torch
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader

from config import Config

# 复用阶段 1 的组件
from text_pretrain_data import TextChunkDataset, MLMPreprocessor


class MLMPreprocessor:
    """MLM 掩码预处理器"""
    
    def __init__(self, tokenizer, mask_prob=None):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob if mask_prob is not None else Config.data.MASK_PROBABILITY
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)
        
        self.special_tokens = {
            tokenizer.pad_token_id, tokenizer.cls_token_id,
            tokenizer.sep_token_id, tokenizer.mask_token_id,
            tokenizer.unk_token_id
        }
    
    def __call__(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 MLM 掩码"""
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
                    # 简单策略：直接替换为 [MASK]
                    masked_input[b, s] = self.mask_id
        
        labels[~mask] = -100
        return masked_input, labels


class PartialMaskPreprocessor:
    """部分掩码预处理器（从指定位置开始全部掩码）"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
    
    def __call__(
        self, 
        input_ids: torch.Tensor, 
        mask_start: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从指定位置开始全部掩码
        
        Args:
            input_ids: (batch, seq_len) - 输入 token IDs
            mask_start: int - 从这个位置开始掩码
        
        Returns:
            masked_input: (batch, seq_len) - 被掩码的输入
            labels: (batch, seq_len) - 标签（只有掩码位置有效，其他为 -100）
        """
        batch_size, seq_len = input_ids.shape
        labels = input_ids.clone()
        masked_input = input_ids.clone()
        
        # 创建掩码（mask_start 之后全部掩码）
        mask = torch.arange(seq_len).unsqueeze(0) >= mask_start
        
        # 应用掩码
        masked_input[mask] = self.mask_id
        
        # 设置 labels（未掩码的位置设为 -100）
        labels[~mask] = -100
        
        return masked_input, labels


class TextChunkDataset(Dataset):
    """文本块数据集 - 读取、分词、智能切分"""
    
    def __init__(self, file_paths: List[str], chunk_size=1024, overlap=128, 
                 max_chunks: Optional[int] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # 加载本地分词器
        tokenizer_path = Path(__file__).parent / 'bert-base-chinese'
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"未找到分词器文件夹：{tokenizer_path}\n"
                "请确保 bert-base-chinese 文件夹与 text_qa_data.py 在同一目录"
            )
        
        self.tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path))
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
        for char in '。！？；!?;,,':
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


class SimpleTextLoader:
    """
    简单文本加载器（用于预训练/文本补全）
    
    注意：推荐使用 PretrainDataProcessor (pretrain_data.py)，这个类只是为了接口统一
    """
    
    def __init__(
        self,
        file_paths: List[str],
        batch_size: int = 4,
        chunk_size: int = 1024,
        overlap: int = 128,
        mask_prob: float = 0.15,
        num_workers: int = 4,
        max_chunks: Optional[int] = None
    ):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # 数据集
        self.dataset = TextChunkDataset(
            file_paths=file_paths,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks=max_chunks
        )
        
        # 预处理器
        self.tokenizer = self.dataset.tokenizer
        self.mlm_processor = MLMPreprocessor(
            self.tokenizer,
            mask_prob=mask_prob
        )
        
        # DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self.dataloader:
            # 应用 MLM 掩码
            masked_input, labels = self.mlm_processor(batch)
            
            # 创建 attention mask（非 padding 位置为 1）
            attention_mask = (batch != self.tokenizer.pad_token_id).long()
            
            yield {
                'input_ids': masked_input,
                'labels': labels,
                'attention_mask': attention_mask,
                'metadata': {
                    'type': 'simple',
                    'mask_start': 0  # MLM 是随机掩码，没有固定起点
                }
            }
    
    def __len__(self):
        return len(self.dataloader)


class QADataset(Dataset):
    """问答数据集"""
    
    def __init__(self, file_paths: List[str], tokenizer, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.samples = []
        
        # 读取问答数据
        for fp in file_paths:
            fp = Path(fp)
            if not fp.exists():
                continue
            
            with open(fp, 'r', encoding='utf-8') as f:
                if fp.suffix == '.jsonl':
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'question' in data and 'answer' in data:
                                self.samples.append({
                                    'question': data['question'],
                                    'answer': data['answer']
                                })
                        except:
                            continue
        
        if max_samples:
            self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


class QALoader:
    """
    问答加载器（用于指令微调）
    
    功能：
    - 读取问答对数据（.jsonl）
    - 支持单块或多块输入
    - 问题部分完整，答案部分掩码
    - 学习理解指令并回答
    
    返回数据格式：
    {
        'chunks': List[Tensor],         # 文本块列表
        'labels': List[Tensor],         # 每个块的标签
        'attention_mask': Tensor,       # 注意力掩码
        'metadata': {
            'type': 'qa',
            'questions': List[str],     # 原始问题列表
            'answers': List[str],       # 原始答案列表
            'num_chunks': int           # 文本块数量
        }
    }
    """
    
    def __init__(
        self,
        file_paths: List[str],
        batch_size: int = 4,
        chunk_size: int = 1024,
        num_workers: int = 4,
        max_samples: Optional[int] = None,
        max_chunks: int = 1  # 默认单块，可配置为多块
    ):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        
        # 加载本地分词器
        tokenizer_path = Path(__file__).parent / 'bert-base-chinese'
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"未找到分词器文件夹：{tokenizer_path}\n"
                "请确保 bert-base-chinese 文件夹与 text_qa_data.py 在同一目录"
            )
        
        self.tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path))
        
        # 数据集
        self.dataset = QADataset(
            file_paths=file_paths,
            tokenizer=self.tokenizer,
            max_samples=max_samples
        )
        
        # 部分掩码预处理器
        self.partial_mask_processor = PartialMaskPreprocessor(self.tokenizer)
        
        # DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
    
    def _split_into_chunks(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """将长序列切分成多个文本块"""
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk = tokens[i:i+self.chunk_size]
            if len(chunk) < self.chunk_size:
                # padding
                chunk = torch.cat([
                    chunk,
                    torch.full(
                        (self.chunk_size - len(chunk),),
                        self.tokenizer.pad_token_id,
                        dtype=chunk.dtype
                    )
                ])
            chunks.append(chunk)
            
            # 检查是否超过最大块数
            if len(chunks) >= self.max_chunks:
                break
        
        return chunks
    
    def _collate_fn(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        
        将问答对切分成多个文本块，最后一块包含掩码
        """
        questions = [s['question'] for s in samples]
        answers = [s['answer'] for s in samples]
        
        # 拼接问题和答案
        full_texts = [q + a for q, a in zip(questions, answers)]
        
        # Tokenize
        encoded = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.chunk_size * self.max_chunks,  # 允许多块
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids']  # (batch, total_len)
        attention_mask = encoded['attention_mask']
        
        # 计算每个样本的问题长度（掩码起点）
        question_encoded = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        question_lengths = [
            len(q_ids[q_mask.bool()]) 
            for q_ids, q_mask in zip(question_encoded['input_ids'], 
                                     question_encoded['attention_mask'])
        ]
        
        # 对每个样本分别处理
        all_batch_chunks = []
        all_batch_labels = []
        
        for i in range(len(full_texts)):
            sample_tokens = input_ids[i]  # (total_len,)
            sample_question_len = question_lengths[i]
            
            # 切分成多个块
            chunks = self._split_into_chunks(sample_tokens)
            
            # 为每个块创建 labels
            chunk_labels = []
            current_pos = 0
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_start = chunk_idx * self.chunk_size
                chunk_end = (chunk_idx + 1) * self.chunk_size
                
                # 判断这个块是否包含问题
                if chunk_end <= sample_question_len:
                    # 这个块完全是问题 → 无掩码，loss 忽略
                    labels = torch.full_like(chunk, -100)
                elif chunk_start < sample_question_len:
                    # 这个块部分包含问题（跨块情况）
                    # 问题部分无掩码，答案部分掩码
                    labels = chunk.clone()
                    mask_start_in_chunk = sample_question_len - chunk_start
                    labels[:mask_start_in_chunk] = -100  # 问题部分忽略
                    chunk[mask_start_in_chunk:] = self.tokenizer.mask_token_id
                else:
                    # 这个块完全是答案 → 全部掩码
                    labels = chunk.clone()
                    chunk[:] = self.tokenizer.mask_token_id
                
                chunk_labels.append(labels)
            
            all_batch_chunks.append(chunks)
            all_batch_labels.append(chunk_labels)
        
        # 注意：由于每个样本的块数可能不同，这里需要特殊处理
        # 简单方案：返回 list，由模型处理
        return {
            'chunks_list': all_batch_chunks,      # List[List[Tensor]]
            'labels_list': all_batch_labels,      # List[List[Tensor]]
            'attention_mask': attention_mask,
            'metadata': {
                'type': 'qa',
                'questions': questions,
                'answers': answers,
                'question_lengths': question_lengths,
                'num_chunks_per_sample': [len(chunks) for chunks in all_batch_chunks]
            }
        }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self.dataloader:
            yield self._collate_fn(batch)
    
    def __len__(self):
        return len(self.dataloader)


def create_data_loader(
    data_type: str,
    file_paths: List[str],
    batch_size: int = 4,
    chunk_size: int = 1024,
    overlap: int = 128,
    mask_prob: float = 0.15,
    num_workers: int = 4,
    max_chunks: Optional[int] = None,
    max_samples: Optional[int] = None,
    **kwargs
):
    """
    创建数据加载器的工厂函数
    
    Args:
        data_type: 数据类型
            - 'simple': 简单文本补全（预训练）
            - 'qa': 问答（指令微调）
            - 'long_text': 长文本生成（待完善）
        file_paths: 数据文件路径列表
        batch_size: 批次大小
        chunk_size: 文本块大小
        overlap: 重叠大小
        mask_prob: 掩码概率（仅 simple 模式）
        num_workers: DataLoader 工作线程数
        max_chunks: 最大文本块数（仅 simple 模式）
        max_samples: 最大样本数（仅 qa 模式）
        **kwargs: 其他参数
    
    Returns:
        对应的数据加载器实例
    
    使用示例：
        # 预训练
        loader = create_data_loader(
            data_type='simple',
            file_paths=['wiki.txt', 'books.txt'],
            batch_size=4,
            mask_prob=0.15
        )
        
        # 指令微调
        loader = create_data_loader(
            data_type='qa',
            file_paths=['instructions.jsonl'],
            batch_size=4
        )
    """
    
    if data_type == 'simple':
        return SimpleTextLoader(
            file_paths=file_paths,
            batch_size=batch_size,
            chunk_size=chunk_size,
            overlap=overlap,
            mask_prob=mask_prob,
            num_workers=num_workers,
            max_chunks=max_chunks
        )
    
    elif data_type == 'qa':
        return QALoader(
            file_paths=file_paths,
            batch_size=batch_size,
            chunk_size=chunk_size,
            num_workers=num_workers,
            max_samples=max_samples
        )
    
    elif data_type == 'long_text':
        raise NotImplementedError(
            "LongTextLoader 尚未实现，敬请期待！"
            "\n提示：这个类型用于长文本自回归生成训练，"
            "需要支持多块输入和逐轮训练。"
        )
    
    else:
        raise ValueError(
            f"未知的数据类型：{data_type}\n"
            f"支持的类型：'simple', 'qa', 'long_text'"
        )


def create_qa_test_data(output_dir="test_data"):
    """创建测试用的问答数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    samples = [
        {
            "question": "什么是人工智能？",
            "answer": "人工智能（Artificial Intelligence）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
        },
        {
            "question": "如何学习 Python 编程？",
            "answer": "学习 Python 可以从以下几个方面入手：1. 学习基础语法，包括变量、数据类型、控制流等；2. 练习编写小程序，如计算器、猜数字游戏等；3. 学习常用库，如 numpy、pandas、matplotlib 等；4. 参与实际项目，通过实践提升技能；5. 阅读优秀代码，学习最佳实践。"
        },
        {
            "question": "解释一下机器学习。",
            "answer": "机器学习是人工智能的核心领域，它使计算机能够从数据中学习而无需显式编程。机器学习通过分析大量数据来发现模式和规律，然后利用这些模式进行预测或决策。主要类型包括：监督学习（如分类、回归）、无监督学习（如聚类、降维）、强化学习（通过与环境交互学习策略）等。"
        }
    ]
    
    qa_file = output_dir / "instructions.jsonl"
    with open(qa_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"问答测试数据已生成到 {qa_file}")
    return [str(qa_file)]


class InstructionTuningDataset(Dataset):
    """指令微调数据集"""
    
    def __init__(self, file_paths: List[str], tokenizer, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.samples = []
        
        # 读取问答数据
        for fp in file_paths:
            fp = Path(fp)
            if not fp.exists():
                continue
            
            with open(fp, 'r', encoding='utf-8') as f:
                if fp.suffix == '.jsonl':
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'question' in data and 'answer' in data:
                                # 预先分词
                                full_text = data['question'] + data['answer']
                                tokens = tokenizer.encode(full_text)
                                
                                # 找到问题和答案的分界点
                                question_len = len(tokenizer.encode(data['question']))
                                
                                self.samples.append({
                                    'tokens': tokens,
                                    'question_len': question_len,
                                    'metadata': data
                                })
                        except:
                            continue
        
        if max_samples:
            self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class InstructionTuningDataLoader:
    """指令微调数据加载器（多轮自回归生成）"""
    
    def __init__(
        self,
        file_paths: List[str],
        batch_size: int = 2,
        chunk_size: int = 1024,
        max_question_chunks: int = 5,
        max_answer_chunks: int = 10,
        num_workers: int = 2
    ):
        # 加载本地分词器
        tokenizer_path = Path(__file__).parent / 'bert-base-chinese'
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"未找到分词器文件夹：{tokenizer_path}\n"
                "请确保 bert-base-chinese 文件夹与 text_qa_data.py 在同一目录"
            )
        
        self.tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path))
        self.chunk_size = chunk_size
        self.max_question_chunks = max_question_chunks
        self.max_answer_chunks = max_answer_chunks
        
        self.dataset = InstructionTuningDataset(file_paths, self.tokenizer)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _split_into_chunks(self, tokens: List[int]) -> List[List[int]]:
        """将 token 序列切分成多个文本块"""
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk = tokens[i:i+self.chunk_size]
            if len(chunk) < self.chunk_size:
                chunk += [self.tokenizer.pad_token_id] * (self.chunk_size - len(chunk))
            chunks.append(chunk)
            if len(chunks) >= self.max_question_chunks + self.max_answer_chunks:
                break
        return chunks
    
    def _collate_fn(self, samples: List[Dict]) -> Dict:
        """
        整理批次数据
        
        返回一个列表，包含这个 batch 所有样本的所有训练轮次
        """
        all_rounds_data = []
        
        for sample in samples:
            tokens = sample['tokens']
            question_len = sample['question_len']
            
            # 切分成块
            chunks = self._split_into_chunks(tokens)
            num_chunks = len(chunks)
            
            # 计算问题占多少块
            question_num_chunks = (question_len + self.chunk_size - 1) // self.chunk_size
            
            # 答案从第几块开始
            answer_start_chunk = question_num_chunks
            
            # 为每一块答案生成一轮训练数据
            for pred_chunk_idx in range(answer_start_chunk, num_chunks):
                round_data = {
                    'chunks': [],          # 所有块（问题 + 答案）
                    'labels': [],          # 每块的标签
                    'pred_chunk_idx': pred_chunk_idx,  # 当前要预测的块索引
                    'total_chunks': num_chunks,
                    'question_chunks': question_num_chunks,
                    'metadata': {
                        'sample_id': len(all_rounds_data),
                        'round': pred_chunk_idx - answer_start_chunk + 1,
                        'total_rounds': num_chunks - answer_start_chunk
                    }
                }
                
                # 添加所有块
                for i, chunk in enumerate(chunks):
                    chunk_tensor = torch.tensor(chunk, dtype=torch.long)
                    
                    if i < answer_start_chunk:
                        # 问题块：完整，loss 忽略
                        labels = torch.full_like(chunk_tensor, -100)
                        round_data['chunks'].append(chunk_tensor)
                        round_data['labels'].append(labels)
                    
                    elif i == pred_chunk_idx:
                        # 当前要预测的答案块：全部掩码
                        masked_chunk = torch.full_like(chunk_tensor, self.tokenizer.mask_token_id)
                        labels = chunk_tensor.clone()
                        # padding 位置的 loss 忽略
                        labels[chunk_tensor == self.tokenizer.pad_token_id] = -100
                        
                        round_data['chunks'].append(masked_chunk)
                        round_data['labels'].append(labels)
                    
                    elif i > pred_chunk_idx:
                        # 未来的答案块：也用掩码（但不计算 loss）
                        masked_chunk = torch.full_like(chunk_tensor, self.tokenizer.mask_token_id)
                        labels = torch.full_like(chunk_tensor, -100)
                        
                        round_data['chunks'].append(masked_chunk)
                        round_data['labels'].append(labels)
                
                all_rounds_data.append(round_data)
        
        # 返回第一个轮次（保持训练顺序）
        # 注意：实际训练中应该通过外部循环控制轮次，从第 1 轮到最后一轮逐步训练
        if all_rounds_data:
            return all_rounds_data[0]
        else:
            return None
    
    def __iter__(self):
        for batch in self.dataloader:
            if batch is not None:
                yield batch
    
    def __len__(self):
        return len(self.dataset)


# 测试代码
if __name__ == '__main__':
    print("=== 测试 SimpleTextLoader ===")
    # 这里可以添加简单的测试代码
    print("SimpleTextLoader 已就绪")
    
    print("\n=== 测试 QALoader ===")
    print("QALoader 已就绪")
    
    print("\n使用示例：")
    print("  loader = create_data_loader('simple', ['wiki.txt'], batch_size=4)")
    print("  loader = create_data_loader('qa', ['instructions.jsonl'], batch_size=4)")
