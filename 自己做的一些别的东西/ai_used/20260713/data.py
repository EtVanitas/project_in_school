# data.py
import os
import json
import glob
import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class StreamingMLMDataset(IterableDataset):
    """
    流式 MLM 数据集。
    使用 Qwen3 完整词表，从 data/ 下所有数据文件流式读取、编码和 MLM 掩码。
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        max_len: int = 1024,
        mlm_prob: float = 0.15,
    ):
        self.max_len = max_len
        self.mlm_prob = mlm_prob

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # 添加 [MASK] 特殊 token
        self.tokenizer.add_special_tokens({"mask_token": "<mask>"})

        # Qwen3 完整词表参数
        self.vocab_size = len(self.tokenizer)   # 151670
        self.mask_token_id = self.tokenizer.mask_token_id  # 151669
        self.pad_token_id = self.tokenizer.pad_token_id    # 151643

        # 构建数据源列表
        self._build_sources(data_dir)

    def _build_sources(self, data_dir: str):
        """发现并注册所有数据源"""
        self.sources = []

        # JSONL 数据源
        jsonl_configs = [
            ("baike_qa2019/baike_qa_train.json", "baike_qa"),
            ("baike_qa2019/baike_qa_valid.json", "baike_qa"),
            ("translation2019zh/translation2019zh_train.json", "translation"),
            ("translation2019zh/translation2019zh_valid.json", "translation"),
            ("webtext2019zh/web_text_zh_train.json", "webtext"),
            ("webtext2019zh/web_text_zh_valid.json", "webtext"),
            ("webtext2019zh/web_text_zh_testa.json", "webtext"),
        ]
        for rel_path, stype in jsonl_configs:
            full = os.path.join(data_dir, rel_path)
            if os.path.exists(full):
                self.sources.append(("jsonl", full, stype))

        # wiki_zh_2019 文本文件
        wiki_files = sorted(glob.glob(
            os.path.join(data_dir, "wiki_zh_2019", "wiki_zh", "*", "*")))
        if wiki_files:
            self.sources.append(("wiki_dir", wiki_files, "wiki_zh"))

        # Wikipedia parquet
        parquet_files = sorted(glob.glob(
            os.path.join(data_dir, "wikipedia", "*.parquet")))
        if parquet_files:
            self.sources.append(("parquet_dir", parquet_files, "parquet"))

        total_mb = 0
        print(f"[StreamingMLMDataset] 数据源:")
        for fmt, path_or_files, stype in self.sources:
            if fmt == "jsonl":
                mb = os.path.getsize(path_or_files) / (1024*1024)
                total_mb += mb
                print(f"  [{stype}] {os.path.basename(path_or_files)} ({mb:.0f} MB)")
            elif fmt == "wiki_dir":
                print(f"  [wiki_zh] {len(path_or_files)} 个文件")
            elif fmt == "parquet_dir":
                total_mb_files = sum(os.path.getsize(f) for f in path_or_files) / (1024*1024)
                total_mb += total_mb_files
                print(f"  [parquet] {len(path_or_files)} 个文件 ({total_mb_files:.0f} MB)")
            elif fmt == "txt":
                mb = os.path.getsize(path_or_files) / (1024*1024)
                total_mb += mb
                print(f"  [txt] {os.path.basename(path_or_files)} ({mb:.0f} MB)")
        print(f"  总计: ~{total_mb:.0f} MB")

    def _mlm_mask(self, input_ids: torch.Tensor) -> tuple:
        """
        BERT 风格的 MLM 掩码。
        返回 (masked_input_ids, labels)
        """
        labels = input_ids.clone()
        prob = torch.full(labels.shape, self.mlm_prob)
        # 防止 mask pad_token
        special_mask = (input_ids == self.pad_token_id)
        prob.masked_fill_(special_mask, value=0.0)

        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100

        # 80% → [MASK]
        mask_replace = masked & (torch.rand(labels.shape) < 0.8)
        input_ids[mask_replace] = self.mask_token_id

        # 10% → 随机 token
        rand_replace = masked & ~mask_replace & (torch.rand(labels.shape) < 0.5)
        random_tokens = torch.randint(0, self.vocab_size, labels.shape)
        input_ids[rand_replace] = random_tokens[rand_replace]

        # 剩下 10% 保持原样
        return input_ids, labels

    def _extract_texts(self, record: dict, source_type: str):
        """从不同格式的记录中提取文本"""
        fields = {
            "baike_qa": ["title", "desc", "answer"],
            "translation": ["chinese", "english"],
            "webtext": ["title", "desc", "content"],
            "wiki_zh": ["title", "text"],
            "parquet": ["title", "text"],
            "plain": ["text"],
        }
        texts = []
        for key in fields.get(source_type, []):
            v = record.get(key)
            if v and isinstance(v, str):
                texts.append(v)
        # 对于纯文本，直接把整行当文本
        if isinstance(record, str):
            texts.append(record)
        return texts

    def _iter_jsonl(self, path: str, stype: str):
        """迭代 JSONL 文件"""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for text in self._extract_texts(record, stype):
                    yield text

    def _iter_wiki(self, files: list):
        """迭代 wiki_zh_2019 文件"""
        for wf in files:
            with open(wf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for text in self._extract_texts(record, "wiki_zh"):
                        yield text

    def _iter_parquet(self, files: list):
        """迭代 parquet 文件"""
        import pyarrow.parquet as pq
        for pf in files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            for _, row in df.iterrows():
                for text in self._extract_texts(row.to_dict(), "parquet"):
                    yield text

    def _iter_txt(self, path: str):
        """迭代纯文本文件"""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield line

    def __iter__(self):
        for fmt, path_or_files, stype in self.sources:
            if fmt == "jsonl":
                iterator = self._iter_jsonl(path_or_files, stype)
            elif fmt == "wiki_dir":
                iterator = self._iter_wiki(path_or_files)
            elif fmt == "parquet_dir":
                iterator = self._iter_parquet(path_or_files)
            elif fmt == "txt":
                iterator = self._iter_txt(path_or_files)
            else:
                continue

            for text in iterator:
                if not text or not isinstance(text, str):
                    continue

                # 编码
                ids = self.tokenizer.encode(text, max_length=None, add_special_tokens=False)
                if len(ids) < 10:
                    continue

                # 按 max_len 分块（滑动窗口方式）
                stride = self.max_len // 2
                for start in range(0, len(ids), stride):
                    chunk = ids[start:start + self.max_len - 2]
                    if len(chunk) < 10:
                        break

                    # 转为 tensor
                    input_ids = torch.tensor(chunk, dtype=torch.long)

                    # padding 到固定长度
                    seq_len = input_ids.shape[0]
                    if seq_len < self.max_len:
                        pad_len = self.max_len - seq_len
                        input_ids = torch.cat([
                            input_ids,
                            torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                        ])

                    # MLM 掩码
                    masked_ids, labels = self._mlm_mask(input_ids.clone())

                    yield {
                        "input_ids": masked_ids,
                        "labels": labels,
                    }

    def __len__(self):
        """近似长度，用于进度条估算（实际迭代以数据量为准）"""
        return 10_000_000


# 保持向后兼容：别名
MLMDataset = StreamingMLMDataset