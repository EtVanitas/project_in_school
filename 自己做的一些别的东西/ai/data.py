# data.py
import os
import json
import glob
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class StreamingCLMDataset(IterableDataset):
    """
    流式自回归（CLM）数据集：固定顺序排列中文数据源（百科问答、中英翻译、中文维基），按「条」产出 token 流。
    一条记录 = 一段连贯文本（多字段拼接，字段间插 [SEP]）；训练侧对每条记录
    重置外部状态（窗口/记忆），符合"每次从头读一段话"的语义。

    断点续训：位置状态仅含 (当前源索引, 行号)。编码产出前提交位置，
    恢复后断点所在记录重读重训（偏差 ≤ 1 条），其余 token 严格不重不漏。
    """

    # 各数据源记录中的文本字段
    _FIELD_MAP = {
        "baike_qa": ["title", "desc", "answer"],
        "translation": ["chinese", "english"],
        "wiki_zh": ["title", "text"],
    }

    def __init__(self, data_dir: str, tokenizer_path: str, min_len: int = 10):
        self.min_len = min_len

        # 分词器与文本分隔符：优先 [SEP]，其次 [EOS]，都没有则用 [PAD]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # 超长记录不截断、全量编码：放开长度上限，避免 512 超长警告反复刷屏
        self.tokenizer.model_max_length = 1 << 30
        self.sep_id = (self.tokenizer.sep_token_id
                       or self.tokenizer.eos_token_id
                       or self.tokenizer.pad_token_id)
        if self.sep_id is None:
            raise ValueError("分词器缺少 sep/eos/pad token，无法分隔文本")

        self._build_sources(data_dir)
        # 数据位置状态
        self._state = {"source_idx": 0, "row": 0}

    # ---------- 断点续训 ----------

    def get_state(self) -> dict:
        """当前数据位置（供 checkpoint 保存）；恢复后断点所在记录会重训"""
        return dict(self._state)

    def resume(self, state: dict):
        """从 checkpoint 位置继续：断点所在记录重读重训（偏差 ≤ 1 条）"""
        self._state["source_idx"] = state.get("source_idx", 0)
        self._state["row"] = state.get("row", 0)

    def is_exhausted(self) -> bool:
        """数据流是否已耗尽（恢复时据此判断训练是否已完成）"""
        return self._state["source_idx"] >= len(self.sources)

    # ---------- 数据源发现（固定顺序）----------

    def _build_sources(self, data_dir: str):
        # 训练数据源：百科问答、中英翻译、中文维基（不含 webtext 与英文 wikipedia）
        files = []
        for rel, stype in [
            ("baike_qa2019/baike_qa_train.json", "baike_qa"),
            ("baike_qa2019/baike_qa_valid.json", "baike_qa"),
            ("translation2019zh/translation2019zh_train.json", "translation"),
            ("translation2019zh/translation2019zh_valid.json", "translation"),
        ]:
            path = os.path.join(data_dir, rel)
            if os.path.exists(path):
                files.append((path, stype))
        files += [(f, "wiki_zh") for f in sorted(glob.glob(
            os.path.join(data_dir, "wiki_zh_2019", "wiki_zh", "*", "*")))]

        self.sources = [("jsonl", path, stype) for path, stype in files]
        if not self.sources:
            raise RuntimeError(f"数据目录 {data_dir} 下未找到任何数据文件")

        total_mb = 0
        print("[StreamingCLMDataset] 数据源:")
        for fmt, path, stype in self.sources:
            mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [{stype}] {os.path.basename(path)} ({mb:.0f} MB)")
            total_mb += mb
        print(f"  总计: ~{total_mb:.0f} MB")
        # 粗估总 token 数（供 lr 调度总步数与进度条）：约 1 token/3 字节
        self.total_tokens = int(total_mb * 1024 * 1024 / 3)

    # ---------- 文本提取 ----------

    def _extract_texts(self, record: dict, stype: str):
        """提取单条记录中的文本字段（已保证非空 str）"""
        texts = []
        for key in self._FIELD_MAP.get(stype, []):
            v = record.get(key)
            if v and isinstance(v, str):
                texts.append(v)
        return texts

    # ---------- 逐条读取与编码 ----------

    def _iter_jsonl(self, path: str, stype: str, st: dict):
        """逐行解析 JSON，产出 (文本列表, 行号)（jsonl / wiki_zh 通用）"""
        with open(path, encoding="utf-8") as fh:
            for row, line in enumerate(fh):
                if row < st["row"]:        # 恢复时跳过已读行
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):   # 防御非对象行（list/字符串等）
                        continue
                except json.JSONDecodeError:
                    continue
                texts = self._extract_texts(record, stype)
                if texts:
                    yield texts, row

    def _iter_encoded(self, items, st: dict):
        """逐条编码：提交位置 → 编码 → 产出（每条记录一条 token 流，字段间插分隔符）。
        读取与编码同步推进，断点状态与已进入流的 token 严格同步"""
        for texts, row in items:
            st["row"] = row              # 提交位置：该条未完成，恢复时从此重训
            enc = self.tokenizer(texts, add_special_tokens=False, truncation=False)
            ids = []
            for field_ids in enc["input_ids"]:
                ids.extend(field_ids)
                ids.append(self.sep_id)
            if len(ids) >= self.min_len:  # 过短记录丢弃（无训练价值）
                yield ids

    # ---------- 主迭代 ----------

    def __iter__(self):
        st = self._state
        for si in range(st["source_idx"], len(self.sources)):
            _, path, stype = self.sources[si]
            if si > st["source_idx"]:
                st["row"] = 0               # 断点之后的源从头开始
            st["source_idx"] = si
            yield from self._iter_encoded(self._iter_jsonl(path, stype, st), st)
        # 数据流自然耗尽：标记完成（恢复时据此提示，不再重训）
        st["source_idx"] = len(self.sources)
