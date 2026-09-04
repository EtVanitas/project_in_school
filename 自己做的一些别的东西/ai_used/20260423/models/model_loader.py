"""模型加载器模块
负责加载 Qwen3.5-2B 并分层
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ModelConfig, LoRAConfig
from models.networks import LoRAModuleWrapper


class LayeredModelLoader(nn.Module):
    """将 Qwen3.5-2B 的 24 层解耦为独立组件，所有层均包含 LoRA 适配器"""
    
    def __init__(self):
        super().__init__()
        # 使用全局默认配置
        self.model_config = ModelConfig()
        self.lora_config = LoRAConfig()
        self.layers = nn.ModuleList()
        self.embed_tokens = None
        self.norm = None
        self.lm_head = None
        self.tokenizer = None
        self.model = None
        
        # 自动加载模型
        self.load_from_local()
    
    def load_from_local(self):
        """从本地路径加载 Qwen3.5-2B 模型并分层"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        
        # 加载模型
        kwargs = {
            "device_map": "cuda:0",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "dtype": torch.bfloat16 if self.model_config.dtype == "bfloat16" else torch.float32
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_path,
            **kwargs
        )
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_path,
            trust_remote_code=True
        )
        
        # 打印模型信息
        config = getattr(self.model.config, 'text_config', self.model.config)
        arch = getattr(self.model.config, 'architectures', ['Unknown'])[0] if hasattr(self.model.config, 'architectures') and self.model.config.architectures else 'Unknown'
        print(f"架构：{arch}, 层数：{config.num_hidden_layers}, 维度：{config.hidden_size}, 词表：{config.vocab_size}")
        
        # 提取组件
        self.embed_tokens = self.model.model.embed_tokens
        self.norm = self.model.model.norm
        
        # LM Head
        if getattr(self.model.config, 'tie_word_embeddings', False):
            self.lm_head = self.model.get_output_embeddings()
        else:
            self.lm_head = self.model.lm_head
        
        # 为所有层添加 LoRA
        for layer in self.model.model.layers:
            wrapped = LoRAModuleWrapper(layer)
            self.layers.append(wrapped)
        
        # [关键修复] 冻结所有非 LoRA 参数，只训练 LoRA
        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        
        # 打印摘要
        lora_params = sum(p.numel() for name, p in self.named_parameters() if 'lora_' in name and p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"完成：{len(self.layers)}层")
        print(f"  - 总参数: {total_params/1e6:.2f}M")
        print(f"  - LoRA 可训练参数: {lora_params/1e6:.2f}M ({lora_params/total_params*100:.2f}%)")
    
    def get_trainable_params(self):
        """获取所有可训练参数（LoRA 参数）"""
        return [p for name, p in self.named_parameters() if 'lora_' in name and p.requires_grad]
