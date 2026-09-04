"""
Alice 模型预测测试工具 - 简化版

使用方法：
1. 运行 python test_model_prediction.py
2. 输入一句话
3. 自动掩码并显示模型预测结果
"""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import BertTokenizerFast

from config import Config
from alice_main import MainModel
from text_pretrain_data import CustomEmbedding, MLMPreprocessor


class ModelPredictor:
    """简化版模型预测器"""
    
    def __init__(self, checkpoint_path='checkpoints/pretrain_best.pt'):
        """初始化"""
        self.device = torch.device(Config.train.DEVICE)
        print(f"使用设备：{self.device}")
        
        # 1. 加载分词器
        print("\n加载分词器...")
        tokenizer_path = Path('bert-base-chinese')
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"未找到分词器文件夹：{tokenizer_path}\n"
                "请确保 bert-base-chinese 文件夹在项目根目录"
            )
        self.tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path))
        self.vocab_size = len(self.tokenizer)  # 获取词表大小
        
        # 2. 创建模型
        print("创建模型...")
        self.embedding = CustomEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=Config.model.N,
            padding_idx=self.tokenizer.pad_token_id
        ).to(self.device)
        
        self.model = MainModel().to(self.device)
        self.output_projection = nn.Linear(Config.model.N, self.vocab_size).to(self.device)
        
        # 3. 加载检查点
        if Path(checkpoint_path).exists():
            print(f"\n加载检查点：{checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            self.embedding.load_state_dict(checkpoint['embedding_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.output_projection.load_state_dict(checkpoint['projection_state_dict'])
            
            print(f"✓ 模型权重已加载")
            print(f"✓ 最佳损失：{checkpoint.get('best_loss', 'N/A')}")
        else:
            print(f"\n⚠ 警告：未找到检查点 {checkpoint_path}")
        
        # 设置为评估模式
        self.embedding.eval()
        self.model.eval()
        self.output_projection.eval()
        
        # 创建 MLM 预处理器
        self.mlm_preprocessor = MLMPreprocessor(self.tokenizer)
        
        print("\n" + "="*60)
        print("模型已准备就绪！")
        print("="*60)
    
    def predict(self, text: str):
        """
        预测被掩码的部分
        
        Args:
            text: 输入文本
        
        Returns:
            predictions: 预测结果
        """
        print(f"\n{'='*60}")
        print(f"输入：{text}")
        print(f"{'='*60}\n")
        
        # 1. 准备输入
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        M = Config.model.M
        if len(tokens) < M:
            tokens = tokens + [self.tokenizer.pad_token_id] * (M - len(tokens))
        else:
            tokens = tokens[:M]
        
        input_ids = torch.tensor([tokens], device=self.device)
        masked_input, labels = self.mlm_preprocessor(input_ids)
        embedded = self.embedding(masked_input)
        
        # 显示掩码后的文本
        display_tokens = masked_input[0].cpu().tolist()
        masked_text = self.tokenizer.decode(display_tokens)
        mask_positions = (labels[0] != -100).nonzero(as_tuple=True)[0].tolist()
        
        print(f"掩码后：{masked_text}")
        print(f"掩码位置：{mask_positions}\n")
        
        # 2. 模型前向传播
        with torch.no_grad():
            output_lists = self.model(embedded, epoch=0)
            
            if len(output_lists) == 0:
                print("⚠ 模型没有输出（激活值太低）")
                return []
            
            # 3. 累加所有输出（与训练相同）
            accumulated_logits = None
            for output in output_lists:
                projected = self.output_projection(output)
                if accumulated_logits is None:
                    accumulated_logits = projected
                else:
                    accumulated_logits = accumulated_logits + projected
            
            probabilities = torch.softmax(accumulated_logits, dim=-1)  # [M, vocab_size]
        
        # 4. 提取预测结果
        predictions = []
        original_labels = labels[0]
        
        for masked_pos in mask_positions:
            if masked_pos >= probabilities.shape[0]:
                continue
            
            pos_probs = probabilities[masked_pos]
            top_values, top_indices = torch.topk(pos_probs, k=5)
            
            pred_results = []
            for value, idx in zip(top_values.cpu().tolist(), top_indices.cpu().tolist()):
                token_str = self.tokenizer.decode([idx])
                pred_results.append({
                    'token': token_str,
                    'probability': value,
                    'percentage': f"{value*100:.2f}%"
                })
            
            correct_token_id = original_labels[masked_pos].item()
            correct_token = self.tokenizer.decode([correct_token_id]) if correct_token_id != -100 else "?"
            
            predictions.append({
                'position': masked_pos,
                'correct_token': correct_token,
                'predictions': pred_results
            })
        
        # 5. 显示结果
        self._display_predictions(predictions)
        
        return predictions
    
    def _display_predictions(self, predictions):
        """显示预测结果"""
        print(f"\n{'='*60}")
        print("预测结果")
        print(f"{'='*60}\n")
        
        for pred in predictions:
            pos = pred['position']
            correct = pred['correct_token']
            preds = pred['predictions']
            
            print(f"位置 {pos}:")
            print(f"  正确答案：【{correct}】")
            print(f"  模型预测 Top-5:")
            
            for i, p in enumerate(preds, 1):
                mark = "✓ " if p['token'] == correct else "✗ "
                print(f"    {mark}{i}. 【{p['token']}】 ({p['percentage']})")
            
            print()
        
        # 计算准确率
        valid_predictions = [p for p in predictions if p['correct_token'] != '?']
        if len(valid_predictions) > 0:
            correct_count = sum(
                1 for pred in valid_predictions 
                if pred['predictions'][0]['token'] == pred['correct_token']
            )
            accuracy = correct_count / len(valid_predictions) * 100
            
            print(f"Top-1 准确率：{correct_count}/{len(valid_predictions)} ({accuracy:.1f}%)")
    
    def interactive_mode(self):
        """交互式测试模式"""
        print("\n" + "="*60)
        print("交互式测试模式")
        print("输入文本进行测试，输入 'quit' 退出")
        print("="*60 + "\n")
        
        while True:
            try:
                # 获取用户输入
                text = input("请输入中文文本（或 'quit' 退出）: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("\n测试结束！")
                    break
                
                if len(text) < 5:
                    print("⚠ 文本太短，请至少输入 5 个字符\n")
                    continue
                
                # 进行预测
                predictions = self.predict(text)
                
                if len(predictions) == 0:
                    print("\n⚠ 没有获得预测结果，请尝试更长的文本\n")
                    continue
                
                print()
                
            except KeyboardInterrupt:
                print("\n测试中断！")
                break
            except Exception as e:
                print(f"\n❌ 错误：{e}")
                import traceback
                traceback.print_exc()
                print()


def main():
    """主函数 - 简单的输入 - 预测流程"""
    print("="*60)
    print("Alice 模型预测测试")
    print("="*60)
    
    # 1. 创建预测器
    predictor = ModelPredictor()
    
    # 2. 获取用户输入
    print("\n请输入一句话进行测试：")
    text = input("> ").strip()
    
    if not text:
        print("❌ 输入为空！")
        return
    
    # 3. 进行预测
    predictions = predictor.predict(text)
    
    if len(predictions) == 0:
        print("\n⚠ 没有获得预测结果")


if __name__ == "__main__":
    main()
