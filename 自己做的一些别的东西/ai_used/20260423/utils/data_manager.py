"""组件管理器 - 统一管理训练数据的加载、保存和路径管理

职责：
- 文本数据和基准数据的加载
- JSONL数据集的加载和tokenization
- 轨迹数据的批量保存和加载
- 路径记录的持久化
- Checkpoint的保存和加载(断点续训)
- 训练完成后数据清理（trajectories, path_records）
"""

import os
import json
import glob
import torch
from typing import List, Dict, Tuple
from config import PathConfig


class DataManager:
    """组件管理器 - 统一管理所有训练数据的存取"""
    
    def __init__(self):
        """初始化数据管理器（直接使用全局PathConfig）"""
        # 使用全局 PathConfig
        self.path_config = PathConfig()
        
        # 数据目录（从 PathConfig 获取）
        self.text_dir = self.path_config.text_dir
        self.base_dir = self.path_config.base_dir
        self.traj_dir = self.path_config.trajectory_files_dir
        self.path_records_file = self.path_config.path_records_file
        
        # 确保数据目录存在
        self._ensure_data_dirs()
    
    def _ensure_data_dirs(self):
        """确保数据目录存在（仅数据相关，不包括模型目录）"""
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)
    
    # ==================== 数据加载 ====================
    
    def load_text_and_base(self, text_path: str, base_path: str, device: torch.device):
        """加载单个样本的文本和基准数据"""
        input_ids = torch.load(text_path, weights_only=True).to(device)
        base_data = torch.load(base_path, weights_only=False)
        return input_ids, base_data
    
    def load_batch_from_folders(self, sample_files: List[str], text_folder: str, 
                                base_folder: str, device: torch.device):
        """从指定文件夹批量加载样本（Stage1A 专用）"""
        batch_input_ids = []
        batch_base_data = []
        batch_sample_names = []
        
        for sample_file in sample_files:
            sample_name = sample_file.replace('.pt', '')
            text_path = os.path.join(text_folder, sample_file)
            base_path = os.path.join(base_folder, sample_file)
            
            if not os.path.exists(text_path) or not os.path.exists(base_path):
                continue
            
            input_ids, base_data = self.load_text_and_base(text_path, base_path, device)
            
            batch_input_ids.append(input_ids)
            batch_base_data.append(base_data)
            batch_sample_names.append(sample_name)
        
        if not batch_input_ids:
            return None, None, None
        
        # Stack成batch
        batch_input_ids = torch.stack(batch_input_ids)
        return batch_input_ids, batch_base_data, batch_sample_names

    def load_batch_samples(self, sample_names: List[str], len_key: str, device: torch.device):
        """批量加载相同长度的样本数据（Stage1B 专用）"""
        batch_input_ids = []
        batch_base_data = []
        
        for sample_name in sample_names:
            text_file = os.path.join(self.text_dir, len_key, f'{sample_name}.pt')
            base_file = os.path.join(self.base_dir, len_key, f'{sample_name}.pt')
            
            if not os.path.exists(text_file) or not os.path.exists(base_file):
                continue
            
            input_ids, base_data = self.load_text_and_base(text_file, base_file, device)
            
            batch_input_ids.append(input_ids)
            batch_base_data.append(base_data)
        
        if not batch_input_ids:
            return None, None, None
        
        # Stack成batch
        batch_input_ids = torch.stack(batch_input_ids)
        return batch_input_ids, batch_base_data
    
    # ==================== 轨迹数据管理 ====================
    
    def _load_trajectory_batch(self, traj_file: str):
        """加载单个轨迹批次文件（内部辅助方法）"""
        if not os.path.exists(traj_file):
            return []
        
        try:
            batch_data = torch.load(traj_file, weights_only=False)
            # 兼容新旧格式（list vs tuple）
            if isinstance(batch_data, list):
                return batch_data
            else:
                return [batch_data]
        except Exception as e:
            print(f"警告: 加载 {traj_file} 失败: {e}")
            return []
    
    def save_trajectory_batch(self, trajectories: List[Dict], batch_idx: int):
        """批量保存轨迹数据"""
        if not trajectories:
            return None
        
        os.makedirs(self.traj_dir, exist_ok=True)
        traj_file = os.path.join(self.traj_dir, f'batch_{batch_idx:06d}.pt')
        torch.save(trajectories, traj_file)
        
        print(f"  已保存 {len(trajectories)} 条轨迹到 {traj_file}")
        return traj_file
    
    def load_all_trajectories(self) -> List[Dict]:
        """加载所有轨迹文件"""
        if not os.path.exists(self.traj_dir):
            print("警告: 轨迹目录不存在")
            return []
        
        traj_files = glob.glob(os.path.join(self.traj_dir, '*.pt'))
        if not traj_files:
            print("警告: 没有找到轨迹文件")
            return []
        
        all_triples = []
        for traj_file in traj_files:
            triples = self._load_trajectory_batch(traj_file)
            all_triples.extend(triples)
        
        print(f"加载了 {len(all_triples)} 条三元组")
        return all_triples
    
    # ==================== 路径记录管理 ====================
    
    def load_path_records(self) -> Dict:
        """加载路径记录"""
        if not os.path.exists(self.path_records_file):
            print("未找到路径记录文件，初始化空记录")
            return {}
        
        try:
            with open(self.path_records_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换回 tuple 键
            path_records = {}
            for key_str, samples in data.items():
                path_tuple = tuple(eval(key_str))
                path_records[path_tuple] = samples
            
            print(f"加载路径记录：{len(path_records)} 条路径")
            return path_records
        except Exception as e:
            print(f"加载路径记录失败: {e}，初始化空记录")
            return {}
    
    def save_path_records(self, path_records: Dict):
        """保存路径记录"""
        # 转换 tuple 键为字符串
        data = {}
        for path_tuple, samples in path_records.items():
            key_str = str(list(path_tuple))
            data[key_str] = samples
        
        os.makedirs(os.path.dirname(self.path_records_file), exist_ok=True)
        
        with open(self.path_records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"保存路径记录：{len(path_records)} 条路径")
    
    # ==================== 文件列表管理 ====================
    
    def get_length_folders(self) -> List[str]:
        """获取所有长度文件夹（len_*）"""
        if not os.path.exists(self.text_dir):
            return []
        
        folders = [f for f in os.listdir(self.text_dir) if f.startswith('len_')]
        return sorted(folders)
    
    def get_sample_files(self, folder_path: str, extension: str = '.pt') -> List[str]:
        """获取指定文件夹下的样本文件"""
        if not os.path.exists(folder_path):
            return []
        
        return [f for f in os.listdir(folder_path) if f.endswith(extension)]
    
    def load_random_sample(self, device: torch.device) -> Tuple[int, torch.Tensor]:
        """随机加载一个样本（用于 Stage2）"""
        import random
        
        if not os.path.exists(self.text_dir):
            raise FileNotFoundError(f"文本数据目录不存在：{self.text_dir}")
        
        # 随机选择一个长度文件夹
        length_folders = self.get_length_folders()
        if not length_folders:
            raise FileNotFoundError("没有找到 len_* 文件夹")
        
        selected_folder = random.choice(length_folders)
        seq_len = int(selected_folder.replace('len_', ''))
        
        # 随机选择一个文本文件
        folder_path = os.path.join(self.text_dir, selected_folder)
        text_files = self.get_sample_files(folder_path)
        if not text_files:
            raise FileNotFoundError(f"文本目录为空：{folder_path}")
        
        selected_file = random.choice(text_files)
        
        # 加载 token IDs
        text_file = os.path.join(folder_path, selected_file)
        input_ids = torch.load(text_file, weights_only=True).to(device)
        
        return seq_len, input_ids
    
    # ==================== JSONL 数据集管理 ====================
    
    def load_jsonl_dataset(self, jsonl_path: str) -> List[Dict]:
        """加载JSONL数据集,返回包含input/output的列表"""
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
    
    def save_checkpoint(self, processed_count: int, checkpoint_file: str, 
                       avg_loss: float = None):
        """保存处理进度到checkpoint文件"""
        from datetime import datetime
        
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
    
    def load_checkpoint(self, checkpoint_file: str) -> int:
        """加载checkpoint,返回已处理的样本数"""
        if not os.path.exists(checkpoint_file):
            print(f"未找到checkpoint文件: {checkpoint_file},从头开始训练")
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
    
    # ==================== 数据清理 ====================
    
    def cleanup_training_data(self, remove_trajectories: bool = True, 
                             remove_path_records: bool = True):
        """清理训练数据（Stage1B 和 Stage2 训练完成后调用）"""
        if remove_trajectories and os.path.exists(self.traj_dir):
            import shutil
            try:
                shutil.rmtree(self.traj_dir)
                print(f"已删除轨迹目录：{self.traj_dir}")
            except Exception as e:
                print(f"无法删除轨迹目录：{e}")
        
        if remove_path_records and os.path.exists(self.path_records_file):
            try:
                os.remove(self.path_records_file)
                print(f"已删除路径记录：{self.path_records_file}")
            except Exception as e:
                print(f"无法删除路径记录：{e}")
