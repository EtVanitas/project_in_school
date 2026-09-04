"""Alice 模型主文件"""

import torch
import torch.nn as nn
from typing import List
from config import Config
from alice_components import create_activates_and_reasons, ActivateTester
from activated_pool import ActivatedPool
from remain_manager import RemainManager
from long_term_memory import LongTermMemory


class MainModel(nn.Module):
    """
    Alice 主模型
    """
    
    def __init__(self):
        """初始化模型和组件"""
        super().__init__()
        self.m, self.n = Config.model.M, Config.model.N
        self.max_iterations = Config.flow.MAX_ITERATIONS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建核心组件
        self.activates, self.reasons = create_activates_and_reasons()
        self.activate_tester = ActivateTester(self.activates)
        self.to(self.device)
        
        # 状态管理器
        self.activated_pool = ActivatedPool(max_size=Config.flow.ACTIVATED_POOL_MAX_SIZE)
        self.remain_manager = RemainManager(num_activates=Config.model.NUM_ACTIVATE_CLASSES, m=self.m, n=self.n)
        self.long_term_memory = LongTermMemory(m=self.m, n=self.n)
        
        # 输出管理
        self.output_lists: List[torch.Tensor] = []
    
    def forward(self, initial_x: torch.Tensor, epoch: int = None):
        """
        前向传播（线性多轮迭代）
        
        Args:
            initial_x: (batch_size, m, n) - 3D 输入张量
            epoch: 当前训练 epoch（预留参数，暂未使用）
        
        Returns:
            output_lists: List[Tensor[m, n]] - Type3 Reason 的输出列表
        """
        # 初始化
        batch_size = initial_x.size(0)

        self.activated_pool.clear()
        self.output_lists.clear()
        self.remain_manager.clear()
        short_term_updates = []  # 用于更新短期记忆
        long_term_records = []   # 用于记录长期轨迹
        
        print(f"\n[Forward] 开始处理，batch_size={batch_size}")
        
        # ========== 第 1 轮：处理初始输入 ==========
        # 重置计数器
        self.stats_x1 = 0
        self.stats_x4 = 0
        self.stats_x6 = 0
        
        x_list = [initial_x[i] for i in range(batch_size)]
        self._process_inputs(x_list, short_term_updates, long_term_records)
        
        # 第 1 轮结束，更新短期记忆
        if short_term_updates:
            self.remain_manager.update_batch(short_term_updates)
            print(f"[记忆更新] 短期记忆：{len(short_term_updates)} 条")
            short_term_updates = []  # 清空，准备下一轮
        
        # 打印第 1 轮统计
        print(f"[第 1 轮统计] x1={self.stats_x1}个，x4={self.stats_x4}个，x6={self.stats_x6}个")
        
        self.remain_manager.decay_all()
        
        # ========== 第 2-N 轮：循环处理上一轮的输出 ==========
        for iteration in range(1, self.max_iterations):
            # 检查输出是否已达上限
            if len(self.output_lists) >= Config.flow.OUTPUT_LIST_MAX_SIZE:
                print(f"[终止] 输出已达上限 ({len(self.output_lists)}), 提前结束")
                break
            
            # 从激活池取出上一轮的所有输出
            previous_outputs = self.activated_pool.clear()
            if len(previous_outputs) == 0:
                print(f"[终止] 第{iteration}轮：无新激活")
                break
            
            # 提取 x 值（忽略之前的 act_idx 和 score）
            x_values = [item[0] for item in previous_outputs]
            
            # 重置计数器
            self.stats_x1 = 0
            self.stats_x4 = 0
            self.stats_x6 = 0
            
            print(f"[第{iteration}轮] 处理 {len(x_values)} 个上一轮输出")
            self._process_inputs(x_values, short_term_updates, long_term_records)
            
            # 每轮结束，更新短期记忆
            if short_term_updates:
                self.remain_manager.update_batch(short_term_updates)
                print(f"[记忆更新] 短期记忆：{len(short_term_updates)} 条")
            
            self.remain_manager.decay_all()
            
            # 打印本轮统计
            print(f"[第{iteration}轮统计] x1={self.stats_x1}个，x4={self.stats_x4}个，x6={self.stats_x6}个")
            
            # 清空本轮的短期更新记录，准备下一轮
            short_term_updates = []
        
        # ========== 整个 forward 结束，记录长期记忆 ==========
        if Config.flow.ENABLE_LONG_TERM_MEMORY and long_term_records:
            self.long_term_memory.record_batch(long_term_records)  # 批量记录（自动保存）
            print(f"[记忆更新] 长期记忆：{len(long_term_records)} 条")
        
        print(f"[Forward] 完成，共生成 {len(self.output_lists)} 个输出\n")
        
        return self.output_lists
    
    def _process_inputs(self, x_list, updates, records):
        """
        处理输入列表（通用函数）
        """
        for x in x_list:
            # 遍历 72 个 activate 类别
            for act_idx in range(Config.model.NUM_ACTIVATE_CLASSES):
                # Step 1: 计算激活值
                score = self.activate_tester.compute_activation(x, act_idx)
                
                # ========== 情况 A: 高激活（score >= 0.7）→ x1 ==========
                if score >= Config.flow.HIGH_THRESHOLD:
                    self.stats_x1 += 1  # 统计 x1
                    self._handle_high_activation(
                        x=x, act_idx=act_idx, score=score,
                        input_name="x",
                        updates=updates, records=records
                    )
                
                # ========== 情况 B: 低激活（0.5 <= score < 0.7）→ x2 ==========
                elif Config.flow.LOW_THRESHOLD <= score < Config.flow.HIGH_THRESHOLD:
                    # ---- 短期记忆分支：计算 x4 的激活值 ----
                    remain_short = self.remain_manager.get_remain(act_idx, device=x.device)
                    if x.dtype == torch.float16:
                        remain_short = remain_short.to(torch.float16)
                    x2_with_short = x + remain_short.detach()
                    
                    score_short = self.activate_tester.compute_activation(x2_with_short, act_idx)
                    if score_short >= Config.flow.HIGH_THRESHOLD:
                        self.stats_x4 += 1  # 统计 x4
                        self._handle_high_activation(
                            x=x2_with_short, act_idx=act_idx, score=score_short,
                            input_name="x+short",
                            updates=updates, records=records,
                            record_x=x, record_remain=remain_short
                        )
                    
                    # ---- 长期记忆分支：计算 x6 的激活值 ----
                    if Config.flow.ENABLE_LONG_TERM_MEMORY:
                        remain_long = self.long_term_memory.query(x_query=x, topk=Config.flow.LONG_TERM_TOPK)
                        if x.dtype == torch.float16:
                            remain_long = remain_long.to(torch.float16)
                        x2_with_long = x + remain_long.detach()
                        
                        score_long = self.activate_tester.compute_activation(x2_with_long, act_idx)
                        if score_long >= Config.flow.HIGH_THRESHOLD:
                            self.stats_x6 += 1  # 统计 x6
                            self._handle_high_activation(
                                x=x2_with_long, act_idx=act_idx, score=score_long,
                                input_name="x+long",
                                updates=updates, records=records,
                                record_x=x, record_remain=remain_long
                            )
            # 检查输出是否已达上限
            if len(self.output_lists) >= Config.flow.OUTPUT_LIST_MAX_SIZE:
                print(f"[终止] 输出已达上限 ({len(self.output_lists)}), 提前结束")
                break
    
    def _handle_high_activation(self, x, act_idx, score, input_name, updates, records, record_x=None, record_remain=None):
        """
        处理高激活情况（score >= 0.7）的通用逻辑
        """
        # 预占位检查：池子是否已满且分数不够高
        if not self.activated_pool.can_add(act_idx, score):
            pass
        else:
            # 更新短期记忆
            updates.append((act_idx, x.detach().cpu(), score.cpu()))
        
            # # 记录轨迹：只有 score >= 0.9 才记录（长期记忆高阈值过滤）
            # if record_x is not None and record_remain is not None:
            #     if score >= Config.flow.LONG_TERM_MIN_ACTIVATION:
            #         records.append((record_x.detach().cpu(), record_remain.detach().cpu()))

            # 直接使用激活值乘以 x，然后输入 Reason
            x_scaled = score * x
            x_output = self.reasons[act_idx](x_scaled)
            self.activated_pool.add(x_output, act_idx, score)
            
            # 只有 Type3 Reason 的输出才保存到输出列表
            if act_idx in Config.model.REASON_TYPE3_INDICES:
                # 检查是否达到输出上限
                if len(self.output_lists) >= Config.flow.OUTPUT_LIST_MAX_SIZE:
                    print(f"[终止] 输出已达上限 ({Config.flow.OUTPUT_LIST_MAX_SIZE})，提前结束")
                    return self.output_lists
                self.output_lists.append(x_output)
