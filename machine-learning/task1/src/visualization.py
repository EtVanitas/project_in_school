"""
可视化模块

标准流程：
1. 数据探索可视化（缺失值、目标分布）
2. 模型评估可视化（对比、重要性）
3. 预测结果可视化（预测vs实际、残差分析）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os


class Visualizer:
    """可视化工具类 - 标准化流程"""
    
    def __init__(self, output_dir: str = 'results', 
                 figsize_default: tuple = (10, 6),
                 dpi: int = 300):
        """
        初始化可视化器
        
        参数:
            output_dir: 输出目录
            figsize_default: 默认图表尺寸
            dpi: 图片分辨率
        """
        self.output_dir = output_dir
        self.figsize_default = figsize_default
        self.dpi = dpi
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体和样式（Windows系统）
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        sns.set_style('whitegrid')
    
    def visualize_all(self, data_dict: Dict, model_name: str = "Model") -> List[str]:
        """
        生成所有可视化图表（统一入口）
        
        参数:
            data_dict: 包含所有必要数据的字典
                - missing_df: 缺失值分析DataFrame
                - y_train: 训练集目标变量
                - comparison_df: 模型对比DataFrame
                - feature_names: 特征名称列表
                - importances: 特征重要性数组
                - y_actual: 实际值
                - y_pred: 预测值
            model_name: 模型名称
        
        返回:
            saved_files: 保存的文件路径列表
        """
        print("\n[Generating All Visualizations]")
        
        saved_files = []
        
        # 1. 数据探索阶段
        if 'missing_df' in data_dict:
            path = self.plot_missing_analysis(data_dict['missing_df'])
            saved_files.append(path)
        
        if 'y_train' in data_dict:
            path = self.plot_target_distribution(data_dict['y_train'])
            saved_files.append(path)
        
        # 2. 模型评估阶段
        if 'comparison_df' in data_dict:
            path = self.plot_model_comparison(data_dict['comparison_df'])
            saved_files.append(path)
        
        if 'feature_names' in data_dict and 'importances' in data_dict:
            path = self.plot_feature_importance(
                data_dict['feature_names'],
                data_dict['importances']
            )
            saved_files.append(path)
        
        # 3. 预测结果阶段
        if 'y_actual' in data_dict and 'y_pred' in data_dict:
            path1 = self.plot_predictions_vs_actual(
                data_dict['y_actual'],
                data_dict['y_pred'],
                model_name
            )
            saved_files.append(path1)
            
            path2 = self.plot_residual_analysis(
                data_dict['y_actual'],
                data_dict['y_pred']
            )
            saved_files.append(path2)
        
        print(f"\n[DONE] Generated {len(saved_files)} charts")
        
        return saved_files
    
    def plot_missing_analysis(self, missing_df: pd.DataFrame, 
                             top_n: int = 15,
                             filename: str = '01_missing_analysis.png') -> str:
        """
        缺失值分析图
        
        参数:
            missing_df: 缺失值统计DataFrame
            top_n: 显示Top N特征
            filename: 输出文件名
        
        返回:
            保存的文件路径
        """
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        top_missing = missing_df.head(top_n)
        
        ax.barh(range(len(top_missing)), top_missing['缺失比例(%)'].values,
                color='salmon', edgecolor='black')
        ax.set_yticks(range(len(top_missing)))
        ax.set_yticklabels(top_missing.index, fontsize=9)
        ax.set_xlabel('Missing Ratio (%)', fontsize=12)
        ax.set_title(f'Top {top_n} Features with Most Missing Values', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # 添加数值标签
        for i, (_, row) in enumerate(top_missing.iterrows()):
            ax.text(row['缺失比例(%)'] + 1, i, f"{row['缺失比例(%)']}%",
                    va='center', fontsize=8)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] {filename}")
        return filepath
    
    def plot_target_distribution(self, y_train: pd.Series,
                                filename: str = '02_target_distribution.png') -> str:
        """
        目标变量分布图
        
        参数:
            y_train: 训练集目标变量
            filename: 输出文件名
        
        返回:
            保存的文件路径
        """
        y_log = np.log1p(y_train)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 原始分布
        axes[0].hist(y_train, bins=50, edgecolor='black', color='steelblue', alpha=0.7)
        axes[0].set_title('SalePrice Original Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Price ($)')
        axes[0].set_ylabel('Frequency')
        
        mean_price = y_train.mean()
        median_price = y_train.median()
        axes[0].axvline(mean_price, color='red', linestyle='--',
                       label=f'Mean: ${mean_price:,.0f}')
        axes[0].axvline(median_price, color='green', linestyle='--',
                       label=f'Median: ${median_price:,.0f}')
        axes[0].legend()
        
        # Log变换后
        axes[1].hist(y_log, bins=50, edgecolor='black', color='coral', alpha=0.7)
        axes[1].set_title('Log(SalePrice) Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Log(Price)')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] {filename}")
        print(f"       Original Skewness: {y_train.skew():.3f}, Log Skewness: {y_log.skew():.3f}")
        
        return filepath
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                             filename: str = '03_model_comparison.png') -> str:
        """
        模型性能对比图
        
        参数:
            comparison_df: 模型对比DataFrame
            filename: 输出文件名
        
        返回:
            保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = comparison_df['Model'].values
        rmse_vals = comparison_df['CV RMSE'].values
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        bars = ax.bar(range(len(models)), rmse_vals, color=colors, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('CV RMSE ($)', fontsize=12)
        ax.set_title('Model Performance Comparison (Cross-Validation RMSE)', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + val*0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] {filename}")
        return filepath
    
    def plot_feature_importance(self, feature_names: List[str],
                               importances: np.ndarray,
                               top_n: int = 20,
                               filename: str = '04_feature_importance.png') -> str:
        """
        特征重要性图
        
        参数:
            feature_names: 特征名称列表
            importances: 特征重要性数组
            top_n: 显示Top N特征
            filename: 输出文件名
        
        返回:
            保存的文件路径
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        
        ax.barh(range(top_n), top_features['importance'].values,
                color='steelblue', edgecolor='black')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features['feature'].values, fontsize=9)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] {filename}")
        print(f"       Top 10 Important Features:")
        print(top_features.head(10)[['feature', 'importance']].to_string(index=False))
        
        return filepath
    
    def plot_predictions_vs_actual(self, y_actual: np.ndarray, 
                                  y_pred: np.ndarray,
                                  model_name: str = "Model",
                                  filename: str = '05_predictions_vs_actual.png') -> str:
        """
        预测值vs实际值散点图
        
        参数:
            y_actual: 实际值
            y_pred: 预测值
            model_name: 模型名称
            filename: 输出文件名
        
        返回:
            保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(y_actual, y_pred, alpha=0.6, s=50,
                   color='steelblue', edgecolors='navy')
        
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--',
                linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price ($)', fontsize=12)
        ax.set_ylabel('Predicted Price ($)', fontsize=12)
        ax.set_title(f'{model_name} - Predicted vs Actual', fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] {filename}")
        return filepath
    
    def plot_residual_analysis(self, y_actual: np.ndarray, 
                              y_pred: np.ndarray,
                              filename: str = '06_residual_analysis.png') -> str:
        """
        残差分析图
        
        参数:
            y_actual: 实际值
            y_pred: 预测值
            filename: 输出文件名
        
        返回:
            保存的文件路径
        """
        residuals = y_actual - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 残差分布
        axes[0].hist(residuals, bins=50, edgecolor='black', color='coral', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Residual ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        
        # 残差vs预测值
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=50,
                       color='steelblue', edgecolors='navy')
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Price ($)', fontsize=12)
        axes[1].set_ylabel('Residual ($)', fontsize=12)
        axes[1].set_title('Residual vs Predicted', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] {filename}")
        print(f"       Residual Stats: Mean=${residuals.mean():,.2f}, "
              f"Std=${residuals.std():,.2f}, Median=${np.median(residuals):,.2f}")
        
        return filepath
