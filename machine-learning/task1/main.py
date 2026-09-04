"""
房价预测 - 主程序

完整的机器学习流程：
1. 数据预处理 - 基于业务理解的缺失值填充
2. 特征工程 - 标准化4步流程（构造、偏态、编码、选择）
3. 模型训练 - 自动化评估和超参数调优
4. 可视化 - 生成6张关键图表
5. 提交文件 - 生成Kaggle submission.csv
"""
import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from visualization import Visualizer


def main():
    """主函数 - 完整的房价预测流程"""
    print("="*70)
    print("房价预测机器学习项目")
    print("="*70)
    
    # 阶段1: 数据预处理
    print("\n" + "="*70)
    print("阶段1: 数据预处理")
    print("="*70)
    
    # 数据路径配置
    DATA_DIR = 'house-prices-advanced-regression-techniques'
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    
    # 检查数据文件是否存在
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[ERROR] 数据文件不存在: {train_path} 或 {test_path}")
        print("请确保数据文件在 house-prices-advanced-regression-techniques/ 目录中")
        return
    
    preprocessor = DataPreprocessor(train_path, test_path)
    
    # 分析缺失值
    print("\n【缺失值分析】")
    missing_info = preprocessor.analyze_missing()
    print(f"有缺失值的特征数: {len(missing_info)}")
    print("\nTop 10缺失最多的特征:")
    print(missing_info.head(10))
    
    # 获取最佳预处理结果
    print("\n【应用预处理策略】")
    print("  策略: 基于业务理解的智能填充")
    print("  理由: 实验对比显示三种策略性能相近（RMSE差异<0.5%）")
    print("        但Business Aware更符合业务逻辑，可解释性更强")
    X_train, X_test = preprocessor.preprocess()
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"训练集缺失值: {X_train.isnull().sum().sum()}")
    print(f"测试集缺失值: {X_test.isnull().sum().sum()}")
    
    y_train_log = np.log1p(preprocessor.y_train)
    
    # 阶段2: 特征工程
    print("\n" + "="*70)
    print("阶段2: 特征工程")
    print("="*70)
    
    engineer = FeatureEngineer()
    X_train_feat, X_test_feat, selected_features = engineer.transform(
        X_train, X_test, y_train_log
    )
    
    print(f"\n特征工程完成:")
    print(f"  最终特征数: {len(selected_features)}")
    print(f"  训练集形状: {X_train_feat.shape}")
    print(f"  测试集形状: {X_test_feat.shape}")
    
    # 阶段3: 模型训练
    print("\n" + "="*70)
    print("阶段3: 模型训练与对比")
    print("="*70)
    
    trainer = ModelTrainer()
    best_model_name, best_model, all_results = trainer.train(
        X_train_feat, y_train_log, X_test_feat
    )
    
    # 对比模型性能
    comparison_df = trainer.get_model_comparison()
    print("\n" + comparison_df.to_string(index=False))
    
    print(f"\n[OK] 最佳模型: {best_model_name}")
    
    # 阶段4: 模型评估与可视化
    print("\n" + "="*70)
    print("阶段4: 模型评估与可视化")
    print("="*70)
    
    # 划分验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_feat, y_train_log, test_size=0.2, random_state=42
    )
    
    # 在验证集上评估
    best_model.fit(X_tr, y_tr)
    y_val_pred_log = best_model.predict(X_val)
    
    # 转换回原始尺度
    y_val_pred = np.expm1(y_val_pred_log)
    y_val_actual = np.expm1(y_val)
    
    rmse_val = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
    r2_val = r2_score(y_val_actual, y_val_pred)
    
    print(f"\n验证集性能:")
    print(f"  RMSE: ${rmse_val:,.2f}")
    print(f"  R2 Score: {r2_val:.4f}")
    
    # 获取特征重要性
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_)
    
    # 生成可视化
    viz = Visualizer('results')
    
    # 准备数据字典
    data_dict = {
        'missing_df': missing_info,
        'y_train': preprocessor.y_train,
        'comparison_df': comparison_df,
        'feature_names': selected_features,
        'importances': importances,
        'y_actual': y_val_actual,
        'y_pred': y_val_pred
    }
    
    # 一键生成所有图表
    saved_files = viz.visualize_all(data_dict, model_name=best_model_name)
    
    # 阶段5: 生成提交文件
    print("\n" + "="*70)
    print("阶段5: 生成提交文件")
    print("="*70)
    
    # 使用整个训练集重新训练最佳模型
    best_model.fit(X_train_feat, y_train_log)
    
    # 预测测试集
    test_predictions_log = best_model.predict(X_test_feat)
    test_predictions = np.expm1(test_predictions_log)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'Id': preprocessor.test_ids,
        'SalePrice': test_predictions
    })
    
    submission.to_csv('results/submission.csv', index=False)
    
    print(f"[OK] 提交文件已保存: results/submission.csv")
    print(f"   预测样本数: {len(submission)}")
    print(f"   价格范围: ${test_predictions.min():,.0f} - ${test_predictions.max():,.0f}")
    
    # 总结
    print("\n" + "="*70)
    print("项目完成总结")
    print("="*70)
    print(f"最佳模型: {best_model_name}")
    print(f"验证集RMSE: ${rmse_val:,.2f}")
    print(f"验证集R2: {r2_val:.4f}")
    print(f"特征数量: {len(selected_features)}")
    print(f"生成图表: {len(saved_files)}张")
    print(f"提交文件: results/submission.csv")
    print("\n所有结果已保存到 results/ 目录")
    print("\n[DONE] 项目运行完成！")
    

if __name__ == '__main__':
    main()
