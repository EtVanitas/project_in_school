"""
Plant Seedlings Classification - 主程序入口
使用传统机器学习方法进行植物幼苗分类
"""

from model_training import PlantSeedlingClassifier


def main():
    """完整的训练和预测流程"""
    
    print("="*70)
    print("Plant Seedlings Classification - 传统机器学习方法")
    print("="*70)
    
    # 创建分类器
    classifier = PlantSeedlingClassifier(base_path='plant-seedlings-classification')
    
    # 步骤1: 准备训练集特征
    print("\n【步骤1】准备训练集特征...")
    X_train, y_train, valid_files = classifier.prepare_training_features(
        kernel_size=5
    )
    
    # 步骤2: 训练模型（包含超参数调优）
    print("\n【步骤2】训练模型...")
    print("提示: 使用快速搜索模式 (64种参数组合 × 5折交叉验证)")
    print("预计耗时: 30-60分钟\n")
    
    grid_search = classifier.train_with_grid_search(
        X_train, 
        y_train, 
        use_quick_search=True  # 快速模式
    )
    
    # 步骤3: 预测测试集
    print("\n【步骤3】预测测试集...")
    submission_df = classifier.predict_test_set(
        kernel_size=5
    )
    
    # 步骤4: 保存提交文件
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"\n  提交文件已保存到: {submission_path}")
    
    # 步骤5: 保存模型
    classifier.save_model('model.pkl')
    
    # 输出前10条预测结果
    print("\n前10条预测结果:")
    print(submission_df.head(10).to_string())
    
    print("\n" + "="*70)
    print("✓ 全部完成！")
    print("="*70)


if __name__ == "__main__":
    main()
