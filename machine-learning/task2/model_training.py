"""
模型训练模块
功能:XGBoost训练、GridSearchCV超参数调优、预测和生成提交文件
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import pickle

# 导入自定义模块
from preprocessing import preprocess_batch
from feature_extraction import extract_features_batch


class PlantSeedlingClassifier:
    """植物幼苗分类器"""
    
    def __init__(self, base_path='plant-seedlings-classification'):
        """
        初始化分类器
        
        Args:
            base_path: 数据集根目录
        """
        self.base_path = Path(base_path)
        self.train_path = self.base_path / 'train'
        self.test_path = self.base_path / 'test'
        
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        self.best_params = None
        
    def load_training_data(self):
        """
        加载训练数据
        
        Returns:
            image_paths: 训练图像路径列表
            labels: 对应的标签列表
        """
        image_paths = []
        labels = []
        
        # 遍历所有类别文件夹
        for class_dir in self.train_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for image_file in class_dir.glob('*.png'):
                    image_paths.append(image_file)
                    labels.append(class_name)
        
        print(f"加载了 {len(image_paths)} 张训练图像")
        print(f"类别数量: {len(set(labels))}")
        print(f"类别列表: {sorted(set(labels))}")
        
        return image_paths, labels
    
    def load_test_data(self):
        """
        加载测试数据
        
        Returns:
            test_image_paths: 测试图像路径列表
        """
        test_image_paths = list(self.test_path.glob('*.png'))
        print(f"加载了 {len(test_image_paths)} 张测试图像")
        
        return test_image_paths
    
    def prepare_training_features(self, kernel_size=5):
        """
        准备训练集特征
        
        Args:
            kernel_size: 形态学操作的核大小
            
        Returns:
            X_train: 训练特征矩阵
            y_train: 训练标签数组
            valid_files: 成功处理的文件名列表
        """
        print("="*60)
        print("开始准备训练集特征...")
        print("="*60)
        
        # 加载训练数据
        image_paths, labels = self.load_training_data()
        
        # 批量预处理
        preprocessed_results = preprocess_batch(
            image_paths,
            kernel_size=kernel_size
        )
        
        # 批量提取特征
        features_dict, feature_matrix, feature_names, valid_files = extract_features_batch(
            preprocessed_results
        )
        
        if feature_matrix is None:
            raise ValueError("特征提取失败，没有有效的特征数据")
        
        # 编码标签（只保留有效文件的标签）
        valid_labels = []
        for filename in valid_files:
            for class_dir in self.train_path.iterdir():
                if class_dir.is_dir() and (class_dir / filename).exists():
                    valid_labels.append(class_dir.name)
                    break
        
        # 转换标签为数值
        y_train = self.label_encoder.fit_transform(valid_labels)
        X_train = feature_matrix
        
        self.feature_names = feature_names
        
        print(f"训练集特征矩阵形状: {X_train.shape}")
        print(f"训练集标签数量: {len(y_train)}")
        print(f"特征数量: {X_train.shape[1]}")
        
        return X_train, y_train, valid_files
    
    def train_with_grid_search(self, X_train, y_train, use_quick_search=True):
        """
        使用网格搜索训练XGBoost模型
        
        Args:
            X_train: 训练特征矩阵
            y_train: 训练标签数组
            use_quick_search: 是否使用快速搜索
        """
        print("="*60)
        print("开始XGBoost模型训练和超参数调优...")
        print("="*60)
        
        # 定义参数网格
        if use_quick_search:
            # 快速搜索 - 较小的参数空间
            param_grid = {
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [150, 250],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            print("使用快速参数网格 (64种组合)")
        else:
            # 完整搜索 - 更大的参数空间
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
            }
            print("使用完整参数网格 (972种组合)")
        
        print(f"参数网格: {param_grid}")
        
        # 创建XGBoost分类器
        xgb_classifier = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(self.label_encoder.classes_),
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心加速训练
        )
        
        # 分层K折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 网格搜索 - 使用F1分数(macro)作为主要评判标准
        grid_search = GridSearchCV(
            estimator=xgb_classifier,
            param_grid=param_grid,
            scoring={
                'accuracy': 'accuracy',
                'f1_macro': 'f1_macro',
                'f1_weighted': 'f1_weighted'
            },
            refit='f1_macro',  # 使用F1 macro作为最终选择标准
            cv=cv,
            n_jobs=-1,  # 并行搜索
            verbose=1,
            return_train_score=True,
            pre_dispatch='2*n_jobs'  # 控制并行任务数量
        )
        
        print("开始网格搜索...")
        grid_search.fit(X_train, y_train)
        
        # 输出最佳参数
        self.best_params = grid_search.best_params_
        print(f"最佳参数: {self.best_params}")
        print(f"最佳交叉验证准确率: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
        print(f"最佳交叉验证F1(macro): {grid_search.cv_results_['mean_test_f1_macro'][grid_search.best_index_]:.4f}")
        print(f"最佳交叉验证F1(weighted): {grid_search.cv_results_['mean_test_f1_weighted'][grid_search.best_index_]:.4f}")
        
        # 使用最佳参数训练最终模型
        self.model = grid_search.best_estimator_
        
        # 在训练集上评估
        y_pred_train = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_f1_macro = f1_score(y_train, y_pred_train, average='macro')
        train_f1_weighted = f1_score(y_train, y_pred_train, average='weighted')
        
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"训练集F1(macro): {train_f1_macro:.4f}")
        print(f"训练集F1(weighted): {train_f1_weighted:.4f}")
        
        # 详细分类报告
        print("\n训练集分类报告:")
        print(classification_report(
            y_train, 
            y_pred_train, 
            target_names=self.label_encoder.classes_
        ))
        
        # 输出特征重要性
        self._print_feature_importance()
        
        return grid_search
    
    def _print_feature_importance(self, top_n=20):
        """
        打印特征重要性
        
        Args:
            top_n: 显示前N个最重要的特征
        """
        if self.model is None or self.feature_names is None:
            print("模型或特征名称未设置，无法显示特征重要性")
            return
        
        # 获取特征重要性
        importances = self.model.feature_importances_
        
        # 创建特征名称和重要性的DataFrame
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        # 按重要性排序
        feature_imp_df = feature_imp_df.sort_values('importance', ascending=False)
        
        # 显示前N个
        print(f"\nTop {top_n} 最重要特征:")
        print("="*60)
        for idx, row in feature_imp_df.head(top_n).iterrows():
            print(f"{row['feature']:30s}: {row['importance']:.4f}")
        
        # 统计各类特征的平均重要性
        print("\n各类特征平均重要性:")
        print("-"*60)
        
        # HSV颜色特征 (前48个)
        hsv_imp = importances[:48].mean()
        print(f"HSV颜色特征 (48维): {hsv_imp:.4f}")
        
        # LBP纹理特征 (48-107)
        lbp_imp = importances[48:107].mean()
        print(f"LBP纹理特征 (59维): {lbp_imp:.4f}")
        
        # Haralick纹理特征 (107-113)
        haralick_imp = importances[107:113].mean()
        print(f"Haralick纹理特征 (6维): {haralick_imp:.4f}")
        
        # 形状特征 (113-120)
        shape_imp = importances[113:120].mean()
        print(f"形状特征 (7维): {shape_imp:.4f}")
        
        # 统计特征 (120-132)
        stat_imp = importances[120:132].mean()
        print(f"统计特征 (12维): {stat_imp:.4f}")
    
    def predict_test_set(self, kernel_size=5):
        """
        预测测试集并生成提交文件
        
        Args:
            use_largest_contour: 是否只保留最大轮廓
            kernel_size: 形态学操作的核大小
            
        Returns:
            submission_df: 提交文件的DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练！请先调用 train_with_grid_search()")
        
        print("="*60)
        print("开始预测测试集...")
        print("="*60)
        
        # 加载测试数据
        test_image_paths = self.load_test_data()
        
        # 批量预处理测试图像 (禁用进度输出)
        preprocessed_results = preprocess_batch(
            test_image_paths,
            kernel_size=kernel_size,
            verbose=False
        )
        
        # 批量提取特征 (禁用进度输出)
        features_dict, feature_matrix, feature_names, valid_files = extract_features_batch(
            preprocessed_results,
            verbose=False
        )
        
        if feature_matrix is None:
            raise ValueError("测试集特征提取失败")
        
        # 预测
        print("进行预测...")
        predictions = self.model.predict(feature_matrix)
        
        # 解码标签
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        # 创建提交文件
        submission_data = {
            'file': valid_files,
            'species': predicted_labels
        }
        submission_df = pd.DataFrame(submission_data)
        
        # 按照sample_submission.csv的顺序排列（可选）
        sample_submission = pd.read_csv(self.base_path / 'sample_submission.csv')
        submission_df = submission_df.set_index('file').reindex(sample_submission['file']).reset_index()
        
        print(f"预测完成！共预测 {len(submission_df)} 张图片")
        print(f"预测类别分布:\n{submission_df['species'].value_counts()}")
        
        return submission_df
    
    def save_model(self, model_path='model.pkl'):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path='model.pkl'):
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data['best_params']
        
        print(f"模型已从 {model_path} 加载")
