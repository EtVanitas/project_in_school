"""
模型训练模块

标准流程：
1. 定义模型库
2. 快速评估所有模型
3. 对Top模型进行超参数调优
4. 选择最佳模型
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """模型训练器 - 标准化流程"""
    
    def __init__(self):
        """初始化"""
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_test: pd.DataFrame = None) -> Tuple[str, object, Dict]:
        """
        完整的模型训练流程
        
        参数:
            X_train: 训练集特征
            y_train: 训练集目标变量
            X_test: 测试集特征
        
        返回:
            best_model_name: 最佳模型名称
            best_model: 最佳模型对象
            all_results: 所有模型的评估结果
        """
        print("\n【模型训练完整流程】")
        
        # 步骤1: 快速评估所有基础模型
        print("  步骤1/3: 快速评估所有模型...")
        base_results = self._quick_evaluate(X_train, y_train)
        
        # 步骤2: 对所有模型进行超参数调优
        print("  步骤2/3: 超参数调优...")
        tuned_results = self._tune_all_models(X_train, y_train, base_results)
        
        # 步骤3: 选择最佳模型
        print("  步骤3/3: 选择最佳模型...")
        best_name, best_model = self._select_best_model(tuned_results)
        
        self.best_model_name = best_name
        self.best_model = best_model
        self.results = tuned_results
        
        return best_name, best_model, tuned_results
    
    def _get_model_library(self) -> Dict:
        """定义模型库"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.001, max_iter=10000),
            'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
            ),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
        }
    
    def _quick_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """快速评估所有基础模型"""
        models = self._get_model_library()
        results = {}
        
        for name, model in models.items():
            try:
                # 5折交叉验证
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5, scoring='neg_mean_squared_error'
                )
                rmse_cv = np.sqrt(-cv_scores.mean())
                
                results[name] = {
                    'model': model,
                    'rmse_cv': rmse_cv,
                    'cv_std': cv_scores.std(),
                    'tuned': False
                }
                
                print(f"    {name:25s} RMSE: ${rmse_cv:>10,.2f}")
                
            except Exception as e:
                print(f"    {name:25s} [ERROR] {e}")
                results[name] = None
        
        return results
    
    def _tune_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        base_results: Dict) -> Dict:
        """对所有模型进行超参数调优"""
        valid_results = {k: v for k, v in base_results.items() if v is not None}
        
        print(f"    对所有{len(valid_results)}个模型进行调优:")
        
        tuned_results = base_results.copy()
        
        for name, result in valid_results.items():
            print(f"      调优: {name}...")
            
            try:
                best_model = self._tune_single_model(name, X_train, y_train)
                
                # 重新评估调优后的模型
                cv_scores = cross_val_score(
                    best_model, X_train, y_train,
                    cv=5, scoring='neg_mean_squared_error'
                )
                rmse_cv = np.sqrt(-cv_scores.mean())
                
                tuned_results[name] = {
                    'model': best_model,
                    'rmse_cv': rmse_cv,
                    'cv_std': cv_scores.std(),
                    'tuned': True
                }
                
                improvement = (result['rmse_cv'] - rmse_cv) / result['rmse_cv'] * 100
                print(f"        调优后RMSE: ${rmse_cv:,.2f} (提升{improvement:.1f}%)")
                
            except Exception as e:
                print(f"        [ERROR] 调优失败: {e}")
        
        return tuned_results
    
    def _tune_single_model(self, model_name: str, X_train: pd.DataFrame,
                          y_train: pd.Series):
        """对单个模型进行超参数调优"""
        param_grids = {
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'alpha': [0.0001, 0.001, 0.01, 0.1]
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ElasticNet': {
                'alpha': [0.0001, 0.001, 0.01],
                'l1_ratio': [0.1, 0.5, 0.9]
            }
        }
        
        base_models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(max_iter=10000),
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'ElasticNet': ElasticNet(max_iter=10000),
            'SVR': SVR()
        }
        
        if model_name not in param_grids:
            # 如果不在调优列表中，返回原模型
            return base_models[model_name]
        
        grid_search = GridSearchCV(
            base_models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_
    
    def _select_best_model(self, results: Dict) -> Tuple[str, object]:
        """选择最佳模型"""
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        best_name = min(valid_results.keys(), key=lambda x: valid_results[x]['rmse_cv'])
        best_result = valid_results[best_name]
        
        print(f"    [BEST] {best_name}")
        print(f"           CV RMSE: ${best_result['rmse_cv']:,.2f}")
        print(f"           是否调优: {'是' if best_result['tuned'] else '否'}")
        
        return best_name, best_result['model']
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """使用最佳模型进行预测"""
        if self.best_model is None:
            raise ValueError("请先调用train()方法训练模型")
        
        return self.best_model.predict(X_test)
    
    def get_model_comparison(self) -> pd.DataFrame:
        """获取模型对比表格"""
        if not self.results:
            raise ValueError("请先调用train()方法")
        
        comparison = []
        
        for name, result in self.results.items():
            if result is not None:
                comparison.append({
                    'Model': name,
                    'CV RMSE': result['rmse_cv'],
                    'CV Std': result['cv_std'],
                    'Tuned': 'Yes' if result['tuned'] else 'No'
                })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('CV RMSE')
        
        return df
