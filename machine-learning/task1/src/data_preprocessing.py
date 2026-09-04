"""
数据预处理模块 - 基于业务理解的缺失值填充
"""
import pandas as pd
import numpy as np


class DataPreprocessor:
    """数据预处理器 - 基于业务理解的智能填充"""
    
    def __init__(self, train_path: str, test_path: str):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.y_train = self.train_df['SalePrice']
        self.y_train_log = np.log1p(self.y_train)
        
        # 删除Id和目标变量
        self.X_train = self.train_df.drop(['Id', 'SalePrice'], axis=1)
        self.X_test = self.test_df.drop('Id', axis=1)
        
        self.test_ids = self.test_df['Id']
    
    def analyze_missing(self) -> pd.DataFrame:
        """分析缺失值情况"""
        missing = self.X_train.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        missing_pct = (missing / len(self.X_train) * 100).round(2)
        
        return pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_pct
        })
    
    def preprocess(self) -> tuple:
        """基于业务理解的智能填充（推荐策略）
        
        根据data_description.txt，NA有特定含义：
        - 某些特征的NA表示"无此设施"（如Alley=NA表示没有巷道）
        - 这些应该填充为"None"或0，而非中位数/众数
        """
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        
        # NA表示"无此设施"的特征 → 填充为"None"
        none_cols = [
            'Alley',              # Alley: 巷道类型
            'BsmtQual',           # BsmtQual: 地下室质量
            'BsmtCond',           # BsmtCond: 地下室状况
            'BsmtExposure',       # BsmtExposure: 地下室采光
            'BsmtFinType1',       # BsmtFinType1: 地下室装修类型1
            'BsmtFinType2',       # BsmtFinType2: 地下室装修类型2
            'FireplaceQu',        # FireplaceQu: 壁炉质量
            'GarageType',         # GarageType: 车库位置
            'GarageFinish',       # GarageFinish: 车库内部装修
            'GarageQual',         # GarageQual: 车库质量
            'GarageCond',         # GarageCond: 车库状况
            'PoolQC',             # PoolQC: 泳池质量
            'Fence',              # Fence: 围栏质量
            'MiscFeature',        # MiscFeature: 其他设施
            'MasVnrType'          # MasVnrType: 砌体veneer类型
        ]
        
        for col in none_cols:
            if col in X_train.columns:
                X_train[col] = X_train[col].fillna('None')
                X_test[col] = X_test[col].fillna('None')
        
        # NA表示0的特征
        zero_cols = [
            'MasVnrArea',         # MasVnrArea: 砌体veneer面积
            'GarageYrBlt',        # GarageYrBlt: 车库建造年份
            'GarageCars',         # GarageCars: 车库容量
            'GarageArea',         # GarageArea: 车库面积
            'BsmtFinSF1',         # BsmtFinSF1: 地下室装修类型1面积
            'BsmtFinSF2',         # BsmtFinSF2: 地下室装修类型2面积
            'BsmtUnfSF',          # BsmtUnfSF: 未装修地下室面积
            'TotalBsmtSF',        # TotalBsmtSF: 地下室总面积
            'BsmtFullBath',       # BsmtFullBath: 地下室全浴室数
            'BsmtHalfBath'        # BsmtHalfBath: 地下室半浴室数
        ]
        
        for col in zero_cols:
            if col in X_train.columns:
                X_train[col] = X_train[col].fillna(0)
                X_test[col] = X_test[col].fillna(0)
        
        # 剩余的分类型特征用众数填充
        remaining_cat = X_train.select_dtypes(include=['object']).columns
        for col in remaining_cat:
            if X_train[col].isnull().sum() > 0:
                mode_val = X_train[col].mode()[0]
                X_train[col] = X_train[col].fillna(mode_val)
            if X_test[col].isnull().sum() > 0:
                mode_val = X_train[col].mode()[0]
                X_test[col] = X_test[col].fillna(mode_val)
        
        # 剩余的数值型特征用中位数填充
        remaining_num = X_train.select_dtypes(include=[np.number]).columns
        for col in remaining_num:
            if X_train[col].isnull().sum() > 0:
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
            if X_test[col].isnull().sum() > 0:
                median_val = X_train[col].median()
                X_test[col] = X_test[col].fillna(median_val)
        
        return X_train, X_test
