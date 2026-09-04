"""
特征工程模块

标准流程：
1. 特征构造（交互、二元、比率）
2. 偏态处理
3. 编码（Ordinal + One-Hot）
4. 特征选择
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy.stats import skew


class FeatureEngineer:
    """特征工程师 - 标准化流程"""
    
    def __init__(self):
        """初始化"""
        pass
    
    def transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train_log: pd.Series = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        完整的特征工程流程
        
        参数:
            X_train: 训练集特征
            X_test: 测试集特征
            y_train_log: log变换后的目标变量（用于特征选择）
        
        返回:
            X_train_transformed: 转换后的训练集
            X_test_transformed: 转换后的测试集
            selected_features: 选择的特征列表
        """
        print("\n【特征工程完整流程】")
        
        # 步骤1: 特征构造
        print("  步骤1/4: 特征构造...")
        X_train, X_test = self._create_features(X_train, X_test)
        print(f"    构造后特征数: {X_train.shape[1]}")
        
        # 步骤2: 处理偏态
        print("  步骤2/4: 处理偏态分布...")
        X_train, X_test = self._handle_skewness(X_train, X_test)
        
        # 步骤3: 编码分类型变量
        print("  步骤3/4: 编码分类型变量...")
        X_train, X_test = self._encode_categorical(X_train, X_test)
        print(f"    编码后特征数: {X_train.shape[1]}")
        
        # 步骤4: 特征选择
        print("  步骤4/4: 特征选择...")
        if y_train_log is not None:
            X_train, X_test, selected_features = self._select_features(
                X_train, X_test, y_train_log
            )
        else:
            selected_features = X_train.columns.tolist()
        
        print(f"    最终特征数: {len(selected_features)}")
        
        return X_train, X_test, selected_features
    
    def _create_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """构造新特征（交互、二元、比率）"""
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        # 1. 交互特征 - 面积和时间相关
        X_train['TotalSF'] = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']  # TotalSF: 总居住面积
        X_test['TotalSF'] = X_test['TotalBsmtSF'] + X_test['1stFlrSF'] + X_test['2ndFlrSF']
        
        X_train['TotalPorchSF'] = (X_train['WoodDeckSF'] + X_train['OpenPorchSF'] +  # TotalPorchSF: 总门廊面积
                                   X_train['EnclosedPorch'] + X_train['3SsnPorch'] + 
                                   X_train['ScreenPorch'])
        X_test['TotalPorchSF'] = (X_test['WoodDeckSF'] + X_test['OpenPorchSF'] + 
                                  X_test['EnclosedPorch'] + X_test['3SsnPorch'] + 
                                  X_test['ScreenPorch'])
        
        X_train['HouseAge'] = X_train['YrSold'] - X_train['YearBuilt']  # HouseAge: 房龄
        X_test['HouseAge'] = X_test['YrSold'] - X_test['YearBuilt']
        
        X_train['RemodAge'] = X_train['YrSold'] - X_train['YearRemodAdd']  # RemodAge: 装修后年数
        X_test['RemodAge'] = X_test['YrSold'] - X_test['YearRemodAdd']
        
        X_train['TotalBath'] = (X_train['FullBath'] + X_train['HalfBath'] * 0.5 +  # TotalBath: 总浴室数
                               X_train['BsmtFullBath'] + X_train['BsmtHalfBath'] * 0.5)
        X_test['TotalBath'] = (X_test['FullBath'] + X_test['HalfBath'] * 0.5 + 
                              X_test['BsmtFullBath'] + X_test['BsmtHalfBath'] * 0.5)
        
        # 2. 二元特征 - 是否有某设施
        X_train['HasBsmt'] = (X_train['TotalBsmtSF'] > 0).astype(int)  # HasBsmt: 是否有地下室
        X_test['HasBsmt'] = (X_test['TotalBsmtSF'] > 0).astype(int)
        
        X_train['HasGarage'] = (X_train['GarageArea'] > 0).astype(int)  # HasGarage: 是否有车库
        X_test['HasGarage'] = (X_test['GarageArea'] > 0).astype(int)
        
        X_train['HasFireplace'] = (X_train['Fireplaces'] > 0).astype(int)  # HasFireplace: 是否有壁炉
        X_test['HasFireplace'] = (X_test['Fireplaces'] > 0).astype(int)
        
        X_train['HasPool'] = (X_train['PoolArea'] > 0).astype(int)  # HasPool: 是否有泳池
        X_test['HasPool'] = (X_test['PoolArea'] > 0).astype(int)
        
        # 3. 比率特征
        total_sf = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']
        X_train['1stFlrSF_ratio'] = X_train['1stFlrSF'] / (total_sf + 1)  # 1stFlrSF_ratio: 一楼面积占比
        X_train['BsmtSF_ratio'] = X_train['TotalBsmtSF'] / (total_sf + 1)  # BsmtSF_ratio: 地下室面积占比
        X_train['LivingLotRatio'] = X_train['GrLivArea'] / (X_train['LotArea'] + 1)  # LivingLotRatio: 居住面积与地块面积比
        
        total_sf_test = X_test['TotalBsmtSF'] + X_test['1stFlrSF'] + X_test['2ndFlrSF']
        X_test['1stFlrSF_ratio'] = X_test['1stFlrSF'] / (total_sf_test + 1)
        X_test['BsmtSF_ratio'] = X_test['TotalBsmtSF'] / (total_sf_test + 1)
        X_test['LivingLotRatio'] = X_test['GrLivArea'] / (X_test['LotArea'] + 1)
        
        return X_train, X_test
    
    def _handle_skewness(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        threshold: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """处理偏态分布"""
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        # 选择所有数值型特征
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # 计算每个特征的偏度（skewness）
        skewed_feats = X_train[numerical_cols].apply(
            lambda x: skew(x.dropna())  # skew: 偏度，衡量分布的对称性
        ).sort_values(ascending=False)
        
        # 筛选出偏度超过阈值的特征
        high_skew = skewed_feats[skewed_feats > threshold]
        
        # 对高偏度且全为正数的特征进行log1p变换
        for feat in high_skew.index:
            if (X_train[feat] > 0).all() and (X_test[feat] > 0).all():
                X_train[feat] = np.log1p(X_train[feat])  # log1p: log(1+x)，使分布更接近正态
                X_test[feat] = np.log1p(X_test[feat])
        
        return X_train, X_test
    
    def _encode_categorical(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """编码分类型变量（Ordinal + One-Hot）"""
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        # 1. Ordinal Encoding - 有序分类
        ordinal_mappings = {
            'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},  # ExterQual: 外墙质量
            'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},  # ExterCond: 外墙状况
            'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},  # BsmtQual: 地下室质量
            'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},  # BsmtCond: 地下室状况
            'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0},  # BsmtExposure: 地下室采光
            'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0},  # BsmtFinType1: 地下室装修类型1
            'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0},  # BsmtFinType2: 地下室装修类型2
            'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},  # HeatingQC: 供暖质量
            'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},  # KitchenQual: 厨房质量
            'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},  # FireplaceQu: 壁炉质量
            'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0},  # GarageFinish: 车库内部装修
            'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},  # GarageQual: 车库质量
            'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},  # GarageCond: 车库状况
            'PoolQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'None': 0},  # PoolQC: 泳池质量
            'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0},  # Fence: 围栏质量
            'LandSlope': {'Gtl': 3, 'Mod': 2, 'Sev': 1},  # LandSlope: 地块坡度
            'LotShape': {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1},  # LotShape: 地块形状
            'CentralAir': {'Y': 1, 'N': 0},  # CentralAir: 中央空调
            'PavedDrive': {'Y': 3, 'P': 2, 'N': 1},  # PavedDrive: 铺砌车道
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in X_train.columns:
                X_train[col] = X_train[col].map(mapping)
                X_test[col] = X_test[col].map(mapping)
        
        # 2. One-Hot Encoding - 剩余无序分类
        cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        
        if cat_cols:
            n_train = X_train.shape[0]
            all_data = pd.concat([X_train, X_test], axis=0)
            all_encoded = pd.get_dummies(all_data, columns=cat_cols, drop_first=False)
            
            X_train = all_encoded.iloc[:n_train, :]
            X_test = all_encoded.iloc[n_train:, :]
        
        return X_train, X_test
    
    def _select_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, threshold: float = 0.001) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """基于模型重要性的特征选择"""
        from sklearn.ensemble import RandomForestRegressor
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_features = importance[importance['importance'] >= threshold]['feature'].tolist()
        
        print(f"    Top 10特征:")
        print(importance.head(10)[['feature', 'importance']].to_string(index=False))
        
        return X_train[selected_features], X_test[selected_features], selected_features
