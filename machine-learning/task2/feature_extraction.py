"""
特征提取模块
功能:提取纹理、颜色、形状、统计特征
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def extract_color_features_hsv(image, mask=None):
    """
    提取HSV色彩空间的直方图特征
    
    Args:
        image: BGR格式的图像
        mask: 二值掩膜,只在掩膜区域内计算
        
    Returns:
        hsv_hist: HSV三个通道的直方图特征（每个通道16个bin，共48维）
    """
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 如果有掩膜,只在掩膜区域内计算
    if mask is not None:
        # 确保掩膜是二值的
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
    
    # 计算每个通道的直方图
    h_bins = 16  # H通道bins数
    s_bins = 16  # S通道bins数
    v_bins = 16  # V通道bins数
    
    # H通道 (0-179)
    h_hist = cv2.calcHist([hsv], [0], mask_gray if mask is not None else None, [h_bins], [0, 180])
    # S通道 (0-255)
    s_hist = cv2.calcHist([hsv], [1], mask_gray if mask is not None else None, [s_bins], [0, 256])
    # V通道 (0-255)
    v_hist = cv2.calcHist([hsv], [2], mask_gray if mask is not None else None, [v_bins], [0, 256])
    
    # 归一化直方图
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    
    # 合并特征
    hsv_hist = np.concatenate([h_hist, s_hist, v_hist])
    
    return hsv_hist


def extract_lbp_features(image, radius=1, n_points=8, method='uniform', n_bins=59):
    """
    提取LBP(局部二值模式)纹理特征
    
    Args:
        image: BGR格式的图像
        radius: LBP半径
        n_points: 采样点数
        method: LBP方法 ('uniform', 'default', 'var')
        n_bins: 固定的直方图bins数量
        
    Returns:
        lbp_hist: LBP直方图特征(固定维度)
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 计算LBP
    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    
    # 计算LBP直方图 - 使用固定的bins数量
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return lbp_hist


def extract_haralick_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    提取Haralick纹理特征（基于灰度共生矩阵GLCM）
    
    Args:
        image: BGR格式的图像
        distances: 像素距离列表
        angles: 角度列表（0, pi/4, pi/2, 3pi/4）- 多方向更鲁棒
        
    Returns:
        haralick_features: Haralick特征数组
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 量化灰度级到较少级别以加速计算
    gray_quantized = (gray / 32).astype(np.uint8)
    
    try:
        # 计算GLCM - 使用多个角度
        glcm = graycomatrix(
            gray_quantized, 
            distances=distances, 
            angles=angles,
            levels=8,  # 量化后的灰度级数
            symmetric=True, 
            normed=True
        )
        
        # 提取多个Haralick特征 (对所有角度取平均)
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        asm = graycoprops(glcm, 'ASM').mean()  # 角二阶矩
        
        haralick_features = np.array([
            contrast, 
            dissimilarity, 
            homogeneity, 
            energy, 
            correlation,
            asm
        ])
        
        return haralick_features
        
    except Exception as e:
        return np.zeros(6)


def extract_shape_features(mask):
    """
    从掩膜中提取形状特征
    
    Args:
        mask: 二值掩膜图像
        
    Returns:
        shape_features: 形状特征数组
    """
    # 查找所有轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros(7)
    
    # 计算所有轮廓的总面积和总周长
    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
    
    # 避免除零错误
    if total_perimeter == 0:
        total_perimeter = 1e-6
    
    # 紧凑度 (Circularity)
    compactness = (4 * np.pi * total_area) / (total_perimeter ** 2)
    
    # 合并所有轮廓点来计算整体边界框
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # 宽高比
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # 使用最大的单个轮廓来计算椭圆特征
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 拟合椭圆
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse
        
        # 偏心率
        eccentricity = np.sqrt(1 - (min(major_axis, minor_axis) / max(major_axis, minor_axis)) ** 2)
        
        # 方向
        orientation = angle
    else:
        eccentricity = 0
        orientation = 0
    
    # 凸包相关特征 (基于所有点)
    hull = cv2.convexHull(all_points)
    hull_area = cv2.contourArea(hull)
    solidity = total_area / hull_area if hull_area > 0 else 0
    
    shape_features = np.array([
        total_area,
        total_perimeter,
        compactness,
        aspect_ratio,
        eccentricity,
        solidity,
        orientation
    ])
    
    return shape_features


def extract_statistical_features(image, mask=None):
    """
    提取各通道的统计特征（均值、标准差、峰度、偏度）
    
    Args:
        image: BGR格式的图像
        mask: 二值掩膜,只在掩膜区域内计算
        
    Returns:
        stat_features: 统计特征数组
    """
    features = []
    
    # 如果有掩膜,提取掩膜区域内的像素
    if mask is not None:
        # 确保掩膜是二值的
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        
        # 找到掩膜中非零的位置
        nonzero_indices = np.where(mask_gray > 0)
        
        # 如果没有有效像素,返回零向量
        if len(nonzero_indices[0]) == 0:
            return np.zeros(12)
    
    # 对每个通道计算统计量
    for i in range(3):  # B, G, R三个通道
        channel = image[:, :, i]
        
        # 如果提供了掩膜,只使用掩膜区域内的像素
        if mask is not None:
            pixels = channel[nonzero_indices].astype(np.float64)
        else:
            pixels = channel.ravel().astype(np.float64)
        
        # 均值
        mean_val = np.mean(pixels)
        # 标准差
        std_val = np.std(pixels)
        # 偏度 (Skewness)
        if std_val > 0:
            skewness = np.mean(((pixels - mean_val) / std_val) ** 3)
            # 峰度 (Kurtosis)
            kurtosis = np.mean(((pixels - mean_val) / std_val) ** 4) - 3
        else:
            skewness = 0
            kurtosis = 0
        
        features.extend([mean_val, std_val, skewness, kurtosis])
    
    return np.array(features)


def extract_all_features(processed_image, mask, normalize=True):
    """
    提取所有类型的特征并合并
    
    Args:
        processed_image: 预处理后的图像（BGR格式）
        mask: 对应的掩膜图像
        normalize: 是否对形状特征进行归一化
        
    Returns:
        all_features: 合并后的特征向量
        feature_names: 特征名称列表
    """
    if processed_image is None or mask is None:
        return None, None
    
    try:
        # 1. 颜色特征 (HSV直方图) - 48维 (已归一化,使用掩膜区域)
        color_features = extract_color_features_hsv(processed_image, mask)
        
        # 2. LBP纹理特征 - 59维 (已归一化为density)
        lbp_features = extract_lbp_features(processed_image)
        
        # 3. Haralick纹理特征 - 6维
        haralick_features = extract_haralick_features(processed_image)
        
        # 4. 形状特征 - 7维
        shape_features = extract_shape_features(mask)
        
        # 5. 统计特征 - 12维 (3通道 × 4统计量,使用掩膜区域)
        stat_features = extract_statistical_features(processed_image, mask)
        
        # 对形状特征进行归一化 (除以图像总面积)
        if normalize:
            image_area = mask.shape[0] * mask.shape[1]
            if image_area > 0 and shape_features[0] > 0:  # area > 0
                shape_features[0] = shape_features[0] / image_area  # 归一化面积
                shape_features[1] = shape_features[1] / np.sqrt(image_area)  # 归一化周长
        
        # 合并所有特征
        all_features = np.concatenate([
            color_features,
            lbp_features,
            haralick_features,
            shape_features,
            stat_features
        ])
        
        # 生成特征名称
        feature_names = (
            [f'hsv_h_{i}' for i in range(16)] +
            [f'hsv_s_{i}' for i in range(16)] +
            [f'hsv_v_{i}' for i in range(16)] +
            [f'lbp_{i}' for i in range(len(lbp_features))] +
            ['haralick_contrast', 'haralick_dissimilarity', 'haralick_homogeneity',
             'haralick_energy', 'haralick_correlation', 'haralick_asm'] +
            ['shape_area_norm', 'shape_perimeter_norm', 'shape_compactness', 'shape_aspect_ratio',
             'shape_eccentricity', 'shape_solidity', 'shape_orientation'] +
            [f'stat_b_mean', 'stat_b_std', 'stat_b_skew', 'stat_b_kurt',
             f'stat_g_mean', 'stat_g_std', 'stat_g_skew', 'stat_g_kurt',
             f'stat_r_mean', 'stat_r_std', 'stat_r_skew', 'stat_r_kurt']
        )
        
        return all_features, feature_names
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None


def extract_features_batch(preprocessed_results, verbose=True):
    """
    批量提取特征
    
    Args:
        preprocessed_results: preprocessing.py返回的结果字典
        verbose: 是否输出进度信息
        
    Returns:
        features_dict: 字典，键为文件名，值为特征向量
        feature_matrix: 特征矩阵 (n_samples, n_features)
        feature_names: 特征名称列表
        valid_files: 成功提取特征的文件名列表
    """
    features_dict = {}
    feature_list = []
    valid_files = []
    feature_names = None
    
    total = len(preprocessed_results)
    
    for idx, (filename, result) in enumerate(preprocessed_results.items(), 1):
        if verbose:
            print(f"提取特征进度: {idx}/{total} - {filename}")
        
        if not result['success']:
            if verbose:
                print(f"跳过 {filename}: 预处理失败")
            continue
        
        processed_image = result['processed_image']
        mask = result['mask']
        
        features, names = extract_all_features(processed_image, mask)
        
        if features is not None:
            features_dict[filename] = features
            feature_list.append(features)
            valid_files.append(filename)
            
            # 记录特征名称（只需要一次）
            if feature_names is None:
                feature_names = names
    
    # 转换为特征矩阵
    if feature_list:
        feature_matrix = np.array(feature_list)
        if verbose:
            print(f"特征矩阵形状: {feature_matrix.shape}")
    else:
        feature_matrix = None
        if verbose:
            print("没有成功提取任何特征")
    
    return features_dict, feature_matrix, feature_names, valid_files
