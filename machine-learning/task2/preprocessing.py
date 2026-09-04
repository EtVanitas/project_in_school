"""
数据预处理模块
功能:HSV色彩空间分割、形态学操作、叶片区域提取
"""

import cv2
import numpy as np
from pathlib import Path


def convert_to_hsv(image):
    """
    将BGR图像转换为HSV色彩空间
    
    Args:
        image: BGR格式的numpy数组
        
    Returns:
        HSV格式的numpy数组
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv


def green_mask_hsv(hsv_image, h_min=35, h_max=85, s_min=40, v_min=40):
    """
    基于HSV色彩空间创建绿色掩膜
    
    Args:
        hsv_image: HSV格式的图像
        h_min: 色调最小值 (默认35)
        h_max: 色调最大值 (默认85)
        s_min: 饱和度最小值 (默认40, 过滤低饱和度区域)
        v_min: 明度最小值 (默认40, 过滤过暗区域)
        
    Returns:
        二值掩膜图像
    """
    # 定义绿色的HSV范围
    lower_green = np.array([h_min, s_min, v_min])
    upper_green = np.array([h_max, 255, 255])
    
    # 创建掩膜
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    return mask


def morphological_operations(mask, kernel_size=5):
    """
    形态学操作：闭运算填充孔洞，开运算去除噪点
    
    Args:
        mask: 二值掩膜图像
        kernel_size: 核大小
        
    Returns:
        处理后的掩膜图像
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 闭运算：填充叶片内部的小孔洞
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 开运算：去除小的噪点
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opened


def apply_mask_to_image(image, mask):
    """
    将掩膜应用到原图，只保留目标区域，背景置黑
    
    Args:
        image: 原始BGR图像
        mask: 二值掩膜图像
        
    Returns:
        处理后的图像（背景为黑色）
    """
    # 确保mask是3通道
    if len(mask.shape) == 2:
        mask_3ch = cv2.merge([mask, mask, mask])
    else:
        mask_3ch = mask
    
    # 按位与操作，保留目标区域
    result = cv2.bitwise_and(image, mask_3ch)
    
    return result


def preprocess_image(image_path, kernel_size=5):
    """
    完整的图像预处理流程
    
    Args:
        image_path: 图像文件路径
        kernel_size: 形态学操作的核大小
        
    Returns:
        processed_image: 处理后的图像(BGR格式)
        mask: 最终的掩膜
        success: 是否成功处理
    """
    try:
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None, False
        
        # 步骤1: 转换到HSV色彩空间
        hsv_image = convert_to_hsv(image)
        
        # 步骤2: 绿色范围阈值分割
        mask = green_mask_hsv(hsv_image)
        
        # 步骤3: 形态学操作
        mask = morphological_operations(mask, kernel_size=kernel_size)
        
        # 步骤4: 应用掩膜到原图
        processed_image = apply_mask_to_image(image, mask)
        
        return processed_image, mask, True
        
    except Exception as e:
        return None, None, False


def preprocess_batch(image_paths, output_dir=None, kernel_size=5, verbose=True):
    """
    批量预处理图像
    
    Args:
        image_paths: 图像路径列表
        output_dir: 输出目录(可选,如果提供则保存处理后的图像)
        kernel_size: 形态学操作的核大小
        verbose: 是否输出进度信息
        
    Returns:
        results: 字典,键为文件名,值为(processed_image, mask, success)
    """
    results = {}
    total = len(image_paths)
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx, image_path in enumerate(image_paths, 1):
        filename = Path(image_path).name
        if verbose:
            print(f"处理进度: {idx}/{total} - {filename}")
        
        processed_image, mask, success = preprocess_image(
            image_path, 
            kernel_size=kernel_size
        )
        
        results[filename] = {
            'processed_image': processed_image,
            'mask': mask,
            'success': success
        }
        
        # 如果指定了输出目录且处理成功，保存结果
        if output_dir and success and processed_image is not None:
            output_path = Path(output_dir) / filename
            cv2.imwrite(str(output_path), processed_image)
    
    # 统计成功和失败的数量
    success_count = sum(1 for r in results.values() if r['success'])
    fail_count = total - success_count
    
    if verbose:
        print(f"批量处理完成: 成功 {success_count}, 失败 {fail_count}")
    
    return results
