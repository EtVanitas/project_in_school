"""数据加载与增强模块"""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config


class PlantDataset(Dataset):
    """植物幼苗数据集"""
    
    def __init__(self, image_paths, labels=None, transform=None, is_test=False):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表(测试集为None)
            transform: 数据变换
            is_test: 是否为测试集
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载并转换图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 测试集返回文件名，训练/验证集返回标签
        if self.is_test:
            filename = Path(self.image_paths[idx]).name
            return image, -1, filename
        else:
            return image, self.labels[idx]


def get_train_transforms():
    """获取训练集数据增强变换"""
    return transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])


def get_val_transforms():
    """获取验证/测试集变换"""
    return transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.14)),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_training_data():
    """
    加载训练数据并划分训练集和验证集
    
    Returns:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        class_to_idx: 类别到索引的映射
    """
    print("加载训练数据...")
    
    # 检查数据目录是否存在
    if not config.TRAIN_DIR.exists():
        raise FileNotFoundError(f"训练数据目录不存在: {config.TRAIN_DIR}")
    
    # 构建类别映射
    class_to_idx = {cls: idx for idx, cls in enumerate(config.CLASSES)}
    image_paths, labels = [], []
    
    # 收集所有图像路径和标签
    for class_name in config.CLASSES:
        class_dir = config.TRAIN_DIR / class_name
        if not class_dir.exists():
            print(f"警告: 类别目录不存在 - {class_dir}")
            continue
        
        for img_path in class_dir.glob("*.png"):
            try:
                # 验证图像是否可以正常打开
                img = Image.open(img_path)
                img.verify()
                image_paths.append(str(img_path))
                labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"警告: 无法加载图像 {img_path}: {e}")
    
    if len(image_paths) == 0:
        raise ValueError("没有成功加载任何训练图像")
    
    print(f"总共加载 {len(image_paths)} 张训练图像")
    
    # 划分训练集和验证集(80%/20%)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集: {len(train_paths)} 张, 验证集: {len(val_paths)} 张")
    
    # 创建数据集
    train_dataset = PlantDataset(train_paths, train_labels, transform=get_train_transforms())
    val_dataset = PlantDataset(val_paths, val_labels, transform=get_val_transforms())
    
    return train_dataset, val_dataset, class_to_idx


def load_test_data():
    """
    加载测试数据
    
    Returns:
        test_dataset: 测试数据集
        test_image_paths: 测试图像路径列表
    """
    print("加载测试数据...")
    
    # 检查测试目录是否存在
    if not config.TEST_DIR.exists():
        raise FileNotFoundError(f"测试数据目录不存在: {config.TEST_DIR}")
    
    # 收集所有测试图像
    test_image_paths = sorted([str(p) for p in config.TEST_DIR.glob("*.png")])
    
    if len(test_image_paths) == 0:
        raise ValueError("没有发现任何测试图像")
    
    print(f"总共加载 {len(test_image_paths)} 张测试图像")
    
    # 创建测试数据集
    test_dataset = PlantDataset(test_image_paths, labels=None, 
                               transform=get_val_transforms(), is_test=True)
    
    return test_dataset, test_image_paths


def create_data_loaders():
    """
    创建数据加载器
    
    Returns:
        train_loader, val_loader, test_loader, class_to_idx, test_image_paths
    """
    # 加载数据集
    train_dataset, val_dataset, class_to_idx = load_training_data()
    test_dataset, test_image_paths = load_test_data()
    
    # 根据系统资源调整num_workers
    import multiprocessing
    num_workers = min(config.NUM_WORKERS, multiprocessing.cpu_count())
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    print(f"数据加载器创建完成: 训练{len(train_loader)}批, "
          f"验证{len(val_loader)}批, 测试{len(test_loader)}批\n")
    
    return train_loader, val_loader, test_loader, class_to_idx, test_image_paths
