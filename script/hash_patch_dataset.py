from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir, batch_size, transform_type='train', num_workers=4):
    """
    返回一个可用于训练/验证的 DataLoader
    :param data_dir: 数据集路径
    :param batch_size: 批处理大小
    :param transform_type: 数据增强类型（train 或 val/test）
    :param num_workers: 数据加载线程数
    :return: 加载器 + 类别列表
    """

    # 数据增强策略（训练时随机裁剪、翻转；验证时只Resize）
    if transform_type == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),             # Resize到224x224
            transforms.RandomHorizontalFlip(),         # 随机水平翻转
            transforms.RandomRotation(10),             # 随机旋转±10度
            transforms.ColorJitter(0.2, 0.2, 0.2),    # 随机调整亮度、对比度、饱和度
            transforms.RandomCrop(224, padding=4),     # 随机裁剪
            transforms.ToTensor(),                     # 转张量
            transforms.Normalize(                      # ImageNet标准化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),             # 验证/测试时只做Resize
            transforms.ToTensor(),
            transforms.Normalize(                      # ImageNet标准化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # 创建数据集和加载器
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(transform_type == 'train'),
        num_workers=num_workers,
        pin_memory=True  # 使用固定内存加速数据传输
    )
    
    return loader, dataset.classes  # 返回加载器和类名列表
