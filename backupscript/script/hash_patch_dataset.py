from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir, batch_size, transform_type='train', num_workers=4):
    """
    返回一个可用于训练/验证的 DataLoader
    :param data_dir: 数据集路径，需是 ImageFolder 格式
    :param batch_size: 批处理大小
    :param transform_type: 数据增强类型（train 或 val/test）
    :param num_workers: 数据加载线程数（推荐设为4或更多）
    :return: 加载器 + 类别列表
    """

    # 数据增强策略（训练时随机裁剪、翻转；验证时只Resize）
    if transform_type == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),             # Resize
            transforms.RandomHorizontalFlip(),         # 随机水平翻转
            transforms.RandomCrop(224, padding=4),     # 随机裁剪
            transforms.ToTensor(),                     # 转张量
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # 归一化到[-1,1]
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(transform_type == 'train'), num_workers=num_workers)
    return loader, dataset.classes  # 返回加载器和类名列表
