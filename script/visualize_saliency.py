import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from retrieval_net import SalientFeatureHashNet, ASFE
import matplotlib as mpl
import os
from datetime import datetime
import glob

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_saliency(image_path, model, thresholds=[0.3, 0.5, 0.7], img_size=448, save_dir=None):
    """
    可视化单张图像的显著性特征图
    :param image_path: 输入图像路径
    :param model: 加载好的模型
    :param thresholds: 多个阈值用于对比
    :param img_size: 输入图像大小
    :param save_dir: 保存目录
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    # 提取显著性特征图
    with torch.no_grad():
        backbone_features = model.backbone(input_tensor)
        # 使用较低的阈值获取更详细的显著性图
        model.asfe.threshold = min(thresholds)  # 使用最低阈值获取原始显著性图
        saliency_map = model.asfe(backbone_features)
        
        # 上采样到原始图像大小
        saliency_map = torch.nn.functional.interpolate(
            saliency_map, 
            size=(img_size, img_size), 
            mode='bilinear', 
            align_corners=False
        )
    
    # 转换为numpy数组进行可视化
    saliency_map = saliency_map.squeeze().cpu().numpy()
    
    # 创建图像显示
    n_thresholds = len(thresholds)
    fig = plt.figure(figsize=(15, 3 * (n_thresholds + 1)))
    
    # 创建网格布局
    gs = fig.add_gridspec(n_thresholds + 1, 2, width_ratios=[1, 1])
    
    # 显示原始图像
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=12)
    ax1.axis('off')
    
    # 显示原始显著性图（未二值化）
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(saliency_map, cmap='jet', interpolation='nearest')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Saliency Map (Raw)', fontsize=12)
    ax2.axis('off')
    
    # 显示不同阈值的二值化结果
    for i, threshold in enumerate(thresholds):
        row = i + 1
        ax = fig.add_subplot(gs[row, 1])
        binary_map = (saliency_map > threshold).astype(np.float32)
        ax.imshow(binary_map, cmap='gray', interpolation='nearest')
        ax.set_title(f'Binary Map (Threshold={threshold})', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    
    # 获取图像文件名（不含扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    # 添加时间戳以避免重名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'{image_name}_{timestamp}_saliency.png')
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存
    
    print(f"Processed: {image_path}")
    print(f"Saved to: {save_path}")

def process_images(image_dir, model_path=None, thresholds=[0.3, 0.5, 0.7], img_size=448):
    """
    批量处理目录下的所有图像
    :param image_dir: 图像目录路径
    :param model_path: 模型权重路径
    :param thresholds: 多个阈值用于对比
    :param img_size: 输入图像大小
    """
    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(__file__), 'saliency_maps')
    ensure_dir(save_dir)
    
    # 初始化模型
    model = SalientFeatureHashNet(hash_bits=48, num_classes=3)
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print(f"未找到图像文件在目录: {image_dir}")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每张图像
    for image_path in image_files:
        try:
            visualize_saliency(image_path, model, thresholds, img_size, save_dir)
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            continue
    
    print("所有图像处理完成！")

if __name__ == '__main__':
    # 使用示例
    image_dir = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_patch\hash_dataset_split\test\missing'  # 图像目录
    model_path = r'C:\Workspace_yolo\ultralytics\results\checkpoints_salient_feature_hash\best_model.pth'
    # 批量处理图像
    process_images(image_dir, model_path, thresholds=[0.3, 0.5, 0.7], img_size=448) 