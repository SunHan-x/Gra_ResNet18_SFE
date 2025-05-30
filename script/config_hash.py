# config_hash.py

# 训练集路径（由三个子文件夹构成：broken/flashover/missing）
train_data = r'C:\Workspace_yolo\ultralytics\Patch_Dataset\hash_dataset_split\train'

# 验证集路径
val_data = r'C:\Workspace_yolo\ultralytics\Patch_Dataset\hash_dataset_split\val'

# 测试集路径（用于检索评估）
test_data = r'C:\Workspace_yolo\ultralytics\Patch_Dataset\hash_dataset_split\test'

# 批处理大小（每轮训练处理的样本数）
batch_size = 64

# 哈希码长度，决定最终输出的二进制码位数（论文推荐 48）
hash_bits = 48

# 学习率（论文推荐 0.0001）
lr = 0.0001

# 权重衰减（L2正则项防止过拟合）
weight_decay = 5e-4

# 总训练轮数
epochs = 100

# 训练设备设置（'cuda'表示使用GPU，如果没有可改为'cpu'）
device = 'cuda'

# 模型保存目录
checkpoint_dir = './results/checkpoints_sfdh_fgir/'

# 动态损失权重（论文推荐 [α, β, γ] 分别为哈希相似损失、量化损失、分类损失权重）
loss_weight = [1.0, 0.1, 0.5]

# SSFE模块阈值（用于生成显著性掩码）
ssfe_threshold = 0.7

# 评估相关参数
top_k = 5  # Top-K检索评估
map_top_k = 100  # mAP评估的Top-K值
