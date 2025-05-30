# config_hash.py

# 训练集路径
train_data = r'C:\Workspace_yolo\ultralytics\Patch_Dataset\hash_dataset_split\train'

# 验证集路径
val_data = r'C:\Workspace_yolo\ultralytics\Patch_Dataset\hash_dataset_split\val'

# 测试集路径
test_data = r'C:\Workspace_yolo\ultralytics\Patch_Dataset\hash_dataset_split\test'

# 批处理大小
batch_size = 64

# 哈希码长度
hash_bits = 48

# 学习率
lr = 0.0001

# 权重衰减
weight_decay = 5e-4

# 总训练轮数
epochs = 100

# 训练设备设置
device = 'cuda'

# 模型保存目录
checkpoint_dir = './results/checkpoints'

# 动态损失权重（[α, β, γ] 分别为哈希相似损失、量化损失、分类损失权重）
loss_weight = [1.0, 0.1, 0.5]

# SFE模块阈值
sfe_threshold = 0.7

# 评估相关参数
top_k = 5  # Top-K检索评估
map_top_k = 100  # mAP评估的Top-K值
