# config_hash.py

# 训练集路径（由三个子文件夹构成：broken/flashover/missing）
train_data = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_patch\hash_dataset_split\train'

# 验证集路径
val_data = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_patch\hash_dataset_split\val'

# 测试集路径（用于检索评估）
test_data = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_patch\hash_dataset_split\test'

# 批处理大小（每轮训练处理的样本数）
batch_size = 64

# 哈希码长度，决定最终输出的二进制码位数
hash_bits = 4

# 学习率
lr = 1e-3

# 权重衰减（L2正则项防止过拟合）
weight_decay = 5e-4

# 总训练轮数
epochs = 100

# 训练设备设置（'cuda'表示使用GPU，如果没有可改为'cpu'）
device = 'cuda'

# 模型保存目录
checkpoint_dir = './results/checkpoints_5.8_hash/'
