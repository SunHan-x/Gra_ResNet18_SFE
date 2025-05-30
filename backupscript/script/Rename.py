import os
import shutil

# 原文件夹路径
src_folder = 'C:\Workspace_yolo\绝缘子数据集_1\绝缘子数据集_1\数据集2\CPLID官方绝缘子数据集\\txt'
# 目标文件夹路径
dst_folder = 'C:\Workspace_yolo\MultiClass_Dataset\OriginLabels'

# 创建目标文件夹（如果不存在）
os.makedirs(dst_folder, exist_ok=True)

# 遍历源目录下的所有文件
for filename in os.listdir(src_folder):
    if filename.endswith('.txt'):
        new_name = 'CPLID' + filename
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, new_name)
        shutil.copyfile(src_path, dst_path)
        print(f'复制并重命名: {filename} -> {new_name}')
