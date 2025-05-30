import os
import random
import shutil

def split_dataset(origin_img_dir=r"C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\OriginImages",
                 origin_label_dir=r"C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\OriginLabels",
                 train_ratio=0.7, 
                 valid_ratio=0.2,
                 test_ratio=0.1,
                 random_seed=42):
    """
    按比例复制分割数据集
    
    参数:
        origin_img_dir: 原始图像目录
        origin_label_dir: 原始标签目录
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子(保证可重复性)
    """
    # 设置随机种子保证可重复性
    random.seed(random_seed)

    # 目标路径
    train_img_dir = r"C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\train\images"
    train_label_dir = r"C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\train\labels"
    valid_img_dir = r"C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\valid\images"
    valid_label_dir = r"C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\valid\labels"
    test_img_dir = r"C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\test\images"
    test_label_dir = r"C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\test\labels"

    # 创建目标目录
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # 获取所有文件名(不带扩展名)
    all_files = [f.split('.')[0] for f in os.listdir(origin_img_dir) if f.endswith('.jpg')]
    total_files = len(all_files)
    print(f"发现 {total_files} 个图像文件")

    # 检查图像和标签是否匹配
    for file in all_files:
        if not os.path.exists(os.path.join(origin_label_dir, file+'.txt')):
            raise FileNotFoundError(f"标签文件 {file}.txt 不存在")

    # 随机打乱
    random.shuffle(all_files)

    # 计算分割点
    train_end = int(total_files * train_ratio)
    valid_end = train_end + int(total_files * valid_ratio)

    # 分割文件列表
    train_files = all_files[:train_end]
    valid_files = all_files[train_end:valid_end]
    test_files = all_files[valid_end:]

    # 复制文件函数
    def copy_files(file_list, img_src_dir, label_src_dir, img_dest_dir, label_dest_dir):
        for file in file_list:
            # 复制图像
            shutil.copy2(os.path.join(img_src_dir, file+'.jpg'), 
                        os.path.join(img_dest_dir, file+'.jpg'))
            # 复制标签
            shutil.copy2(os.path.join(label_src_dir, file+'.txt'), 
                        os.path.join(label_dest_dir, file+'.txt'))

    # 执行复制
    copy_files(train_files, origin_img_dir, origin_label_dir, train_img_dir, train_label_dir)
    copy_files(valid_files, origin_img_dir, origin_label_dir, valid_img_dir, valid_label_dir)
    copy_files(test_files, origin_img_dir, origin_label_dir, test_img_dir, test_label_dir)

    # 打印结果
    print("\n分割结果:")
    print(f"训练集: {len(train_files)} 个样本 ({len(train_files)/total_files:.1%})")
    print(f"验证集: {len(valid_files)} 个样本 ({len(valid_files)/total_files:.1%})")
    print(f"测试集: {len(test_files)} 个样本 ({len(test_files)/total_files:.1%})")
    print("\n分割完成! 原始文件保持不变")

if __name__ == "__main__":
    split_dataset()