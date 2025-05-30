import os
import random
import shutil

def split_dataset(src_folder, dst_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    将小图数据集按类别划分到 train/val/test 文件夹
    :param src_folder: 原始小图根目录（broken/flashover/missing）
    :param dst_folder: 划分后保存的根目录
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :param seed: 随机种子，保证复现
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    random.seed(seed)

    # 遍历每个类别文件夹
    for cls_name in os.listdir(src_folder):
        cls_path = os.path.join(src_folder, cls_name)
        if not os.path.isdir(cls_path):
            continue

        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]

        # 拷贝到目标文件夹
        for split_name, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_cls_folder = os.path.join(dst_folder, split_name, cls_name)
            os.makedirs(split_cls_folder, exist_ok=True)
            for img_name in split_imgs:
                src_img_path = os.path.join(cls_path, img_name)
                dst_img_path = os.path.join(split_cls_folder, img_name)
                shutil.copyfile(src_img_path, dst_img_path)

        print(f"✅ 类别 [{cls_name}] 划分完成: {n_total}张 -> train:{len(train_imgs)} val:{len(val_imgs)} test:{len(test_imgs)}")

    print("\n🎯 全部类别划分完成！")

# 示例调用
if __name__ == '__main__':
    src_folder = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_patch\hash_dataset'          # 小图原始数据集路径
    dst_folder = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_patch\hash_dataset_split'    # 划分后保存的新路径
    split_dataset(src_folder, dst_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
