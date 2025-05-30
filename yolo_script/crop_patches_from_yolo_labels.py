import os
import cv2

def crop_patches_from_labels(img_folder, label_folder, save_folder, target_classes=None, img_size=None):
    """
    根据YOLO格式标签裁剪patches
    :param img_folder: 原图文件夹路径
    :param label_folder: 标签文件夹路径
    :param save_folder: 保存patches的根目录
    :param target_classes: 只裁剪特定类别
    :param img_size: 输出patch的统一尺寸
    """

    os.makedirs(save_folder, exist_ok=True)

    img_list = sorted(os.listdir(img_folder))

    for img_name in img_list:
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(img_folder, img_name)
        label_path = os.path.join(label_folder, os.path.splitext(img_name)[0] + '.txt')

        if not os.path.exists(label_path):
            print(f"没找到标签: {label_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center, y_center, bbox_w, bbox_h = map(float, parts[1:5])

            # 如果限定了类别，只保留指定的
            if (target_classes is not None) and (cls_id not in target_classes):
                continue

            # 还原到原图绝对坐标
            x_center *= w
            y_center *= h
            bbox_w *= w
            bbox_h *= h

            x1 = int(x_center - bbox_w / 2)
            y1 = int(y_center - bbox_h / 2)
            x2 = int(x_center + bbox_w / 2)
            y2 = int(y_center + bbox_h / 2)

            # 边界修正
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            # 裁剪patch
            patch = img[y1:y2, x1:x2]

            if img_size is not None:
                patch = cv2.resize(patch, img_size)

            # 保存
            cls_folder = os.path.join(save_folder, str(cls_id))
            os.makedirs(cls_folder, exist_ok=True)
            save_name = f"{os.path.splitext(img_name)[0]}_{idx}.jpg"
            save_path = os.path.join(cls_folder, save_name)
            cv2.imwrite(save_path, patch)

        print(f"   处理完成: {img_name}")

    print("\n   所有图片裁剪完成！")

# 调用
if __name__ == '__main__':
    img_folder = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_patch\OriginImages'      # 原图路径
    label_folder = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_patch\OriginLabels'    # 标签路径
    save_folder = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_patch\hash_dataset'     # 保存patch的根目录
    target_classes = [0, 1, 2]                     
    img_size = (224, 224)                       

    crop_patches_from_labels(img_folder, label_folder, save_folder, target_classes, img_size)
