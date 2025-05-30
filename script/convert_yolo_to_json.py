import os
import json
from PIL import Image

def yolo_to_json(label_dir, image_dir, output_json):
    """
    将YOLO格式标注转为Patch-Level评估用的JSON格式
    :param label_dir: YOLO标注文件夹（.txt）
    :param image_dir: 原始图像文件夹
    :param output_json: 输出json文件路径
    """

    gt_dict = {}

    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, fname)
        image_name = fname.replace('.txt', '.jpg')
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"⚠️ 图像文件不存在: {image_name}，跳过")
            continue

        img = Image.open(image_path)
        w, h = img.size

        with open(label_path, 'r') as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "label": class_id
            })

        if boxes:
            gt_dict[image_name] = boxes

    # 保存为 JSON
    with open(output_json, 'w') as f:
        json.dump(gt_dict, f, indent=2)

    print(f"✅ 转换完成，保存为: {output_json}")

if __name__ == '__main__':
    label_dir = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\OriginLabels'
    image_dir = r'C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\OriginImages'
    output_json = 'gt_patch.json'
    yolo_to_json(label_dir, image_dir, output_json)
