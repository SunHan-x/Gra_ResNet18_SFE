import os
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from retrieval_net import SalientFeatureHashNet
from hash_patch_dataset import get_dataloader
import config_hash as cfg
import matplotlib.pyplot as plt
from datetime import datetime

# ==== 路径配置 ====
# YOLO模型路径
YOLO_MODEL_PATH = r'runs/train/YOLO11_CBAM/weights/best.pt'

# 输入输出路径
SOURCE_IMAGE_DIR = r'predict_image'                  # 源图片文件夹
CROP_SAVE_DIR = r'predict_result/savepatch'          # 裁剪图片保存文件夹
RESULT_SAVE_DIR = r'predict_result/savedata'         # 结果保存文件夹
REPORT_PATH = r'predict_result/detection_report.md'  # 检测报告文件

# 哈希模型相关路径
HASH_MODEL_PATH = r'results\checkpoints_sfdh_fgir\best_model\best_model.pth'  # 哈希模型路径
TRAIN_DATA_PATH = cfg.train_data  # 构建哈希数据库

# ==== 检测缺陷并裁剪patch ====
def detect_and_crop(model_yolo, image_path, crop_save_dir, conf_thresh=0.3):
    img = Image.open(image_path).convert("RGB")
    results = model_yolo.predict(image_path, conf=conf_thresh, save=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    os.makedirs(crop_save_dir, exist_ok=True)
    cropped_paths = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = img.crop((x1, y1, x2, y2))
        crop_path = os.path.join(crop_save_dir, f"{os.path.basename(image_path).split('.')[0]}_crop{idx}.jpg")
        crop.save(crop_path)
        cropped_paths.append(crop_path)
    return cropped_paths, len(boxes)

# ==== 生成检测结果 ====
def write_detection_report(f, image_path, num_defects, crop_paths, save_dir, model_hash, db_imgs, db_labels, db_codes, class_names):
    f.write(f"## {os.path.basename(image_path)}\n\n")
    f.write(f"### 基本信息\n\n")
    f.write(f"- 检测结果：{'发现缺陷' if num_defects > 0 else '未发现缺陷'}\n")
    f.write(f"- 缺陷数量：{num_defects}\n\n")
    
    if num_defects > 0:
        f.write(f"### 缺陷详情\n\n")
        for i, crop_path in enumerate(crop_paths):
            # 获取top1检索结果
            query_code = extract_hash_code(model_hash, crop_path, device=cfg.device)
            dists = hamming_distance(query_code, db_codes)
            top1_idx = torch.topk(-dists, 1).indices[0]
            top1_label = class_names[db_labels[top1_idx]]
            
            retrieval_result_path = os.path.join(RESULT_SAVE_DIR, 'retrieval_results', 
                                               f'query_{os.path.basename(image_path).split(".")[0]}_{i}.png')
            f.write(f"#### 缺陷 #{i+1}\n\n")
            f.write(f"- 缺陷类型：{top1_label}\n")
            f.write(f"- 缺陷图像：![缺陷#{i+1}]({os.path.relpath(crop_path, os.path.dirname(REPORT_PATH))})\n")
            f.write(f"- 检索结果：![检索结果#{i+1}]({os.path.relpath(retrieval_result_path, os.path.dirname(REPORT_PATH))})\n\n")
    else:
        f.write(f"### 备注\n\n")
        f.write(f"该图像未检测到任何缺陷。\n\n")
    f.write("---\n\n")  # 添加分隔线

# ==== 加载哈希模型 ====
def load_hash_model(device):
    model = SalientFeatureHashNet(hash_bits=cfg.hash_bits, num_classes=3).to(device)
    checkpoint = torch.load(HASH_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# ==== 提取哈希码 ====
def extract_hash_code(model, img_path, device):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        code, _, _ = model(img)
        code = (code > 0).float().cpu()
    return code.squeeze(0)

# ==== 汉明距离 ====
def hamming_distance(query, database):
    return (query != database).sum(dim=1)

def denorm(tensor):
    """
    将 [-1, 1] 区间的张量还原到 [0, 1]
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def visualize_topk(query_path, db_imgs, db_labels, db_codes, model, class_names, save_path, topk=5):
    query_code = extract_hash_code(model, query_path, device=cfg.device)
    dists = hamming_distance(query_code, db_codes)
    topk_idx = torch.topk(-dists, topk).indices

    retrieved_imgs = [db_imgs[i] for i in topk_idx]
    retrieved_labels = [db_labels[i] for i in topk_idx]

    # 读取查询图像
    query_img = Image.open(query_path).convert("RGB")
    query_save_path = os.path.join(RESULT_SAVE_DIR, 'query_patches', f'query_patch_{os.path.basename(query_path)}')
    os.makedirs(os.path.dirname(query_save_path), exist_ok=True)
    query_img.save(query_save_path)
    
    plt.figure(figsize=(16, 4))

    # 显示查询图
    plt.subplot(1, topk + 1, 1)
    plt.imshow(query_img)
    plt.title("Query Patch", color='blue')
    plt.axis('off')

    # 显示 Top-K 返回图
    for i in range(topk):
        plt.subplot(1, topk + 1, i + 2)
        img_vis = denorm(retrieved_imgs[i]).permute(1, 2, 0).numpy()
        plt.imshow(img_vis)
        label_text = class_names[retrieved_labels[i]]
        plt.title(f'Top-{i+1}\n{label_text}', fontsize=10)
        plt.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==== 主函数 ====
def main():
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # 加载YOLOv11模型
    model_yolo = YOLO(YOLO_MODEL_PATH)

    # 加载哈希网络
    model_hash = load_hash_model(device)

    # 构建数据库
    train_loader, class_names = get_dataloader(TRAIN_DATA_PATH, batch_size=cfg.batch_size, transform_type='val')
    db_imgs, db_labels, db_codes = [], [], []
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            codes, _, _ = model_hash(imgs)
            db_codes.append((codes > 0).float().cpu())
        db_imgs.extend(imgs.cpu())
        db_labels.extend(labels.tolist())
    db_codes = torch.cat(db_codes)

    # 处理输入图像文件夹
    img_list = [os.path.join(SOURCE_IMAGE_DIR, f) for f in os.listdir(SOURCE_IMAGE_DIR) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # 创建报告文件
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# 缺陷检测报告\n\n")
        f.write("---\n\n")

        for img_path in img_list:
            # 缺陷检测+裁剪
            crop_paths, num_defects = detect_and_crop(model_yolo, img_path, CROP_SAVE_DIR)

            # 写入检测报告
            write_detection_report(f, img_path, num_defects, crop_paths, os.path.dirname(REPORT_PATH),
                                 model_hash, db_imgs, db_labels, db_codes, class_names)

            # 检索每个patch
            for i, crop_path in enumerate(crop_paths):
                save_path = os.path.join(RESULT_SAVE_DIR, 'retrieval_results', 
                                       f'query_{os.path.basename(img_path).split(".")[0]}_{i}.png')
                visualize_topk(crop_path, db_imgs, db_labels, db_codes, model_hash, class_names, save_path)

if __name__ == '__main__':
    main()
