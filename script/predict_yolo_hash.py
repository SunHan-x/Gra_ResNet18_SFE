import os
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from retrieval_net import RetrievalNet
from hash_patch_dataset import get_dataloader
import config_hash as cfg
import matplotlib.pyplot as plt

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
    return cropped_paths

# ==== 加载哈希模型 ====
def load_hash_model(device):
    model = RetrievalNet(hash_bits=cfg.hash_bits, num_classes=3).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_dir, 'best_hash_model.pth'), map_location=device))
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
        code, _ = model(img)
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
    query_img.save(f'./results/yolo_hash/query_patch_{os.path.basename(query_path)}')
    plt.figure(figsize=(16, 4))

    # 显示查询图
    plt.subplot(1, topk + 1, 1)
    plt.imshow(query_img)
    plt.title("Query Patch", color='blue')
    plt.axis('off')

    # 显示 Top-K 返回图
    for i in range(topk):
        plt.subplot(1, topk + 1, i + 2)

        # Tensor 图像反归一化
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
def main(input_path):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # 加载YOLOv11模型
    model_yolo = YOLO(r'C:\Workspace_yolo\ultralytics\runs\train\single_defective_4.23_remove_wrongdata\weights\best.pt')

    # 加载哈希网络
    model_hash = load_hash_model(device)

    # 构建数据库
    train_loader, class_names = get_dataloader(cfg.train_data, batch_size=cfg.batch_size, transform_type='val')
    db_imgs, db_labels, db_codes = [], [], []
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            codes, _ = model_hash(imgs)
            db_codes.append((codes > 0).float().cpu())
        db_imgs.extend(imgs.cpu())
        db_labels.extend(labels.tolist())
    db_codes = torch.cat(db_codes)

    # 处理输入图像或文件夹
    if os.path.isdir(input_path):
        img_list = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.png'))]
    else:
        img_list = [input_path]

    for img_path in img_list:
        # 缺陷检测+裁剪
        crop_paths = detect_and_crop(model_yolo, img_path, crop_save_dir="./temp/crops")

        # 检索每个patch
        for i, crop_path in enumerate(crop_paths):
            save_path = f'./results/yolo_hash/query_{os.path.basename(img_path).split(".")[0]}_{i}.png'
            visualize_topk(crop_path, db_imgs, db_labels, db_codes, model_hash, class_names, save_path)

if __name__ == '__main__':
    input_path = r"C:\Workspace_yolo\ultralytics\Query"
    main(input_path)
