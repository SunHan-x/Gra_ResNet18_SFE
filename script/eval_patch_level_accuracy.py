# yolo2hash
import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from retrieval_net import RetrievalNet
from hash_patch_dataset import get_dataloader
import config_hash as cfg
from tqdm import tqdm


def compute_iou(boxA, boxB):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def hamming_distance(query, db):
    return (query != db).sum(dim=1)


def extract_hash_code(model, img, device):
    with torch.no_grad():
        code, _ = model(img.unsqueeze(0).to(device))
        return (code > 0).float().cpu().squeeze(0)


def evaluate_patch_level(gt_json_path, image_dir):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model_yolo = YOLO(r'C:\Workspace_yolo\ultralytics\runs\train\single_defective_4.24_remove_wrongdata_SE\weights\best.pt')
    model_hash = RetrievalNet(cfg.hash_bits, num_classes=3).to(device)
    model_hash.load_state_dict(torch.load(os.path.join(cfg.checkpoint_dir, 'best_hash_model.pth'), map_location=device))
    model_hash.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # 构建数据库
    db_imgs, db_labels, db_codes = [], [], []
    train_loader, class_names = get_dataloader(cfg.train_data, batch_size=cfg.batch_size, transform_type='val')
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            codes, _ = model_hash(imgs)
            db_codes.append((codes > 0).float().cpu())
        db_imgs.extend(imgs.cpu())
        db_labels.extend(labels.tolist())
    db_codes = torch.cat(db_codes)
    db_labels = torch.tensor(db_labels)

    # 加载标注
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    total, correct = 0, 0
    for fname in tqdm(os.listdir(image_dir)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        fpath = os.path.join(image_dir, fname)
        if fname not in gt_data:
            continue

        # 检测目标
        results = model_yolo.predict(fpath, conf=0.3, save=False)
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        gt_items = gt_data[fname]

        for pred_box in pred_boxes:
            x1, y1, x2, y2 = map(int, pred_box)
            matched_label = None
            for gt in gt_items:
                if compute_iou(pred_box, gt['bbox']) > 0.5:
                    matched_label = gt['label']
                    break

            if matched_label is None:
                continue  # 没匹配上

            total += 1
            patch = Image.open(fpath).convert("RGB").crop((x1, y1, x2, y2))
            patch_tensor = transform(patch)
            q_code = extract_hash_code(model_hash, patch_tensor, device)
            top1_label = db_labels[torch.topk(-hamming_distance(q_code, db_codes), 1).indices[0]].item()

            if top1_label == matched_label:
                correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\nPatch-Level Top-1 Accuracy: {acc * 100:.2f}%")
    print(f"匹配Patch总数: {total}, 正确分类: {correct}")

    return acc
