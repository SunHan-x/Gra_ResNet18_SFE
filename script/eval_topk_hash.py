import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from hash_patch_dataset import get_dataloader
from retrieval_net import RetrievalNet
import config_hash as cfg

# 反归一化函数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def denormalize(tensor_img):
    tensor_img = tensor_img.clone()
    for c in range(3):
        tensor_img[c] = tensor_img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return torch.clamp(tensor_img, 0, 1)

def hamming_distance(query, database):
    return (query != database).sum(dim=1)

def visualize_topk(query_img, query_label, db_imgs, db_labels, db_codes, model, class_names, save_path, topk=5):
    with torch.no_grad():
        model.eval()
        img = query_img.unsqueeze(0).to(cfg.device)
        query_code, _ = model(img)
        query_code = (query_code > 0).float().cpu()

    dist = hamming_distance(query_code.squeeze(0), db_codes)
    topk_idx = torch.topk(-dist, topk).indices

    retrieved_imgs = [db_imgs[i] for i in topk_idx]
    retrieved_labels = [db_labels[i].item() for i in topk_idx]

    plt.figure(figsize=(16, 4))

    # Query 图像
    plt.subplot(1, topk + 1, 1)
    plt.imshow(denormalize(query_img).permute(1, 2, 0))
    plt.title(f'Query\n{class_names[query_label]}', color='blue')
    plt.axis('off')

    # Top-K 检索图像
    for i in range(topk):
        plt.subplot(1, topk + 1, i + 2)
        plt.imshow(denormalize(retrieved_imgs[i]).permute(1, 2, 0))
        correct = (retrieved_labels[i] == query_label)
        color = 'green' if correct else 'red'
        plt.title(f'Top-{i+1}\n{class_names[retrieved_labels[i]]}', color=color)
        plt.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # 加载训练集建立哈希库
    train_loader, class_names = get_dataloader(cfg.train_data, batch_size=cfg.batch_size, transform_type='val')
    test_loader, _ = get_dataloader(cfg.test_data, batch_size=1, transform_type='val')

    model = RetrievalNet(hash_bits=cfg.hash_bits, num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_dir, 'best_hash_model.pth'), map_location=device))
    model.eval()

    # 建立数据库
    db_imgs, db_labels, db_codes = [], [], []
    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            hash_codes, _ = model(imgs)
            db_imgs.extend(imgs.cpu())
            db_labels.extend(labels.cpu())
            db_codes.append((hash_codes > 0).float().cpu())

    db_codes = torch.cat(db_codes)
    db_labels = torch.tensor(db_labels)

    # 遍历测试集
    top1_correct, top5_correct, total = 0, 0, 0
    for i, (imgs, labels) in enumerate(test_loader):
        query_img = imgs[0]
        query_label = labels[0].item()

        with torch.no_grad():
            img = query_img.unsqueeze(0).to(device)
            query_code, _ = model(img)
            query_code = (query_code > 0).float().cpu()

        dist = hamming_distance(query_code.squeeze(0), db_codes)
        topk_idx = torch.topk(-dist, k=5).indices
        retrieved_labels = [db_labels[j].item() for j in topk_idx]

        # Top-1 / Top-5 统计
        if retrieved_labels[0] == query_label:
            top1_correct += 1
        if query_label in retrieved_labels:
            top5_correct += 1
        total += 1

        # 保存检索图像
        save_path = f'./results/visualization_topk/query_{i:04d}_{class_names[query_label]}.png'
        visualize_topk(query_img, query_label, db_imgs, db_labels, db_codes, model, class_names, save_path)

    # 输出整体准确率
    print(f"Top-1 Accuracy: {100.0 * top1_correct / total:.2f}%")
    print(f"Top-5 Accuracy: {100.0 * top5_correct / total:.2f}%")

if __name__ == '__main__':
    main()
