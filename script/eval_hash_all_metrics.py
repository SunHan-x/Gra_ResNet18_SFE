# === 必要导入 ===
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from retrieval_net import SalientFeatureHashNet
from hash_patch_dataset import get_dataloader
import config_hash as cfg


def hamming_distance(query, database):
    """
    计算 query 与数据库所有 hash 向量的汉明距离
    :param query: 单个二值哈希向量 (hash_bits,)
    :param database: 多个哈希码 (N, hash_bits)
    :return: 汉明距离 (N,)
    """
    return (query != database).sum(dim=1)


def visualize_topk(query_img, query_label, db_imgs, db_labels, db_codes, model, class_names, save_path, topk=5):
    """
    可视化Top-K检索结果
    """
    plt.figure(figsize=(15, 3))
    
    # 显示查询图像
    plt.subplot(1, topk + 1, 1)
    plt.imshow(denormalize(query_img.permute(1, 2, 0).cpu()))
    plt.title(f'Query\n{class_names[query_label]}')
    plt.axis('off')
    
    # 计算距离并获取Top-K
    with torch.no_grad():
        query_code, _ = model(query_img.unsqueeze(0).to(next(model.parameters()).device))
        query_code = (query_code > 0).float().cpu()
    
    dist = hamming_distance(query_code.squeeze(0), db_codes)
    topk_idx = torch.topk(-dist, k=topk).indices
    
    # 显示检索结果
    for i, idx in enumerate(topk_idx):
        plt.subplot(1, topk + 1, i + 2)
        plt.imshow(denormalize(db_imgs[idx].permute(1, 2, 0).cpu()))
        pred_label = db_labels[idx].item()
        plt.title(f'#{i+1}\n{class_names[pred_label]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_map(query_codes, query_labels, db_codes, db_labels, topk=None):
    """
    计算所有查询的 mean Average Precision
    :param query_codes: 查询哈希码集合
    :param query_labels: 查询标签集合
    :param db_codes: 数据库哈希码集合
    :param db_labels: 数据库标签集合
    :param topk: 只在Top-K排序内评估
    :return: mAP score
    """
    mAP = 0
    query_codes = query_codes.bool()
    db_codes = db_codes.bool()

    for i in range(len(query_codes)):
        query = query_codes[i]
        query_label = query_labels[i]

        dists = hamming_distance(query, db_codes)
        sorted_idxs = torch.argsort(dists)  # 越相似越前

        if topk:
            sorted_idxs = sorted_idxs[:topk]

        correct = (db_labels[sorted_idxs] == query_label).float()

        if correct.sum() == 0:
            continue  # 没有相关样本，跳过该查询

        ranks = torch.arange(1, len(correct) + 1).float()
        precision_at_k = torch.cumsum(correct, dim=0) / ranks
        AP = (precision_at_k * correct).sum() / correct.sum()  # Average Precision
        mAP += AP.item()

    return mAP / len(query_codes)


def denormalize(tensor):
    """
    将 [-1, 1] 区间的张量还原到 [0, 1]
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def main():
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型并恢复权重
    model = SalientFeatureHashNet(hash_bits=cfg.hash_bits, num_classes=3).to(device)
    checkpoint = torch.load(os.path.join(cfg.checkpoint_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载完成！验证准确率: {checkpoint['val_acc']*100:.2f}%")

    # 加载训练集（用于建库）和测试集（用于查询）
    train_loader, class_names = get_dataloader(cfg.train_data, batch_size=cfg.batch_size, transform_type='val')
    test_loader, _ = get_dataloader(cfg.test_data, batch_size=1, transform_type='val')
    print(f"类别: {class_names}")

    # 提取训练集哈希数据库
    print("构建哈希数据库...")
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
    print(f"数据库构建完成！共 {len(db_codes)} 个样本")

    # 初始化统计
    top1_correct, topk_correct = 0, 0
    query_hashes, query_labels = [], []
    total = 0

    # 遍历测试集做检索评估
    print("\n开始检索评估...")
    os.makedirs('./results/visualization_topk', exist_ok=True)
    
    for i, (img, label) in enumerate(test_loader):
        img = img[0]
        label = label[0].item()

        with torch.no_grad():
            query_code, _ = model(img.unsqueeze(0).to(device))
            query_code = (query_code > 0).float().cpu()

        dist = hamming_distance(query_code.squeeze(0), db_codes)
        topk_idx = torch.topk(-dist, k=cfg.top_k).indices
        retrieved_labels = [db_labels[i].item() for i in topk_idx]

        # Top-K统计
        top1_correct += int(retrieved_labels[0] == label)
        topk_correct += int(label in retrieved_labels)

        query_hashes.append(query_code.squeeze(0))
        query_labels.append(label)
        total += 1

        # 保存可视化图像
        save_path = f'./results/visualization_topk/query_{i:04d}_{class_names[label]}.png'
        visualize_topk(img, label, db_imgs, db_labels, db_codes, model, class_names, save_path, topk=cfg.top_k)

    # 拼接所有查询码和标签
    query_hashes = torch.stack(query_hashes)
    query_labels = torch.tensor(query_labels)

    # 计算mAP
    map_score = compute_map(query_hashes, query_labels, db_codes, db_labels, topk=cfg.map_top_k)

    # 打印最终结果
    print("\n检索评估结果：")
    print(f"Top-1 Accuracy: {top1_correct / total * 100:.2f}%")
    print(f"Top-{cfg.top_k} Accuracy: {topk_correct / total * 100:.2f}%")
    print(f"mAP@{cfg.map_top_k}: {map_score * 100:.2f}%")
    print(f"\n评估完成！可视化结果保存在 ./results/visualization_topk/ 目录下")


if __name__ == '__main__':
    main()
