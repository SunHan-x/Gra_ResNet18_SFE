# === 必要导入 ===
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from retrieval_net import RetrievalNet
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
    生成并保存Top-K图像检索结果图
    :param query_img: 查询图像张量
    :param query_label: 查询标签
    :param db_imgs, db_labels, db_codes: 数据库图像、标签、哈希码
    :param model: 哈希模型
    :param class_names: 类别名列表
    :param save_path: 图像保存路径
    :param topk: 展示前K个检索结果
    """
    model.eval()
    with torch.no_grad():
        img = query_img.unsqueeze(0).to(cfg.device)
        query_code, _ = model(img)
        query_code = (query_code > 0).float().cpu()

    dist = hamming_distance(query_code.squeeze(0), db_codes)
    topk_idx = torch.topk(-dist, topk).indices  # 排序距离最小（最相似）

    retrieved_imgs = [db_imgs[i] for i in topk_idx]
    retrieved_labels = [db_labels[i].item() for i in topk_idx]

    # 可视化绘图
    plt.figure(figsize=(16, 4))
    plt.subplot(1, topk + 1, 1)
    plt.imshow(query_img.permute(1, 2, 0))
    plt.title(f'Query\n{class_names[query_label]}', color='blue')
    plt.axis('off')

    for i in range(topk):
        plt.subplot(1, topk + 1, i + 2)
        plt.imshow(retrieved_imgs[i].permute(1, 2, 0))
        color = 'green' if retrieved_labels[i] == query_label else 'red'
        plt.title(f'Top-{i+1}\n{class_names[retrieved_labels[i]]}', color=color)
        plt.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
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


def main():
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # 加载模型并恢复权重
    model = RetrievalNet(hash_bits=cfg.hash_bits, num_classes=3).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_dir, 'best_hash_model.pth'), map_location=device))
    model.eval()

    # 加载训练集（用于建库）和测试集（用于查询）
    train_loader, class_names = get_dataloader(cfg.train_data, batch_size=cfg.batch_size, transform_type='val')
    test_loader, _ = get_dataloader(cfg.test_data, batch_size=1, transform_type='val')

    # 提取训练集哈希数据库
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

    # 初始化统计
    top1_correct, top5_correct = 0, 0
    query_hashes, query_labels = [], []
    total = 0

    # 遍历测试集做检索评估
    for i, (img, label) in enumerate(test_loader):
        img = img[0]
        label = label[0].item()

        with torch.no_grad():
            query_code, _ = model(img.unsqueeze(0).to(device))
            query_code = (query_code > 0).float().cpu()

        dist = hamming_distance(query_code.squeeze(0), db_codes)
        topk_idx = torch.topk(-dist, k=5).indices
        retrieved_labels = [db_labels[i].item() for i in topk_idx]

        # Top-K统计
        top1_correct += int(retrieved_labels[0] == label)
        top5_correct += int(label in retrieved_labels)

        query_hashes.append(query_code.squeeze(0))
        query_labels.append(label)
        total += 1

        # 保存可视化图像
        save_path = f'./results/visualization_topk/query_{i:04d}_{class_names[label]}.png'
        visualize_topk(img, label, db_imgs, db_labels, db_codes, model, class_names, save_path, topk=5)

    # 拼接所有查询码和标签
    query_hashes = torch.stack(query_hashes)
    query_labels = torch.tensor(query_labels)

    # 计算mAP
    map_score = compute_map(query_hashes, query_labels, db_codes, db_labels, topk=100)

    # 打印最终结果
    print("\n📊 检索评估结果：")
    print(f"Top-1 Accuracy: {top1_correct / total * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5_correct / total * 100:.2f}%")
    print(f"mAP@100       : {map_score * 100:.2f}%")


if __name__ == '__main__':
    main()
