import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from retrieval_net import RetrievalNet
from hash_patch_dataset import get_dataloader
import config_hash as cfg

def tsne_visualize():
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # 加载测试数据
    test_loader, class_names = get_dataloader(cfg.test_data, batch_size=cfg.batch_size, transform_type='val')

    model = RetrievalNet(hash_bits=cfg.hash_bits, num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_dir, 'best_hash_model.pth'), map_location=device))
    model.eval()

    all_codes = []
    all_labels = []

    # 提取全部哈希码 + 标签
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            hash_codes, _ = model(imgs)
            hash_codes = hash_codes.cpu().numpy()
            all_codes.append(hash_codes)
            all_labels.append(labels.numpy())

    all_codes = np.concatenate(all_codes, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 2维降维
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    codes_2d = tsne.fit_transform(all_codes)

    # 可视化
    plt.figure(figsize=(10, 8))
    colors = ['#FF6347', '#1E90FF', '#32CD32', '#FFD700', '#BA55D3', '#FF8C00']  # 可扩展
    for class_idx in range(len(class_names)):
        indices = (all_labels == class_idx)
        plt.scatter(codes_2d[indices, 0], codes_2d[indices, 1],
                    label=class_names[class_idx],
                    s=40, alpha=0.8, c=colors[class_idx % len(colors)])

    plt.legend()
    plt.title('t-SNE Visualization of Hash Codes')
    plt.xticks([])
    plt.yticks([])
    os.makedirs('./results/tsne', exist_ok=True)
    plt.savefig('./results/tsne/hash_tsne.png', dpi=300)
    plt.show()
    print("✅ t-SNE 图已保存至 ./results/tsne/hash_tsne.png")

if __name__ == '__main__':
    tsne_visualize()
