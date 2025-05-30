import os
import torch
import torch.nn as nn
import torch.optim as optim
from retrieval_net import RetrievalNet, CenterLoss
from hash_patch_dataset import get_dataloader
import config_hash as cfg

def hash_reg_loss(hash_codes):
    """
    Hash正则项：鼓励输出靠近±1，便于后续二值化
    """
    return ((hash_codes.abs() - 1) ** 2).mean()

def train():
    # 获取设备
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    # 加载训练/验证数据
    train_loader, classes = get_dataloader(cfg.train_data, cfg.batch_size, transform_type='train')
    val_loader, _ = get_dataloader(cfg.val_data, cfg.batch_size, transform_type='val')
    num_classes = len(classes)
    print(f"✅ 类别数: {num_classes}，类名: {classes}")

    # 初始化模型与损失函数
    model = RetrievalNet(hash_bits=cfg.hash_bits, num_classes=num_classes).to(device)
    ce_loss_fn = nn.CrossEntropyLoss()
    center_loss_fn = CenterLoss(num_classes, cfg.hash_bits).to(device)

    # 超参数设置
    lambda_center = 0.01
    lambda_hash = 0.1

    # 优化器（同时优化模型和center loss的参数）
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': center_loss_fn.parameters(), 'lr': cfg.lr * 0.5}  # center loss用较小学习率
    ], lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 训练过程
    best_acc = 0.0
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            hash_codes, logits = model(imgs)

            ce_loss = ce_loss_fn(logits, labels)
            reg_loss = hash_reg_loss(hash_codes)
            c_loss = center_loss_fn(hash_codes, labels)

            total_batch_loss = ce_loss + lambda_center * c_loss + lambda_hash * reg_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                _, logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(f"📅 Epoch [{epoch+1}/{cfg.epochs}] "
              f"- Loss: {avg_train_loss:.4f} "
              f"- Val Acc: {val_acc*100:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'center_state_dict': center_loss_fn.state_dict()
            }, os.path.join(cfg.checkpoint_dir, 'best_hash_model.pth'))
            print(f"✅ 最佳模型已保存！（Val Acc: {best_acc*100:.2f}%）")

    print(f"\n🎯 训练完成！最佳验证准确率: {best_acc*100:.2f}%")

if __name__ == '__main__':
    train()
