import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from retrieval_net import SalientFeatureHashNet
from hash_patch_dataset import get_dataloader
import config_hash as cfg

def train():
    # 获取设备
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    # 创建保存训练记录的目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.checkpoint_dir, f'training_log_cls_only_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 用于记录训练数据的DataFrame
    columns = ['epoch', 'total_loss', 'val_acc']
    training_log = pd.DataFrame(columns=columns)

    # 加载训练/验证数据
    train_loader, classes = get_dataloader(cfg.train_data, cfg.batch_size, transform_type='train')
    val_loader, _ = get_dataloader(cfg.val_data, cfg.batch_size, transform_type='val')
    num_classes = len(classes)
    print(f"✅ 类别数: {num_classes}，类名: {classes}")

    # 初始化模型与损失函数
    model = SalientFeatureHashNet(
        hash_bits=cfg.hash_bits,
        num_classes=num_classes,
        loss_weight=[0, 0, 1]  # 只使用分类损失
    ).to(device)
    
    ce_loss_fn = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 用于记录损失值的列表
    history = {
        'total_loss': [],
        'val_acc': []
    }

    # 训练过程
    best_acc = 0.0
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            _, logits, _ = model(imgs)
            
            # 只计算分类损失
            loss = ce_loss_fn(logits, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计损失
            total_loss += loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)

        # 记录损失值
        history['total_loss'].append(avg_loss)

        # 验证阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                _, logits, _ = model(imgs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        # 记录损失值
        history['val_acc'].append(val_acc)

        # 将当前epoch的数据添加到DataFrame
        epoch_data = pd.DataFrame([{
            'epoch': epoch + 1,
            'total_loss': avg_loss,
            'val_acc': val_acc
        }])
        training_log = pd.concat([training_log, epoch_data], ignore_index=True)

        # 每个epoch都保存一次CSV文件
        training_log.to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)

        # 打印训练信息
        print(f"\n📅 Epoch [{epoch+1}/{cfg.epochs}]")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Val Acc: {val_acc*100:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(cfg.checkpoint_dir, 'best_model_cls_only.pth'))
            print(f"✅ 最佳模型已保存！（Val Acc: {best_acc*100:.2f}%）")

    print(f"\n🎯 Training completed! Best validation accuracy: {best_acc*100:.2f}%")

    # 绘制损失曲线
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(history['total_loss'], label='Total Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (Classification Only)')
    plt.legend()
    plt.grid(True)

    # 绘制验证准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.grid(True)

    # 保存图表
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

if __name__ == '__main__':
    train() 