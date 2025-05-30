import os
import sys
# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from retrieval_net import SFDH_FGIR
from hash_patch_dataset import get_dataloader
import config_hash as cfg

def train():
    # 获取设备
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    # 创建保存训练记录的目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.checkpoint_dir, f'training_log_no_loss_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 用于记录训练数据的DataFrame
    columns = ['epoch', 'val_acc']
    training_log = pd.DataFrame(columns=columns)

    # 加载训练/验证数据
    train_loader, classes = get_dataloader(cfg.train_data, cfg.batch_size, transform_type='train')
    val_loader, _ = get_dataloader(cfg.val_data, cfg.batch_size, transform_type='val')
    num_classes = len(classes)
    print(f"✅ 类别数: {num_classes}，类名: {classes}")

    # 初始化模型
    model = SFDH_FGIR(
        hash_bits=cfg.hash_bits,
        num_classes=num_classes,
        loss_weight=[0, 0, 0]  # 不使用任何损失权重
    ).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 用于记录准确率的列表
    history = {
        'val_acc': []
    }

    # 训练过程
    best_acc = 0.0
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        
        # 只进行前向传播，不计算损失
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            _, _, _ = model(imgs)
            
            # 由于没有损失函数，这里直接进行一步优化
            # 使用一个小的随机扰动来更新参数
            for param in model.parameters():
                if param.grad is None:
                    param.grad = torch.randn_like(param) * 1e-6
            optimizer.step()

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

        # 记录准确率
        history['val_acc'].append(val_acc)

        # 将当前epoch的数据添加到DataFrame
        epoch_data = pd.DataFrame([{
            'epoch': epoch + 1,
            'val_acc': val_acc
        }])
        training_log = pd.concat([training_log, epoch_data], ignore_index=True)

        # 每个epoch都保存一次CSV文件
        training_log.to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)

        # 打印训练信息
        print(f"\n📅 Epoch [{epoch+1}/{cfg.epochs}]")
        print(f"   Val Acc: {val_acc*100:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(cfg.checkpoint_dir, 'best_model_no_loss.pth'))
            print(f"✅ 最佳模型已保存！（Val Acc: {best_acc*100:.2f}%）")

    print(f"\n🎯 Training completed! Best validation accuracy: {best_acc*100:.2f}%")

    # 绘制验证准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve (No Loss Function)')
    plt.legend()
    plt.grid(True)

    # 保存图表
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

if __name__ == '__main__':
    train() 