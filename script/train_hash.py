import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from retrieval_net import SalientFeatureHashNet, CenterLoss
from hash_patch_dataset import get_dataloader
import config_hash as cfg

def hash_similarity_loss(query_codes, db_codes, labels):
    """
    计算哈希相似度损失函数
    参数:
        query_codes: 查询图像的哈希码，形状为 (B, hash_bits)，B为批次大小
        db_codes: 数据库图像的哈希码，形状为 (N, hash_bits)，N为数据库大小
        labels: 查询图像的标签，形状为 (B,)
    返回:
        损失值：基于哈希码相似度与标签相似度之间的差异
    """
    # 构建标签相似度矩阵：相同标签的位置为1，不同标签的位置为0
    label_sim = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    
    # 计算哈希码之间的余弦相似度，并归一化到[0,1]范围
    hash_sim = torch.matmul(query_codes, db_codes.t()) / cfg.hash_bits
    
    # 计算均方误差损失
    loss = torch.mean((hash_sim - label_sim) ** 2)
    return loss

def quantization_loss(hash_codes):
    """
    计算量化损失函数
    目的：促使哈希码向二值化值（-1或1）靠近
    参数:
        hash_codes: 模型输出的哈希码
    返回:
        量化损失值
    """
    return torch.mean((hash_codes.abs() - 1) ** 2)

def analyze_training_log(training_log):
    """
    分析训练日志数据，计算关键训练指标
    参数:
        training_log: 包含训练过程中所有指标的DataFrame
    返回:
        包含分析结果的字典，包括最佳epoch、最终性能等指标
    """
    # 获取最佳验证准确率对应的epoch信息
    best_epoch = training_log.loc[training_log['val_acc'].idxmax()]
    final_epoch = training_log.iloc[-1]
    
    # 计算总损失下降百分比
    loss_drop_rate = (training_log['total_loss'].iloc[0] - training_log['total_loss'].iloc[-1]) / training_log['total_loss'].iloc[0] * 100
    
    # 计算验证准确率提升百分比
    acc_improvement = (training_log['val_acc'].iloc[-1] - training_log['val_acc'].iloc[0]) * 100
    
    # 计算收敛速度：损失值下降到初始值90%所需的epoch数
    convergence_threshold = training_log['total_loss'].iloc[0] * 0.9
    convergence_epoch = training_log[training_log['total_loss'] <= convergence_threshold].index[0] + 1
    
    print("\n   训练分析报告:")
    print(f"   最佳Epoch: {best_epoch['epoch']}")
    print(f"   最佳验证准确率: {best_epoch['val_acc']*100:.2f}%")
    print(f"   最终损失值: {final_epoch['total_loss']:.4f}")
    print(f"   损失下降率: {loss_drop_rate:.2f}%")
    print(f"   准确率提升: {acc_improvement:.2f}%")
    print(f"   收敛速度: {convergence_epoch} epochs")
    
    return {
        'best_epoch': best_epoch,
        'final_epoch': final_epoch,
        'loss_drop_rate': loss_drop_rate,
        'acc_improvement': acc_improvement,
        'convergence_epoch': convergence_epoch
    }

def train():
    """
    模型训练主函数
    功能：
    1. 初始化模型、损失函数和优化器
    2. 执行训练循环
    3. 记录训练过程
    4. 保存最佳模型
    5. 生成训练分析报告和可视化图表
    """
    # 设置训练设备
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建训练记录保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.checkpoint_dir, f'training_log_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化训练日志DataFrame
    columns = ['epoch', 'total_loss', 'hash_loss', 'quant_loss', 'cls_loss', 'val_acc']
    training_log = pd.DataFrame(columns=columns)

    # 加载训练和验证数据集
    train_loader, classes = get_dataloader(cfg.train_data, cfg.batch_size, transform_type='train')
    val_loader, _ = get_dataloader(cfg.val_data, cfg.batch_size, transform_type='val')
    num_classes = len(classes)
    print(f"类别数: {num_classes}，类名: {classes}")

    # 初始化模型和损失函数
    model = SalientFeatureHashNet(
        hash_bits=cfg.hash_bits,
        num_classes=num_classes,
        loss_weight=cfg.loss_weight
    ).to(device)
    
    ce_loss_fn = nn.CrossEntropyLoss()  # 分类损失
    center_loss_fn = CenterLoss(num_classes, cfg.hash_bits).to(device)  # 中心损失

    # 配置优化器，为不同参数组设置不同的学习率
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': center_loss_fn.parameters(), 'lr': cfg.lr * 0.5}
    ], lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 初始化训练历史记录字典
    history = {
        'total_loss': [],
        'hash_loss': [],
        'quant_loss': [],
        'cls_loss': [],
        'val_acc': []
    }

    # 训练循环
    best_acc = 0.0
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_hash_loss = 0.0
        total_quant_loss = 0.0
        total_cls_loss = 0.0

        # 训练阶段
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            hash_codes, logits, loss_weight = model(imgs)
            
            # 计算各项损失
            hash_loss = hash_similarity_loss(hash_codes, hash_codes, labels)  # 哈希相似度损失
            quant_loss = quantization_loss(hash_codes)  # 量化损失
            cls_loss = ce_loss_fn(logits, labels)  # 分类损失
            center_loss = center_loss_fn(hash_codes, labels)  # 中心损失

            # 计算总损失（使用动态权重）
            total_batch_loss = (
                loss_weight[0] * hash_loss +    # 哈希相似度损失
                loss_weight[1] * quant_loss +   # 量化损失
                loss_weight[2] * cls_loss +     # 分类损失
                0.01 * center_loss             # 中心损失（固定权重）
            )

            # 反向传播和优化
            total_batch_loss.backward()
            optimizer.step()

            # 累计损失值
            total_loss += total_batch_loss.item()
            total_hash_loss += hash_loss.item()
            total_quant_loss += quant_loss.item()
            total_cls_loss += cls_loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_hash_loss = total_hash_loss / len(train_loader)
        avg_quant_loss = total_quant_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)

        # 记录训练损失
        history['total_loss'].append(avg_loss)
        history['hash_loss'].append(avg_hash_loss)
        history['quant_loss'].append(avg_quant_loss)
        history['cls_loss'].append(avg_cls_loss)

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

        # 记录验证准确率
        history['val_acc'].append(val_acc)

        # 更新训练日志
        epoch_data = pd.DataFrame([{
            'epoch': epoch + 1,
            'total_loss': avg_loss,
            'hash_loss': avg_hash_loss,
            'quant_loss': avg_quant_loss,
            'cls_loss': avg_cls_loss,
            'val_acc': val_acc
        }])
        training_log = pd.concat([training_log, epoch_data], ignore_index=True)

        # 保存训练日志
        training_log.to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)

        # 打印训练进度
        print(f"\n   Epoch [{epoch+1}/{cfg.epochs}]")
        print(f"   损失: {avg_loss:.4f} (哈希: {avg_hash_loss:.4f}, 量化: {avg_quant_loss:.4f}, 分类: {avg_cls_loss:.4f})")
        print(f"   验证准确率: {val_acc*100:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'center_state_dict': center_loss_fn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(cfg.checkpoint_dir, 'best_model.pth'))
            print(f"   最佳模型已保存！（验证准确率: {best_acc*100:.2f}%）")

    print(f"\n   训练完成！最佳验证准确率: {best_acc*100:.2f}%")

    # 分析训练数据
    analysis_results = analyze_training_log(training_log)
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(history['total_loss'], label='总损失', color='red')
    plt.plot(history['hash_loss'], label='哈希损失', color='blue')
    plt.plot(history['quant_loss'], label='量化损失', color='green')
    plt.plot(history['cls_loss'], label='分类损失', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)

    # 绘制验证准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(history['val_acc'], label='验证准确率', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title('验证准确率曲线')
    plt.legend()
    plt.grid(True)

    # 标记最佳模型点
    best_epoch = analysis_results['best_epoch']['epoch'] - 1
    plt.subplot(2, 1, 2)
    plt.plot(best_epoch, history['val_acc'][best_epoch], 'ro', label='最佳模型')
    plt.legend()

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    # 保存分析结果
    analysis_df = pd.DataFrame([{
        'best_epoch': analysis_results['best_epoch']['epoch'],
        'best_val_acc': analysis_results['best_epoch']['val_acc'],
        'final_loss': analysis_results['final_epoch']['total_loss'],
        'loss_drop_rate': analysis_results['loss_drop_rate'],
        'acc_improvement': analysis_results['acc_improvement'],
        'convergence_epoch': analysis_results['convergence_epoch']
    }])
    analysis_df.to_csv(os.path.join(save_dir, 'training_analysis.csv'), index=False)
    
    print(f"   训练曲线已保存至: {os.path.join(save_dir, 'training_curves.png')}")
    print(f"   训练日志已保存至: {os.path.join(save_dir, 'training_log.csv')}")
    print(f"   训练分析已保存至: {os.path.join(save_dir, 'training_analysis.csv')}")

if __name__ == '__main__':
    train()
