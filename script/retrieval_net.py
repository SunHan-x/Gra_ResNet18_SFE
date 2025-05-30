import torch
import torch.nn as nn
import torchvision.models as models

class ASFE(nn.Module):
    """自适应显著性特征提取模块 (Adaptive Salient Feature Extraction)
    
    通过通道注意力机制自适应地突出缺陷区域的显著特征，抑制背景干扰。
    使用阈值机制生成二值掩码，保留高响应区域的特征信息。
    """
    def __init__(self, threshold=0.7):
        """
        Args:
            threshold (float): 特征显著性阈值，用于生成二值掩码
        """
        super(ASFE, self).__init__()
        self.threshold = threshold
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图，形状为 [B, C, H, W]
                B: 批次大小
                C: 通道数
                H: 特征图高度
                W: 特征图宽度
                
        Returns:
            torch.Tensor: 显著性特征图，形状为 [B, 1, H, W]
        """
        A = x.mean(dim=1, keepdim=True)  # 生成通道聚合注意力图 A ∈ [B, 1, H, W]
        M = (A > self.threshold).float()  # 生成自适应二值掩码 M ∈ [B, 1, H, W]
        F_salient = M * A  # 提取显著性区域特征
        return F_salient

class SalientFeatureHashNet(nn.Module):
    """基于显著性特征的细粒度深度哈希图像检索模型
    
    结合全局特征和局部显著性特征，生成紧凑的哈希码用于图像检索。
    使用ResNet18作为骨干网络，通过ASFE模块提取显著性特征。
    """
    def __init__(self, hash_bits=48, num_classes=3, loss_weight=None):
        """
        Args:
            hash_bits (int): 哈希码长度，决定检索精度和存储效率
            num_classes (int): 分类任务中的类别数量
            loss_weight (list): 损失函数权重列表 [α, β, γ]
                α: 哈希相似性损失权重
                β: 量化损失权重
                γ: 分类损失权重
        """
        super(SalientFeatureHashNet, self).__init__()
        
        # 初始化ResNet18骨干网络
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 移除最后的平均池化层和全连接层
        
        # 初始化显著性特征提取模块
        self.asfe = ASFE(threshold=0.7)
        
        # 全局特征提取模块
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 局部特征处理模块
        self.local_fc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),  # 降维处理，减少计算量
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 哈希编码层
        self.hash_layer = nn.Sequential(
            nn.Linear(16 + 512, hash_bits),  # 融合局部特征(16维)和全局特征(512维)
            nn.BatchNorm1d(hash_bits),  # 特征归一化，提高训练稳定性
            nn.Tanh()  # 将输出限制在[-1,1]范围内，便于二值化
        )
        
        # 分类层
        self.classifier = nn.Linear(hash_bits, num_classes)
        
        # 设置损失权重
        self.loss_weight = loss_weight if loss_weight is not None else [1.0, 0.1, 0.5]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入图像，形状为 [B, 3, H, W]
                
        Returns:
            tuple: (hash_code, logits, loss_weight)
                hash_code: 生成的哈希码，形状为 [B, hash_bits]
                logits: 分类预测结果，形状为 [B, num_classes]
                loss_weight: 损失权重列表
        """
        # 1. 提取骨干网络特征
        F_backbone = self.backbone(x)  # [B, 512, H, W]
        
        # 2. 提取显著性特征
        F_local_map = self.asfe(F_backbone)  # [B, 1, H, W]
        F_local = self.local_fc(F_local_map).squeeze(-1).squeeze(-1)  # [B, 16]
        
        # 3. 提取全局特征
        F_global = self.global_pool(F_backbone).squeeze(-1).squeeze(-1)  # [B, 512]
        
        # 4. 特征融合
        F_concat = torch.cat([F_local, F_global], dim=1)  # [B, 528]
        
        # 5. 生成哈希码
        hash_code = self.hash_layer(F_concat)  # [B, hash_bits]
        
        # 6. 分类预测
        logits = self.classifier(hash_code)  # [B, num_classes]
        
        return hash_code, logits, self.loss_weight

class CenterLoss(nn.Module):
    """中心损失函数
    
    通过最小化特征与其对应类别中心的距离，增强特征的类内紧凑性。
    有助于生成更具判别性的哈希码。
    """
    def __init__(self, num_classes, feat_dim):
        """
        Args:
            num_classes (int): 类别数量
            feat_dim (int): 特征维度
        """
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, features, labels):
        """
        Args:
            features (torch.Tensor): 特征向量，形状为 [B, D]
            labels (torch.Tensor): 类别标签，形状为 [B]
                
        Returns:
            torch.Tensor: 中心损失值
        """
        centers_batch = self.centers[labels]
        loss = ((features - centers_batch) ** 2).sum() / features.size(0)
        return loss
