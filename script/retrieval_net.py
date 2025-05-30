import torch
import torch.nn as nn
import torchvision.models as models

class ASFE(nn.Module):
    """自适应显著性特征提取模块（Adaptive Salient Feature Extraction Module）
    通过通道注意力机制自适应地突出缺陷区域的显著特征，抑制背景干扰
    """
    def __init__(self, threshold=0.7):
        super(ASFE, self).__init__()
        self.threshold = threshold
        
    def forward(self, x):
        # x: B x C x H x W
        A = x.mean(dim=1, keepdim=True)  # 通道聚合注意力图 A ∈ B x 1 x H x W
        M = (A > self.threshold).float()  # 自适应二值掩码 ∈ B x 1 x H x W
        F_salient = M * A  # 显著性区域特征图
        return F_salient  # 返回用于局部特征抽取

class SFDH_FGIR(nn.Module):
    """基于显著性特征的细粒度深度哈希图像检索模型（轻量版）"""
    def __init__(self, hash_bits=48, num_classes=3, loss_weight=None):
        """
        :param hash_bits: 哈希码长度
        :param num_classes: 类别数
        :param loss_weight: 损失权重 [α, β, γ]
        """
        super(SFDH_FGIR, self).__init__()
        
        # 使用预训练的ResNet18作为骨干网络（更轻量）
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 去掉avgpool和fc层
        
        # 自适应显著性特征提取模块
        self.asfe = ASFE(threshold=0.7)
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 局部特征处理（ResNet18输出通道为512，比ResNet50的2048小）
        self.local_fc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),  # 输入通道为1（ASFE输出），输出通道减小
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 特征融合后的哈希层（输入维度相应调整）
        self.hash_layer = nn.Sequential(
            nn.Linear(16 + 512, hash_bits),  # 16(局部) + 512(全局，ResNet18的特征维度)
            nn.BatchNorm1d(hash_bits),  # 保证位平衡
            nn.Tanh()  # 输出范围[-1,1]
        )
        
        # 分类层
        self.classifier = nn.Linear(hash_bits, num_classes)
        
        # 动态损失权重
        self.loss_weight = loss_weight if loss_weight is not None else [1.0, 0.1, 0.5]  # [α, β, γ]

    def forward(self, x):
        # 1. 提取ResNet特征
        F_backbone = self.backbone(x)  # (B, 512, H, W)，ResNet18输出通道为512
        
        # 2. 显著性特征提取
        F_local_map = self.asfe(F_backbone)  # (B, 1, H, W)
        F_local = self.local_fc(F_local_map).squeeze(-1).squeeze(-1)  # (B, 16)
        
        # 3. 全局特征提取
        F_global = self.global_pool(F_backbone).squeeze(-1).squeeze(-1)  # (B, 512)
        
        # 4. 特征融合
        F_concat = torch.cat([F_local, F_global], dim=1)  # (B, 528)
        
        # 5. 生成哈希码
        hash_code = self.hash_layer(F_concat)  # (B, hash_bits)
        
        # 6. 分类预测
        logits = self.classifier(hash_code)  # (B, num_classes)
        
        return hash_code, logits, self.loss_weight

class CenterLoss(nn.Module):
    """中心损失，用于特征聚类"""
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, features, labels):
        # features: (B, D), labels: (B,)
        centers_batch = self.centers[labels]
        loss = ((features - centers_batch) ** 2).sum() / features.size(0)
        return loss
