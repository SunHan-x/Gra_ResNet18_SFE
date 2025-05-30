import torch
import torch.nn as nn
import torchvision.models as models

class RetrievalNet(nn.Module):
    def __init__(self, hash_bits=4, num_classes=3):
        """
        :param hash_bits: 哈希输出码长度
        :param num_classes: 类别数（这里是3）
        """
        super(RetrievalNet, self).__init__()

        # 使用预训练的ResNet18作为特征提取骨干
        self.backbone = models.resnet18(pretrained=True)

        # 获取ResNet原始最后全连接层的输入特征数
        in_features = self.backbone.fc.in_features

        # 移除原始分类层
        self.backbone.fc = nn.Identity()

        # 哈希映射层：将ResNet提取到的特征映射为哈希码
        self.hash_layer = nn.Sequential(
            nn.Linear(in_features, hash_bits),
            nn.Tanh()  # 限制输出到[-1, 1]，适合后续二值化为0/1
        )

        # 用于训练分类用的全连接层（用于CrossEntropy）
        self.classifier = nn.Linear(hash_bits, num_classes)

    def forward(self, x):
        feat = self.backbone(x)                # 提取ResNet特征
        hash_code = self.hash_layer(feat)      # 哈希码生成
        logits = self.classifier(hash_code)    # 分类输出
        return hash_code, logits

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        # features: (B, D), labels: (B,)
        centers_batch = self.centers[labels]
        loss = ((features - centers_batch) ** 2).sum() / features.size(0)
        return loss
