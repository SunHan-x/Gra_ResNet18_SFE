```mermaid
graph LR
    subgraph 输入
        A[输入图像<br/>3×H×W] --> B[ResNet18主干网络]
    end

    subgraph ResNet18主干网络
        B --> C[特征提取<br/>512维特征向量]
    end

    subgraph 显著性特征提取模块
        C --> D[通道注意力<br/>生成显著性特征]
        D --> E[局部特征<br/>16维]
    end

    subgraph 特征融合与哈希编码
        C --> F1[全局特征<br/>512维]
        E --> F2[局部特征<br/>16维]
        F1 --> G[特征拼接<br/>528维]
        F2 --> G
        G --> H[哈希编码层<br/>48位哈希码]
    end

    subgraph 多损失优化
        H --> I[总损失函数<br/>L = αLsim + βLquant + γLcls + λLcenter]
    end

    subgraph 输出
        H --> J[二值化哈希码<br/>用于相似图像检索]
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style C fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style D fill:#bfb,stroke:#333,stroke-width:2px,color:#000
    style F1 fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style F2 fill:#bfb,stroke:#333,stroke-width:2px,color:#000
    style G fill:#fbb,stroke:#333,stroke-width:2px,color:#000
    style H fill:#fbb,stroke:#333,stroke-width:2px,color:#000
    style I fill:#fbf,stroke:#333,stroke-width:2px,color:#000
    style J fill:#f9f,stroke:#333,stroke-width:2px,color:#000

    %% 设置所有文本为黑色
    classDef default fill:#fff,stroke:#333,stroke-width:2px,color:#000
```

## 模型架构说明

该模型架构图展示了基于ResNet18的深度哈希图像检索模型的关键结构，主要包含以下核心组件及其输入输出维度：

1. **输入层**：
   - 输入：3通道图像 (3×H×W)

2. **ResNet18主干网络**：
   - 输入：3×H×W
   - 输出：512维特征向量

3. **显著性特征提取模块**：
   - 输入：512维特征向量
   - 输出：16维局部特征

4. **特征融合与哈希编码**：
   - 输入1：全局特征（来自ResNet18，512维）
   - 输入2：局部特征（来自显著性特征提取模块，16维）
   - 融合：将两种特征拼接得到528维特征向量
   - 输出：48位哈希码

5. **多损失优化**：
   - 总损失函数：L = αLsim + βLquant + γLcls + λLcenter
   - 其中：Lsim为哈希相似度损失，Lquant为量化损失，Lcls为分类损失，Lcenter为中心损失

6. **输出层**：
   - 二值化哈希码：用于相似图像检索，通过计算汉明距离实现快速检索 