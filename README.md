# ResNet18_SFE

毕业设计：基于YOLO注意力增强与哈希检索的绝缘子缺陷检测方法

## 项目结构详解

### 核心代码目录 (`ultralytics/`)
- `models/`: 包含YOLO模型定义
- `nn/`: 神经网络相关的基础组件和工具
- `utils/`: 通用工具函数，包括数据处理、可视化等
- `engine/`: 训练和推理引擎的核心实现
- `data/`: 数据加载和预处理相关代码
- `cfg/`: 模型配置文件
- `trackers/`: 目标跟踪相关实现
- `solutions/`: 特定任务的解决方案
- `hub/`: 模型权重和预训练模型管理
- `assets/`: 项目资源文件

### 脚本目录 (`script/`)
- `train_hash.py`: 哈希网络训练脚本
- `retrieval_net.py`: 哈希检索网络结构
- `visualize_saliency.py`: 灰度图显著性图可视化工具
- `eval_hash_all_metrics.py`: 哈希模型评估脚本
- `predict_yolo_hash.py`: YOLO+哈希联合预测
- `hash_patch_dataset.py`: 图像块数据集处理
- `convert_yolo_to_json.py`: YOLO格式转JSON工具
- `config_hash.py`: 哈希网络配置文件
- `dataprocess/`: 数据预处理相关脚本

### 其他重要目录
- `checkpoints/`: 存储训练好的模型权重
- `results/`: 存储实验结果和输出
- `examples/`: 示例代码和使用案例
- `tests/`: 单元测试和集成测试
- `docs/`: 项目文档
- `docker/`: Docker相关配置文件
- `build/`: 构建输出目录

### 配置文件
- `requirements.txt`: 项目依赖包列表

## 数据集
项目使用的数据集通过以下链接下载：
https://pan.quark.cn/s/6c532e1fb33a

## 项目简介

本项目实现了一个基于YOLO与ResNet18骨干网络的目标检测与分类系统。

## 主要功能模块

- 特征提取
- 目标检测
- 图像检索
- 可视化工具
- 评估系统

## 技术特点

- 采用ResNet18作为骨干网络，平衡性能和效率
- 集成显著性特征提取模块提升特征表达能力
- 利用哈希编码，实现高效检索

## 环境要求

- Python 3.8+
- CUDA 11.6+
- PyTorch 2.6.0
- 其他依赖见 requirements.txt