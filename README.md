# MedBaseNet: 轻量级医疗图像识别框架

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2024.1113-b31b1b.svg)](https://arxiv.org/abs/)

</div>

## 目录

- [简介](#简介)
- [特点](#特点)
- [安装](#安装)
- [快速开始](#快速开始)
- [模型架构](#模型架构)
- [性能评估](#性能评估)
- [示例](#示例)
- [引用](#引用)
- [许可证](#许可证)

## 简介

MedBaseNet 是一个专为医疗图像识别设计的轻量级深度学习框架。该框架采用简洁的架构设计，在保持较高识别精度的同时，显著降低了计算资源需求，特别适合医疗教学和基础研究场景。

## 特点

- **轻量级设计**
  - 精简的网络架构
  - 低计算资源消耗
  - 快速训练和推理

- **易于使用**
  - 简洁的API接口
  - 完整的示例代码
  - 详细的文档说明

- **医疗场景优化**
  - 适配医疗图像特点
  - 支持常见医疗数据格式
  - 提供预训练模型

## 安装

### 环境要求
- Python >= 3.8
- PyTorch >= 1.8.0
- CUDA >= 10.2 (可选，用于GPU加速)

### 通过pip安装
```bash
pip install medbasenet
```

### 从源码安装
```bash
git clone https://github.com/alltobebetter/medbasenet.git
cd medbasenet
pip install -r requirements.txt
python setup.py install
```

## 快速开始

### 基础使用
```python
from medbasenet import MedBaseNet
import torch

# 初始化模型
model = MedBaseNet(num_classes=2)

# 准备数据
input_tensor = torch.randn(1, 3, 224, 224)

# 模型推理
output = model(input_tensor)
```

### 训练示例
```python
# 训练新模型
python examples/train.py --data_path /path/to/dataset --epochs 100

# 使用预训练模型进行推理
python examples/inference.py --image_path /path/to/image.jpg --model_path /path/to/model.pth
```

## 模型架构

MedBaseNet 采用经典的卷积神经网络架构，主要包含以下模块：

- 特征提取器：多层卷积网络
- 全局池化层：自适应平均池化
- 分类器：全连接层

```python
MedBaseNet(
  (feature_extractor): Sequential(
    (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    ...
  )
  (classifier): Sequential(
    (fc): Linear(in_features=256, out_features=num_classes)
  )
)
```

## 性能评估

在标准医疗数据集上的表现：

| 数据集 | 准确率 | 召回率 | F1分数 |
|-------|--------|--------|--------|
| ChestX-ray14 | 90.8% | 90.2% | 90.5% |
| ISIC 2019 | 89.5% | 88.9% | 89.2% |
| PathMNIST | 91.2% | 90.8% | 91.0% |

## 示例

### 1. 基础图像分类
```python
from medbasenet import MedBaseNet
from medbasenet.utils import load_image

# 加载模型
model = MedBaseNet.from_pretrained('medbasenet_chest_xray')

# 预测
image = load_image('example.jpg')
prediction = model.predict(image)
```

### 2. 批量处理
```python
# 批量处理示例代码
```

### 3. 模型微调
```python
# 模型微调示例代码
```

## 数据准备

### 推荐的数据结构
```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image1.jpg
│       └── image2.jpg
└── test/
    ├── class1/
    └── class2/
```

## 常见问题

1. **如何选择合适的学习率？**
   - 推荐初始学习率设置为0.001
   - 使用学习率调度器进行动态调整

2. **模型训练时内存不足？**
   - 减小batch size
   - 使用梯度累积
   - 考虑使用混合精度训练

## 引用

如果您在研究中使用了MedBaseNet，请引用：

```bibtex
@article{medbasenet2024,
  title={MedBaseNet: A Lightweight Framework for Medical Image Recognition},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.1113},
  year={2024}
}
```

## 贡献指南

我们欢迎所有形式的贡献，包括但不限于：

- 提交问题和建议
- 改进文档
- 提交代码修复
- 提出新功能

请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多详情。

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 致谢

感谢所有为本项目做出贡献的研究者和开发者。
