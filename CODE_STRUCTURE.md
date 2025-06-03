# ISNet 代码结构文档

本文档详细记录了 ISNet 项目的代码结构和各文件功能。

## 1. 核心模型

### 1.1 主模型

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `old/model/ISNet.py` | ISNet 主模型实现 (PyTorch) | 待迁移 | 包含完整的 ISNet 架构, TTOA, DCNv2 |
| `src/isnet/models/isnet_jittor.py` | Jittor 版本的 ISNet 实现 | 已迁移 | Uses `TTOA` from `dcnv2_jittor.py` for DCNv2-based feature aggregation. |
| `old/model/TTOA.py` | 目标感知注意力模块 | 待迁移 | 依赖 DCNv2 |
| `old/model/DCNv2/dcn_v2.py` | DCNv2 实现 | 待迁移 | 需要 Jittor 版本 |

### 1.2 骨干网络

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `old/model/network/Resnet.py` | ResNet 骨干网络 | 已迁移 | 迁移到 `resnet_jittor.py` |
| `src/isnet/models/resnet_jittor.py` | Jittor 版本的 ResNet | 已迁移 | 适配 Jittor 接口 |
| `old/model/network/SEresnext.py` | SE-ResNeXt 网络 | 未迁移 | 可选组件 |
| `src/isnet/models/dcnv2_jittor.py` | Jittor DCNv2 和 TTOA 实现 | 已迁移 | Contains `DeformConv2D` and `TTOA` (Texture Transfer and Offset Aggregation) module. |

## 2. 数据加载与处理

### 2.1 数据集

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `old/model/lib/datasets/transform.py` | 数据增强和预处理 | 已迁移 | 迁移到 `transforms.py` |
| `old/model/utils1/dataset_IRSTD1K.py` | IRSTD-1K 数据集加载 | 已迁移 | 迁移到 `sirst_dataset.py` |
| `src/isnet/datasets/sirst_dataset.py` | Jittor 版本的数据集加载 | 已迁移 | 支持 IRSTD-1K |
| `src/isnet/datasets/transforms.py` | Jittor 版本的数据增强 | 已迁移 | 适配 Jittor 接口 |

### 2.2 数据增强

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `old/model/my_functionals/custom_functional.py` | 自定义函数操作 | 部分迁移 | 需要 Jittor 实现 |

## 3. 损失函数

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `old/model/loss.py` | SoftIoU 损失 | 已迁移 | 迁移到 `soft_iou_loss.py` |
| `old/model/loss1.py` | 广义 Wasserstein Dice 损失 | 未迁移 | 可选损失函数 |
| `src/isnet/losses/soft_iou_loss.py` | Jittor 版本 SoftIoU | 已迁移 | 适配 Jittor 接口 |
| `src/isnet/losses/dice_loss.py` | Dice 损失 | 新增 | Jittor 实现 |
| `src/isnet/losses/edge_loss.py` | 边缘损失 | 新增 | Jittor 实现 |
| `src/isnet/losses/combined_loss.py` | 组合损失函数 | 新增 | 支持多任务学习 |

## 4. 工具函数

### 4.1 可视化

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `old/model/utils/edge_utils.py` | PyTorch 边缘处理和可视化工具 | 已迁移 | 部分功能迁移到 `visualization.py` |
| `src/isnet/utils/video_processing.py` | 视频处理工具 | 新增 | 支持视频输入/输出 |

### 4.2 工具类

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `old/model/utils/AttrDict.py` | 属性字典 | 已迁移 | 迁移到 Jittor 工具类 |
| `src/isnet/utils/jittor_utils.py` | Jittor 工具函数 | 新增 | 常用工具函数 |
| `src/isnet/utils/gradient_utils.py` | 梯度计算工具 | 新增 | Contains `GetGradientNopadding` for edge map generation. |

## 5. 训练与评估

### 5.1 训练

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `old/train.py` | 训练脚本 | 已迁移 | 迁移到 `scripts/train.py` |
| `src/isnet/scripts/train.py` | Jittor 训练脚本 | 已迁移 | Manages training loop, loss calculation (including combined edge loss), metrics, checkpoints. |

### 5.2 评估与推理

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `src/isnet/scripts/evaluate.py` | 模型评估 | 新增 | 支持多种评估指标 |
| `src/scripts/predict_image.py` | 图像推理 | 新增 | 支持单张图像推理 |
| `src/scripts/predict_video.py` | 视频推理 | 新增 | 支持视频流处理 |

## 6. 配置与工具

### 6.1 配置文件

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `old/model/config.py` | 配置文件 | 已迁移 | 迁移到 YAML 配置 |
| `src/isnet/configs/default_config.yaml` | 默认配置 | 新增 | 使用 YAML 格式 |

## 7. 测试

### 7.1 单元测试

| 文件路径 | 功能描述 | 迁移状态 | 备注 |
|----------|----------|----------|------|
| `src/isnet/models/test_isnet_jittor.py` | 模型测试 | 新增 | 基本功能测试 |

## 8. 文档

### 8.1 说明文档

| 文件路径 | 功能描述 | 状态 | 备注 |
|----------|----------|------|------|
| `README.md` | 项目说明 | 待更新 | 需要更新 Jittor 相关内容 |
| `MIGRATION_STATUS.md` | 迁移状态 | 新增 | 跟踪迁移进度 |
| `CODE_STRUCTURE.md` | 代码结构 | 新增 | 本文档 |

## 9. 依赖管理

### 9.1 环境配置

| 文件路径 | 功能描述 | 状态 | 备注 |
|----------|----------|------|------|
| `requirements.txt` | Python 依赖 | 待更新 | 需要更新 Jittor 依赖 |
| `environment.yml` | Conda 环境 | 待创建 | 推荐使用 |
