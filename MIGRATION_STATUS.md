# ISNet 迁移状态跟踪

本文档记录了从 PyTorch 到 Jittor 的迁移状态。

## 核心模块迁移状态

### 1. 模型架构

| 模块 | 旧路径 | 新路径 | 状态 | 说明 |
|------|--------|--------|------|------|
| ISNet 主模型 | old/model/ISNet.py | src/isnet/models/isnet_jittor.py | ✅ 已迁移 | 主要模型结构已迁移 |
| ResNet 骨干网 | old/model/network/Resnet.py | src/isnet/models/resnet_jittor.py | ✅ 已迁移 | 骨干网络已迁移 |
| 公共模块 | old/model/my_functionals/ | src/isnet/models/common_modules.py | ✅ 已迁移 | 包含 BasicBlock, GatedSpatialConv2d 等 |
| TTOA 模块 | old/model/TTOA.py | src/isnet/models/DCNv2/ | ❌ 待迁移 | 需要 Jittor 版本的 DCNv2 |

### 2. 数据加载与预处理

| 模块 | 旧路径 | 新路径 | 状态 | 说明 |
|------|--------|--------|------|------|
| 数据集 | old/model/utils1/dataset_IRSTD1K.py | src/isnet/datasets/sirst_dataset.py | ✅ 已迁移 | 数据集加载逻辑 |
| 数据增强 | old/model/lib/datasets/transform.py | src/isnet/datasets/transforms.py | ✅ 已迁移 | 数据增强和预处理 |

### 3. 损失函数

| 模块 | 旧路径 | 新路径 | 状态 | 说明 |
|------|--------|--------|------|------|
| SoftIoULoss | old/model/loss.py | src/isnet/losses/soft_iou_loss.py | ✅ 已迁移 | IoU 损失 |
| DiceLoss | - | src/isnet/losses/dice_loss.py | ✅ 新增 | Dice 损失 |
| EdgeLoss | - | src/isnet/losses/edge_loss.py | ✅ 新增 | 边缘损失 |
| CombinedLoss | - | src/isnet/losses/combined_loss.py | ✅ 新增 | 组合损失 |

### 4. 工具函数

| 模块 | 旧路径 | 新路径 | 状态 | 说明 |
|------|--------|--------|------|------|
| 可视化 | old/model/utils/edge_utils.py | src/isnet/utils/visualization.py | ✅ 已迁移 | 结果可视化 |
| 视频处理 | - | src/isnet/utils/video_processing.py | ✅ 新增 | 视频处理工具 |
| Jittor 工具 | - | src/isnet/utils/jittor_utils.py | ✅ 新增 | Jittor 特定工具 |

### 5. 训练与评估

| 模块 | 旧路径 | 新路径 | 状态 | 说明 |
|------|--------|--------|------|------|
| 训练脚本 | old/train.py | src/isnet/scripts/train.py | ✅ 已迁移 | 训练流程 |
| 评估脚本 | - | src/isnet/scripts/evaluate.py | ✅ 新增 | 模型评估 |
| 推理脚本 | - | src/scripts/predict_*.py | ✅ 新增 | 图像/视频推理 |

## 待完成工作

1. **DCNv2/TTOA 模块迁移**
   - 需要实现 Jittor 版本的 DCNv2
   - 当前使用 TTOA_stub 占位，需要替换为实际实现

2. **测试与验证**
   - 需要完成单元测试
   - 需要验证模型性能

3. **文档**
   - 完善 API 文档
   - 更新 README 和示例

## 迁移指南

1. **PyTorch 到 Jittor 的主要变化**
   - `torch.Tensor` → `jittor.Var`
   - `torch.nn.Module` → `jittor.nn.Module`
   - 优化器接口调整
   - 数据加载器调整

2. **已知问题**
   - DCNv2 需要特殊处理
   - 某些 PyTorch 操作在 Jittor 中可能没有直接对应

## 测试状态

- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能测试
