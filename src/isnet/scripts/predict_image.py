#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isnet.utils.jittor_utils import load_config, load_checkpoint, setup_jittor
from isnet.models.isnet_jittor import ISNet
from isnet.utils.visualization import make_overlay_image


def preprocess_image(image_path, target_size=None):
    """
    预处理输入图像
    
    Args:
        image_path: 图像路径
        target_size: 目标尺寸 (width, height)
        
    Returns:
        processed_image: 预处理后的图像，形状为[1, 3, H, W]的Jittor张量
        original_image: 原始图像，BGR格式
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 保存原始图像用于可视化
    original_image = image.copy()
    
    # 调整大小如果需要
    if target_size is not None:
        image = cv2.resize(image, target_size)
    
    # 转换为RGB并归一化
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    
    # 转换为Jittor张量格式[1, 3, H, W]
    image = image.transpose(2, 0, 1)  # [3, H, W]
    image = np.expand_dims(image, axis=0)  # [1, 3, H, W]
    image = jt.array(image)
    
    return image, original_image


def main():
    """
    使用ISNet模型预测单张图像的主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用ISNet预测单张图像')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--image', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果保存路径，如果不指定则显示结果')
    parser.add_argument('--size', type=str, default=None,
                        help='处理尺寸 (width,height)，例如 "640,480"')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='分割阈值，默认0.5')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='使用GPU进行推理')
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               args.config)
    config = load_config(config_path)
    
    # 更新配置中的GPU设置
    if 'hardware' not in config:
        config['hardware'] = {}
    config['hardware']['use_gpu'] = args.gpu
    
    # 设置Jittor环境
    setup_jittor(config)
    
    # 解析目标尺寸
    target_size = None
    if args.size is not None:
        width, height = map(int, args.size.split(','))
        target_size = (width, height)
    
    # 创建模型
    model = ISNet(
        layer_blocks=config['model']['layer_blocks'],
        channels=config['model']['channels']
    )
    
    # 加载检查点
    print(f"加载模型检查点: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)
    
    # 预处理图像
    print(f"处理图像: {args.image}")
    input_image, original_image = preprocess_image(args.image, target_size)
    
    # 创建边缘图输入
    edge_map = jt.zeros((1, 3, input_image.shape[2], input_image.shape[3]))
    
    # 设置为评估模式并进行推理
    model.eval()
    with jt.no_grad():
        prediction, edge_output = model(input_image, edge_map)
        prediction = jt.sigmoid(prediction)
    
    # 处理预测结果
    pred_numpy = prediction[0].numpy()  # [1, H, W]
    pred_numpy = pred_numpy[0]  # [H, W]
    
    # 二值化预测结果
    threshold = args.threshold
    pred_binary = (pred_numpy > threshold).astype(np.uint8) * 255
    
    # 如果尺寸不匹配，调整回原始尺寸
    h, w = original_image.shape[:2]
    if pred_binary.shape != (h, w):
        pred_binary = cv2.resize(pred_binary, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 创建可视化结果
    # 生成分割掩码的彩色叠加
    color = (0, 255, 0)  # 绿色，BGR格式
    overlay = make_overlay_image(original_image, pred_binary, color=color)
    
    # 创建并排显示的结果
    result_img = np.hstack((original_image, overlay))
    
    # 保存或显示结果
    if args.output is not None:
        print(f"保存结果到: {args.output}")
        cv2.imwrite(args.output, result_img)
    else:
        # 显示结果
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.title('原始图像')
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(122)
        plt.title('分割结果')
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("预测完成")


if __name__ == "__main__":
    main()
