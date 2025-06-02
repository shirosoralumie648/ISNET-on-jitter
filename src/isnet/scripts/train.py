#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
import os
import sys
import argparse
import yaml
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 使用绝对导入
from isnet.utils.jittor_utils import load_config
from isnet.models.isnet_jittor import ISNet
from isnet.datasets.sirst_dataset import SirstDataset
from isnet.trainer import Trainer


def main():
    """训练ISNet模型的主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练ISNet模型')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='使用GPU训练')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='强制使用CPU训练')
    args = parser.parse_args()
    
    # 加载配置文件
    # 引用配置文件的路径处理
    # 如果是绝对路径，直接使用
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        # 如果是相对路径，则从当前工作目录开始解析
        config_path = os.path.abspath(args.config)
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 更新配置中的GPU设置
    if 'hardware' not in config:
        config['hardware'] = {}
    # 如果指定了--cpu参数，则强制使用CPU
    if args.cpu:
        config['hardware']['use_gpu'] = False
        print("强制使用CPU训练...")
    else:
        config['hardware']['use_gpu'] = args.gpu
    
    # 创建数据集
    print("创建训练数据集...")
    train_dataset = SirstDataset(
        root_dir=config['dataset']['root_dir'],
        dataset_name=config['dataset']['name'],
        split='train',
        augmentations=config['dataset'].get('augmentations', False),
        img_size=config['dataset'].get('img_size', [256, 256])
    )
    train_loader = train_dataset.set_attrs(
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=config['hardware'].get('num_workers', 4)
    )
    
    print("创建验证数据集...")
    val_dataset = SirstDataset(
        root_dir=config['dataset']['root_dir'],
        dataset_name=config['dataset']['name'],
        split='val',
        augmentations=False,
        img_size=config['dataset'].get('img_size', [256, 256])  # 不对验证集进行裁剪
    )
    val_loader = val_dataset.set_attrs(
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config['hardware'].get('num_workers', 4)
    )
    
    # 创建模型
    print("创建ISNet模型...")
    model = ISNet(
        layer_blocks=config['model']['layer_blocks'],
        channels=config['model']['channels']
    )
    
    # 创建训练器
    print("初始化训练器...")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 开始训练
    print("开始训练模型...")
    best_metric = trainer.train(resume_from=args.resume)
    print(f"训练完成, 最佳验证指标: {best_metric:.4f}")
    
    # 绘制训练过程图表
    log_dir = os.path.join(trainer.checkpoint_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    plot_path = os.path.join(log_dir, 'training_progress.png')
    trainer.plot_training_progress(save_path=plot_path)
    

if __name__ == "__main__":
    main()
