#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
import os
import sys
import argparse
import yaml
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isnet.utils.jittor_utils import load_config
from isnet.models.isnet_jittor import ISNet
from isnet.datasets.sirst_dataset import SIRSTDataset
from isnet.trainer import Trainer


def main():
    """训练ISNet模型的主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练ISNet模型')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='使用GPU训练')
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               args.config)
    config = load_config(config_path)
    
    # 更新配置中的GPU设置
    if 'hardware' not in config:
        config['hardware'] = {}
    config['hardware']['use_gpu'] = args.gpu
    
    # 创建数据集
    print("创建训练数据集...")
    train_dataset = SIRSTDataset(
        root_dir=config['dataset']['root_dir'],
        dataset_name=config['dataset']['name'],
        split='train',
        augment=config['dataset']['augment'],
        crop_size=config['dataset']['crop_size']
    )
    train_loader = train_dataset.set_attrs(
        batch_size=config['train']['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=config['hardware'].get('num_workers', 4)
    )
    
    print("创建验证数据集...")
    val_dataset = SIRSTDataset(
        root_dir=config['dataset']['root_dir'],
        dataset_name=config['dataset']['name'],
        split='val',
        augment=False,
        crop_size=None  # 不对验证集进行裁剪
    )
    val_loader = val_dataset.set_attrs(
        batch_size=config['train']['batch_size'],
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
