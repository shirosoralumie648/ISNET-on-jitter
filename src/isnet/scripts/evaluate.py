#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isnet.utils.jittor_utils import load_config, load_checkpoint, setup_jittor
from isnet.models.isnet_jittor import ISNet
from isnet.datasets.sirst_dataset import SIRSTDataset
from isnet.trainer import Trainer
from isnet.metrics.iou_metrics import IoUMetric, PD_FA, ROCMetric


def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """
    绘制ROC曲线
    
    Args:
        fpr: 假阳性率
        tpr: 真阳性率
        roc_auc: ROC曲线下面积
        save_path: 保存路径
    """
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('ISNet ROC曲线')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"ROC曲线已保存到: {save_path}")
    
    plt.show()


def plot_pd_fa(fa_values, pd_values, fa_threshold=None, pd_threshold=None, save_path=None):
    """
    绘制PD-FA曲线
    
    Args:
        fa_values: FA值列表
        pd_values: PD值列表
        fa_threshold: FA阈值行
        pd_threshold: PD阈值行
        save_path: 保存路径
    """
    plt.figure(figsize=(8, 8))
    plt.plot(fa_values, pd_values, 'b-', linewidth=2, label='PD-FA曲线')
    
    # 绘制阈值线
    if fa_threshold is not None:
        plt.axvline(x=fa_threshold, color='r', linestyle='--', label=f'FA阈值 = {fa_threshold}')
    
    if pd_threshold is not None:
        plt.axhline(y=pd_threshold, color='g', linestyle='--', label=f'PD阈值 = {pd_threshold}')
    
    plt.xlim([0, min(max(fa_values), 1)])
    plt.ylim([0, 1.05])
    plt.xlabel('虚警率 (FA)')
    plt.ylabel('检测概率 (PD)')
    plt.title('ISNet PD-FA曲线')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"PD-FA曲线已保存到: {save_path}")
    
    plt.show()


def main():
    """
    在测试集上评估ISNet模型的主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估ISNet模型')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--dataset', type=str, default=None,
                        help='指定数据集名称，如果不指定则使用配置文件中的设置')
    parser.add_argument('--split', type=str, default='test',
                        help='数据集分割，默认为test')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果保存目录')
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
    
    # 设置数据集名称
    dataset_name = args.dataset if args.dataset else config['dataset']['name']
    
    # 创建测试数据集
    print(f"创建{args.split}数据集...")
    test_dataset = SIRSTDataset(
        root_dir=config['dataset']['root_dir'],
        dataset_name=dataset_name,
        split=args.split,
        augment=False,
        crop_size=None  # 不对测试集进行裁剪
    )
    test_loader = test_dataset.set_attrs(
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
    
    # 加载检查点
    print(f"加载模型检查点: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)
    
    # 创建指标计算器
    iou_metric = IoUMetric()
    pd_fa_metric = PD_FA()
    roc_metric = ROCMetric()
    
    # 设置为评估模式
    model.eval()
    
    # 评估模型
    print("开始评估模型...")
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 评估过程
    with jt.no_grad():
        for batch in test_loader:
            # 解析批次数据
            images, masks, edges = batch['image'], batch['mask'], batch['edge']
            
            # 前向传播
            predictions, edge_outputs = model(images, edges)
            
            # 应用sigmoid激活
            predictions = jt.sigmoid(predictions)
            
            # 更新指标
            iou_metric.update(predictions, masks)
            pd_fa_metric.update(predictions, masks)
            roc_metric.update(predictions, masks)
    
    # 计算IoU指标
    iou_result = iou_metric.compute()
    print(f"IoU: {iou_result:.4f}")
    
    # 计算PD-FA指标
    pd_fa_result = pd_fa_metric.compute()
    print(f"PD@FA=10^-6: {pd_fa_result['pd']:.4f}")
    print(f"FA@PD=0.75: {pd_fa_result['fa']:.8f}")
    
    # 计算ROC指标
    roc_result = roc_metric.compute()
    print(f"AUC: {roc_result['auc']:.4f}")
    
    # 绘制ROC曲线
    if args.output_dir:
        roc_path = os.path.join(args.output_dir, 'roc_curve.png')
        plot_roc_curve(roc_result['fpr'], roc_result['tpr'], roc_result['auc'], save_path=roc_path)
        
        # 绘制PD-FA曲线
        pd_fa_path = os.path.join(args.output_dir, 'pd_fa_curve.png')
        plot_pd_fa(pd_fa_result['fa_values'], pd_fa_result['pd_values'], 
                  fa_threshold=1e-6, pd_threshold=0.75, save_path=pd_fa_path)
    else:
        plot_roc_curve(roc_result['fpr'], roc_result['tpr'], roc_result['auc'])
        plot_pd_fa(pd_fa_result['fa_values'], pd_fa_result['pd_values'], 
                  fa_threshold=1e-6, pd_threshold=0.75)
    
    print("评估完成")


if __name__ == "__main__":
    main()
