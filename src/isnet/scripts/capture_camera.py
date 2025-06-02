#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isnet.utils.jittor_utils import load_config, load_checkpoint, setup_jittor
from isnet.models.isnet_jittor import ISNet
from isnet.utils.video_processing import capture_camera_feed


def main():
    """
    使用ISNet模型通过摄像头进行实时目标检测的主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用ISNet通过摄像头进行实时检测')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='摄像头ID，默认为0')
    parser.add_argument('--size', type=str, default='640,480',
                        help='处理尺寸 (width,height)，默认为640x480')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='使用GPU进行推理')
    parser.add_argument('--color', type=str, default='0,255,0',
                        help='叠加颜色 (B,G,R)，默认为绿色"0,255,0"')
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
    target_size = (640, 480)  # 默认尺寸
    if args.size is not None:
        width, height = map(int, args.size.split(','))
        target_size = (width, height)
    
    # 解析叠加颜色
    overlay_color = tuple(map(int, args.color.split(',')))
    
    # 创建模型
    model = ISNet(
        layer_blocks=config['model']['layer_blocks'],
        channels=config['model']['channels']
    )
    
    # 加载检查点
    print(f"加载模型检查点: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)
    
    # 启动摄像头检测
    print(f"启动摄像头ID {args.camera_id} 的实时检测...")
    print("按ESC键退出检测")
    
    capture_camera_feed(
        model=model,
        camera_id=args.camera_id,
        target_size=target_size,
        overlay_color=overlay_color
    )
    
    print("实时检测已结束")


if __name__ == "__main__":
    main()
