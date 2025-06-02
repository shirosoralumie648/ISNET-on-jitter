#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isnet.utils.jittor_utils import load_config, load_checkpoint, setup_jittor
from isnet.models.isnet_jittor import ISNet
from isnet.utils.video_processing import process_video, read_video_info


def compute(self):
    """
    计算并返回平均PD和平均FA (与get方法功能相同，提供兼容性)
    
    Returns:
        (平均PD, 平均FA)
    """
    return self.get()

def main():
    """
    u4f7fu7528ISNetu6a21u578bu9884u6d4bu89c6u9891u7684u4e3bu51fdu6570
    """
    # u89e3u6790u547du4ee4u884cu53c2u6570
    parser = argparse.ArgumentParser(description='u4f7fu7528ISNetu9884u6d4bu89c6u9891')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='u914du7f6eu6587u4ef6u8defu5f84')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='u6a21u578bu68c0u67e5u70b9u8defu5f84')
    parser.add_argument('--video', type=str, required=True,
                        help='u8f93u5165u89c6u9891u8defu5f84')
    parser.add_argument('--output', type=str, required=True,
                        help='u8f93u51fau89c6u9891u8defu5f84')
    parser.add_argument('--size', type=str, default=None,
                        help='u5904u7406u5c3au5bf8 (width,height)uff0cu4f8bu5982 "640,480"')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='u6279u5904u7406u5927u5c0fuff0cu9ed8u8ba41')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='u4f7fu7528GPUu8fdbu884cu63a8u7406')
    parser.add_argument('--color', type=str, default='0,255,0',
                        help='u53e0u52a0u989cu8272 (B,G,R)uff0cu9ed8u8ba4u4e3au7effu8272"0,255,0"')
    args = parser.parse_args()
    
    # u52a0u8f7du914du7f6eu6587u4ef6
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               args.config)
    config = load_config(config_path)
    
    # u66f4u65b0u914du7f6eu4e2du7684GPUu8bbeu7f6e
    if 'hardware' not in config:
        config['hardware'] = {}
    config['hardware']['use_gpu'] = args.gpu
    
    # u8bbeu7f6eJittoru73afu5883
    setup_jittor(config)
    
    # u89e3u6790u76eeu6807u5c3au5bf8
    target_size = None
    if args.size is not None:
        width, height = map(int, args.size.split(','))
        target_size = (width, height)
    
    # u89e3u6790u53e0u52a0u989cu8272
    overlay_color = tuple(map(int, args.color.split(',')))
    
    # u521bu5efau6a21u578b
    model = ISNet(
        layer_blocks=config['model']['layer_blocks'],
        channels=config['model']['channels']
    )
    
    # u52a0u8f7du68c0u67e5u70b9
    print(f"u52a0u8f7du6a21u578bu68c0u67e5u70b9: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)
    
    # u83b7u53d6u89c6u9891u4fe1u606f
    print(f"u5206u6790u89c6u9891: {args.video}")
    width, height, fps, total_frames = read_video_info(args.video)
    print(f"u89c6u9891u4fe1u606f: {width}x{height}, {fps:.2f}fps, {total_frames}u5e27")
    
    # u5904u7406u89c6u9891
    print("u5f00u59cbu5904u7406u89c6u9891...")
    process_video(
        model=model,
        video_path=args.video,
        output_path=args.output,
        target_size=target_size,
        batch_size=args.batch_size,
        show_progress=True,
        overlay_color=overlay_color
    )
    
    print(f"u89c6u9891u5904u7406u5b8cu6210uff0cu7ed3u679cu5df2u4fddu5b58u5230: {args.output}")


if __name__ == "__main__":
    main()
