import jittor as jt
import numpy as np
import cv2
import os
import yaml
from jittor import nn

def set_random_seed(seed):
    """
    设置随机种子以确保实验可复现
    """
    jt.set_global_seed(seed)
    np.random.seed(seed)
    
def save_checkpoint(model, optimizer, epoch, save_path, is_best=False):
    """
    保存模型检查点
    
    Args:
        model: 要保存的模型
        optimizer: 优化器状态
        epoch: 当前训练轮次
        save_path: 保存路径
        is_best: 是否为当前最佳模型
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'epoch': epoch
    }
    
    jt.save(checkpoint, save_path)
    
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pkl')
        jt.save(checkpoint, best_path)

def load_checkpoint(model, checkpoint_path, optimizer=None):
    """
    加载模型检查点
    
    Args:
        model: 要加载权重的模型
        checkpoint_path: 检查点路径
        optimizer: 可选的优化器
        
    Returns:
        epoch: 模型训练的轮次
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = jt.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint.get('epoch', 0)

def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def sobel_edge_detection(img):
    """
    使用Sobel算子计算边缘图
    
    Args:
        img: 输入图像，形状为[H, W, C]或[H, W]，取值范围为[0,1]的浮点数
        
    Returns:
        边缘图，形状为[H, W, 3]，取值范围为[0,1]的浮点数
    """
    if len(img.shape) == 3:
        # 转换为灰度图
        if img.shape[2] == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img[:, :, 0] * 255).astype(np.uint8)
    else:
        gray = (img * 255).astype(np.uint8)
    
    # 计算Sobel边缘
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 组合x和y方向梯度
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # 归一化到[0,1]范围
    magnitude = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude
    
    # 转换为三通道图像
    edge_map = np.stack([magnitude, magnitude, magnitude], axis=2)
    
    return edge_map

def setup_jittor(config):
    """
    根据配置设置Jittor的运行环境
    
    Args:
        config: 包含硬件配置的字典
    """
    if config.get('hardware', {}).get('use_gpu', True):
        # 使用GPU
        jt.flags.use_cuda = 1
    else:
        # 使用CPU
        jt.flags.use_cuda = 0
    
    # 设置随机种子以确保实验可复现
    seed = config.get('hardware', {}).get('seed', 42)
    set_random_seed(seed)

def create_optimizer(config, model_parameters):
    """
    根据配置创建优化器
    
    Args:
        config: 包含优化器配置的字典
        model_parameters: 模型参数
        
    Returns:
        optimizer: Jittor优化器
    """
    optimizer_name = config['train']['optimizer'].lower()
    lr = config['train']['learning_rate']
    weight_decay = config['train']['weight_decay']
    
    if optimizer_name == 'adam':
        return nn.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return nn.SGD(model_parameters, lr=lr, momentum=config['train']['momentum'], 
                      weight_decay=weight_decay)
    elif optimizer_name == 'adagrad':
        # Jittor不支持Adagrad，使用Adam替代
        print("警告: Jittor不支持Adagrad优化器，将使用Adam替代")
        return nn.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

def create_lr_scheduler(config, optimizer):
    """
    根据配置创建学习率调度器
    
    Args:
        config: 包含学习率调度器配置的字典
        optimizer: Jittor优化器
        
    Returns:
        scheduler: 学习率调度器对象或函数
    """
    scheduler_name = config['train']['lr_scheduler'].lower()
    
    if scheduler_name == 'step':
        step_size = config['train']['step_size']
        gamma = config['train']['gamma']
        
        def step_scheduler(optimizer, epoch):
            if epoch > 0 and epoch % step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= gamma
        
        return step_scheduler
    
    elif scheduler_name == 'poly':
        max_epochs = config['train']['epochs']
        power = config['train']['lr_power']
        
        def poly_scheduler(optimizer, epoch):
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['train']['learning_rate'] * \
                                  (1 - epoch / max_epochs) ** power
        
        return poly_scheduler
    
    elif scheduler_name == 'cosine':
        max_epochs = config['train']['epochs']
        
        def cosine_scheduler(optimizer, epoch):
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['train']['learning_rate'] * \
                                  0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
        
        return cosine_scheduler
    
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")
