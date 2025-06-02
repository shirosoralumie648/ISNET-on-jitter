import numpy as np
import cv2
import jittor as jt
import matplotlib.pyplot as plt

def visualize_prediction(image, mask, prediction, edge=None, alpha=0.5):
    """
    将分割预测结果可视化为彩色叠加图
    
    Args:
        image: 原始图像，形状为[C, H, W]或[H, W, C]，值范围[0,1]
        mask: 真实分割掩码，形状为[1, H, W]或[H, W]，值范围[0,1]
        prediction: 预测分割掩码，形状为[1, H, W]或[H, W]，值范围[0,1]
        edge: 可选的边缘预测，形状为[1, H, W]或[H, W]，值范围[0,1]
        alpha: 叠加透明度
        
    Returns:
        可视化结果图像，形状为[H, W, C]，值范围[0,1]
    """
    # 确保图像为HWC格式
    if isinstance(image, jt.Var):
        image = image.numpy()
    if isinstance(mask, jt.Var):
        mask = mask.numpy()
    if isinstance(prediction, jt.Var):
        prediction = prediction.numpy()
    if edge is not None and isinstance(edge, jt.Var):
        edge = edge.numpy()
    
    # 转换通道顺序如果需要
    if image.shape[0] == 3 and len(image.shape) == 3:  # CHW格式
        image = image.transpose(1, 2, 0)
    
    if len(mask.shape) == 3 and mask.shape[0] == 1:  # [1,H,W]格式
        mask = mask[0]
    
    if len(prediction.shape) == 3 and prediction.shape[0] == 1:  # [1,H,W]格式
        prediction = prediction[0]
    
    if edge is not None and len(edge.shape) == 3 and edge.shape[0] == 1:  # [1,H,W]格式
        edge = edge[0]
    
    # 创建RGB图像
    h, w = image.shape[:2]
    
    # 将图像值范围调整为[0,1]如果需要
    if image.max() > 1.0:
        image = image / 255.0
    
    # 为可视化创建彩色掩码
    # 红色为预测，绿色为真实值，黄色为重叠区域
    vis_img = image.copy()
    
    # 二值化预测和掩码
    pred_binary = (prediction > 0.5).astype(np.float32)
    mask_binary = (mask > 0.5).astype(np.float32)
    
    # 生成彩色掩码 - 红色为预测，绿色为真实值
    red_mask = np.zeros((h, w, 3), dtype=np.float32)
    red_mask[..., 0] = pred_binary  # 红色通道
    
    green_mask = np.zeros((h, w, 3), dtype=np.float32)
    green_mask[..., 1] = mask_binary  # 绿色通道
    
    # 融合
    vis_img = vis_img * (1 - alpha) + (red_mask + green_mask) * alpha
    
    # 确保值在[0,1]范围内
    vis_img = np.clip(vis_img, 0, 1)
    
    # 如果有边缘预测，添加到可视化中
    if edge is not None:
        # 用蓝色显示边缘
        edge_binary = (edge > 0.5).astype(np.float32)
        blue_mask = np.zeros((h, w, 3), dtype=np.float32)
        blue_mask[..., 2] = edge_binary  # 蓝色通道
        
        # 叠加边缘
        vis_img = vis_img * 0.7 + blue_mask * 0.3
        vis_img = np.clip(vis_img, 0, 1)
    
    return vis_img

def create_comparison_grid(images, masks, predictions, edges=None, alpha=0.5, figsize=(12, 12)):
    """
    创建网格显示多个图像和它们的分割结果
    
    Args:
        images: 图像列表，每个形状为[C, H, W]或[H, W, C]
        masks: 真实掩码列表，每个形状为[1, H, W]或[H, W]
        predictions: 预测掩码列表，每个形状为[1, H, W]或[H, W]
        edges: 可选的边缘预测列表
        alpha: 叠加透明度
        figsize: 图形大小
        
    Returns:
        网格图像
    """
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n):
        if edges is not None:
            vis_img = visualize_prediction(images[i], masks[i], predictions[i], edges[i], alpha)
        else:
            vis_img = visualize_prediction(images[i], masks[i], predictions[i], None, alpha)
        
        axes[i].imshow(vis_img)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    # 隐藏未使用的子图
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def save_visualization(fig, save_path):
    """
    保存可视化图像
    
    Args:
        fig: matplotlib图形对象
        save_path: 保存路径
    """
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def make_overlay_image(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    在图像上叠加单一颜色的掩码
    
    Args:
        image: 图像数组，形状为[H, W, C]，值范围[0,255]的uint8类型
        mask: 掩码数组，形状为[H, W]，值范围[0,1]的浮点型或[0,255]的uint8类型
        color: 叠加颜色，BGR格式
        alpha: 叠加透明度
        
    Returns:
        叠加后的图像，形状为[H, W, C]，值范围[0,255]的uint8类型
    """
    # 确保图像为uint8类型，值范围[0,255]
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 确保掩码为uint8类型，值范围[0,255]
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # 创建彩色掩码
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[:, :, i] = mask * color[i]
    
    # 叠加掩码到原图
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    return overlay
