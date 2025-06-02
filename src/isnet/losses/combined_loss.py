import jittor as jt
import jittor.nn as nn
from .dice_loss import DiceLoss, BCEDiceLoss
from .edge_loss import EdgeLoss

class CombinedLoss(nn.Module):
    """
    ISNet的组合损失函数，结合主分割损失和边缘损失。
    主分割损失使用BCEDice损失，边缘损失使用BCE损失。
    """
    def __init__(self, main_weight=1.0, edge_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.main_loss = BCEDiceLoss(bce_weight=0.5)
        self.edge_loss = EdgeLoss()
        self.main_weight = main_weight
        self.edge_weight = edge_weight
        
    def execute(self, main_pred, main_target, edge_pred, edge_target):
        """
        Args:
            main_pred: 主分割预测，形状为[B, 1, H, W]，值在[0,1]范围内或未经过sigmoid激活
            main_target: 主分割目标，形状为[B, 1, H, W]，值在[0,1]范围内
            edge_pred: 边缘预测，形状为[B, 1, H, W]，值在[0,1]范围内
            edge_target: 边缘目标，形状为[B, 1, H, W]，值在[0,1]范围内
        Returns:
            总损失值，主分割损失值，边缘损失值
        """
        # 计算主分割损失
        main_loss_val = self.main_loss(main_pred, main_target)
        
        # 计算边缘损失
        edge_loss_val = self.edge_loss(edge_pred, edge_target)
        
        # 计算总损失
        total_loss = self.main_weight * main_loss_val + self.edge_weight * edge_loss_val
        
        return total_loss, main_loss_val, edge_loss_val
