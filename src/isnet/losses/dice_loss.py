import jittor as jt
import jittor.nn as nn

class DiceLoss(nn.Module):
    """Dice损失函数，适用于分割任务。
    这是ISNet中使用的SoftLoULoss的Jittor实现版本。
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def execute(self, pred, target):
        """
        Args:
            pred: 预测结果，形状为[B, 1, H, W]，尚未经过sigmoid激活
            target: 目标掩码，形状为[B, 1, H, W]，值在[0,1]范围内
        Returns:
            损失值
        """
        # 将预测结果转换为概率值
        pred = jt.sigmoid(pred)
        
        # 计算交集
        intersection = (pred * target).sum(dims=(1, 2, 3))
        
        # 计算并集
        union = pred.sum(dims=(1, 2, 3)) + target.sum(dims=(1, 2, 3))
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 返回损失值（1 - Dice系数的平均值）
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """结合了BCE损失和Dice损失的复合损失函数"""
    def __init__(self, smooth=1.0, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.dice = DiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        
    def execute(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        
        # 组合损失
        loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        return loss
