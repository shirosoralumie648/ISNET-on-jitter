import jittor as jt
import jittor.nn as nn

class EdgeLoss(nn.Module):
    """u8fb9u7f18u635fu5931u51fdu6570uff0cu7528u4e8eu76d1u7763u8fb9u7f18u68c0u6d4bu5206u652fu7684u8badu7ec3u3002
    u4f7fu7528BCEu635fu5931u8ba1u7b97u8fb9u7f18u9884u6d4bu4e0eu76eeu6807u8fb9u7f18u7684u5deeu5f02u3002
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.bce = nn.BCELoss()
        
    def execute(self, pred_edge, target_edge):
        """
        Args:
            pred_edge: u9884u6d4bu7684u8fb9u7f18u56feuff0cu5f62u72b6u4e3a[B, 1, H, W]uff0cu503cu5728[0,1]u8303u56f4u5185
            target_edge: u76eeu6807u8fb9u7f18u56feuff0cu5f62u72b6u4e3a[B, 1, H, W]uff0cu503cu5728[0,1]u8303u56f4u5185
        Returns:
            u635fu5931u503c
        """
        return self.bce(pred_edge, target_edge)


class WeightedBCELoss(nn.Module):
    """u52a0u6743u7684BCEu635fu5931u51fdu6570uff0cu53efu4ee5u4e3au6b63u8d1fu6837u672cu8bbeu7f6eu4e0du540cu7684u6743u91cdu3002
    u9002u7528u4e8eu89e3u51b3u8fb9u7f18u68c0u6d4bu4e2du6b63u8d1fu6837u672cu4e0du5e73u8861u7684u95eeu9898u3002
    """
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
    def execute(self, pred, target):
        """
        Args:
            pred: u9884u6d4bu7ed3u679cuff0cu5f62u72b6u4e3a[B, 1, H, W]uff0cu503cu5728[0,1]u8303u56f4u5185
            target: u76eeu6807u63a9u7801uff0cu5f62u72b6u4e3a[B, 1, H, W]uff0cu503cu5728[0,1]u8303u56f4u5185
        Returns:
            u635fu5931u503c
        """
        # u8ba1u7b97u6807u51c6BCEu635fu5931
        eps = 1e-7
        pred = jt.clamp(pred, eps, 1-eps)  # u907fu514dlog(0)
        
        # u5206u522bu8ba1u7b97u6b63u8d1fu6837u672cu7684u635fu5931
        pos_loss = -target * jt.log(pred)
        neg_loss = -(1 - target) * jt.log(1 - pred)
        
        # u52a0u6743u5e73u5747
        loss = self.pos_weight * pos_loss + self.neg_weight * neg_loss
        
        return loss.mean()
