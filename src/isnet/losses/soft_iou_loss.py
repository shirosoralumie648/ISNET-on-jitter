import jittor as jt
import jittor.nn as nn
import numpy as np

class SoftIoULoss(nn.Module):
    def __init__(self, batch=32):
        super(SoftIoULoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def execute(self, pred, target):
        pred = jt.sigmoid(pred)
        smooth = 0.00

        intersection = pred * target

        intersection_sum = jt.sum(intersection, dims=[1,2,3])
        pred_sum = jt.sum(pred, dims=[1,2,3])
        target_sum = jt.sum(target, dims=[1,2,3])
        
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - jt.mean(loss)
        loss1 = self.bce_loss(pred, target)
        return loss + loss1

class SoftIoULoss1(nn.Module):
    def __init__(self, batch=32):
        super(SoftIoULoss1, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def execute(self, pred, target):
        pred = jt.sigmoid(pred)
        smooth = 0.00

        intersection = pred * target

        intersection_sum = jt.sum(intersection, dims=[1,2,3])
        pred_sum = jt.sum(pred, dims=[1,2,3])
        target_sum = jt.sum(target, dims=[1,2,3])
        
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - jt.mean(loss)
        return loss