import jittor as jt
import numpy as np

class IoUMetric:
    """
    IoU评估指标，用于计算分割模型的交并比。
    支持计算批量数据的平均IoU。
    """
    def __init__(self, threshold=0.5):
        """
        Args:
            threshold: 将预测概率二值化的阈值
        """
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        """
        重置累积指标
        """
        self.total_iou = 0.0
        self.num_samples = 0
        
    def update(self, pred, target):
        """
        更新指标计算
        
        Args:
            pred: 预测结果, 形状为[B, 1, H, W], 取值范围[0,1]的浮点值
            target: 目标掩码, 形状为[B, 1, H, W], 取值范围[0,1]的浮点值
        """
        # 将预测结果二值化
        pred = (pred > self.threshold).float()
        
        # 计算交集和并集
        intersection = (pred * target).sum((1, 2, 3))
        union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) - intersection
        
        # 避免除零
        union = jt.maximum(union, jt.ones_like(union))
        
        # 计算每个样本的IoU
        iou = intersection / union
        
        # 累积
        self.total_iou += iou.sum().item()
        self.num_samples += pred.shape[0]
        
    def get(self):
        """
        返回平均IoU
        
        Returns:
            平均IoU值
        """
        if self.num_samples == 0:
            return 0.0
        return self.total_iou / self.num_samples
        
    def compute(self):
        """
        计算并返回平均IoU (与get方法功能相同，提供兼容性)
        
        Returns:
            平均IoU值
        """
        return self.get()


class PD_FA:
    """
    概率检测(PD)和虚警率(FA)度量，用于红外小目标检测评估。
    PD：正确检测的目标像素占总目标像素的比例。
    FA：每帧图像中误检像素的数量。
    """
    def __init__(self, threshold=0.5):
        """
        Args:
            threshold: 将预测概率二值化的阈值
        """
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        """
        重置累积指标
        """
        self.total_pd = 0.0
        self.total_fa = 0.0
        self.num_samples = 0
        
    def update(self, pred, target):
        """
        更新指标计算
        
        Args:
            pred: 预测结果, 形状为[B, 1, H, W], 取值范围[0,1]的浮点值
            target: 目标掩码, 形状为[B, 1, H, W], 取值范围[0,1]的浮点值
        """
        # 将预测结果二值化
        pred_bin = (pred > self.threshold).float()
        
        # 计算正确检测的目标像素（真正例TP）
        tp = (pred_bin * target).sum((1, 2, 3))
        
        # 计算总目标像素
        total_target = target.sum((1, 2, 3))
        
        # 计算虚警像素（假正例FP）
        fp = (pred_bin * (1 - target)).sum((1, 2, 3))
        
        # 计算每个样本的PD和FA
        pd = tp / jt.maximum(total_target, jt.ones_like(total_target))  # 避免除零
        fa = fp  # 每帧图像中的虚警像素数
        
        # 累积
        self.total_pd += pd.sum().item()
        self.total_fa += fa.sum().item()
        self.num_samples += pred.shape[0]
        
    def get(self):
        """
        返回平均PD和平均FA
        
        Returns:
            (平均PD, 平均FA)
        """
        if self.num_samples == 0:
            return 0.0, 0.0
        return self.total_pd / self.num_samples, self.total_fa / self.num_samples
        
    def compute(self):
        """
        计算并返回平均PD和平均FA (与get方法功能相同，提供兼容性)
        
        Returns:
            (平均PD, 平均FA)
        """
        return self.get()


class ROCMetric:
    """
    ROC曲线相关的度量，用于计算不同阈值下的PD和FA值对。
    可用于绘制ROC曲线和计算AUC值。
    """
    def __init__(self, thresholds=None, max_fpr=0.3):
        """
        Args:
            thresholds: 用于计算ROC点的阈值列表
            max_fpr: 计算部分AUC时使用的最大虚警率
        """
        if thresholds is None:
            # 默认使用10个阈值点
            self.thresholds = np.linspace(0, 1, 11)
        else:
            self.thresholds = thresholds
        self.max_fpr = max_fpr
        self.reset()
        
    def reset(self):
        """
        重置累积指标
        """
        self.pred_list = []
        self.target_list = []
        
    def update(self, pred, target):
        """
        更新指标计算
        
        Args:
            pred: 预测结果, 形状为[B, 1, H, W], 取值范围[0,1]的浮点值
            target: 目标掩码, 形状为[B, 1, H, W], 取值范围[0,1]的浮点值
        """
        # 转换为numpy以便后续处理
        self.pred_list.append(pred.numpy())
        self.target_list.append(target.numpy())
        
    def get(self):
        """
        计算ROC曲线上的点和AUC值
        
        Returns:
            (pd_list, fa_list, auc)
        """
        if not self.pred_list:
            return [], [], 0.0
        
        # 合并所有批次的预测和目标
        all_pred = np.concatenate(self.pred_list, axis=0)
        all_target = np.concatenate(self.target_list, axis=0)
        
        # 计算不同阈值下的PD和FA
        pd_list = []
        fa_list = []
        
        for thresh in self.thresholds:
            pred_bin = (all_pred > thresh).astype(np.float32)
            
            # 计算TP, FP, 总目标像素
            tp = (pred_bin * all_target).sum()
            fp = (pred_bin * (1 - all_target)).sum()
            total_target = all_target.sum()
            total_neg = (1 - all_target).sum()
            
            # 计算PD和FA
            pd = tp / max(total_target, 1)  # 避免除零
            fa = fp / max(total_neg, 1)  # 归一化FA为虚警率
            
            pd_list.append(pd)
            fa_list.append(fa)
        
        # 计算AUC - 使用梯形法则
        # 只考虑FA <= max_fpr的部分
        valid_idx = [i for i, fa in enumerate(fa_list) if fa <= self.max_fpr]
        
        if len(valid_idx) < 2:
            return pd_list, fa_list, 0.0
        
        # 提取有效部分并确保FA递增
        valid_pd = [pd_list[i] for i in valid_idx]
        valid_fa = [fa_list[i] for i in valid_idx]
        
        # 对点进行排序
        sorted_idx = np.argsort(valid_fa)
        sorted_pd = [valid_pd[i] for i in sorted_idx]
        sorted_fa = [valid_fa[i] for i in sorted_idx]
        
        # 添加原点
        if sorted_fa[0] > 0:
            sorted_pd.insert(0, 0.0)
            sorted_fa.insert(0, 0.0)
        
        # 计算AUC
        auc = 0.0
        for i in range(1, len(sorted_fa)):
            auc += (sorted_fa[i] - sorted_fa[i-1]) * (sorted_pd[i] + sorted_pd[i-1]) / 2
        
        # 归一化AUC为最大FA值
        auc = auc / self.max_fpr
        
        return pd_list, fa_list, auc
