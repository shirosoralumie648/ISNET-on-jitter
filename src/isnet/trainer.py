import jittor as jt
import os
import time
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

from .utils.jittor_utils import save_checkpoint, load_checkpoint, setup_jittor
from .utils.jittor_utils import create_optimizer, create_lr_scheduler
from .utils.visualization import visualize_prediction, create_comparison_grid, save_visualization

class Trainer:


    """
    ISNet训练器，负责处理训练和验证逻辑
    """
    def __init__(self, model, config, train_loader=None, val_loader=None):
        """
        初始化训练器
        
        Args:
            model: ISNet模型实例
            config: 配置字典
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 设置Jittor环境
        setup_jittor(config)
        
        # 创建优化器
        self.optimizer = create_optimizer(config, self.model.parameters())
        
        # 创建学习率调度器
        self.lr_scheduler = create_lr_scheduler(config, self.optimizer)
        
        # 创建损失函数
        self.criterion = self._create_loss_function()
        
        # 创建指标计算器
        self.metrics = self._create_metrics()
        
        # 训练状态变量
        self.start_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # 创建保存目录
        self.checkpoint_dir = os.path.join(
            config['checkpoint']['save_dir'], 
            f"{config['model']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 可视化目录
        self.visualization_dir = os.path.join(self.checkpoint_dir, 'visualizations')
        os.makedirs(self.visualization_dir, exist_ok=True)
    
    def _create_loss_function(self):
        """
        根据配置创建损失函数
        
        Returns:
            损失函数实例
        """
        loss_name = self.config['loss']['name'].lower()
        
        if loss_name == 'combined':
            from .losses.combined_loss import CombinedLoss
            return CombinedLoss(
                main_weight=self.config['loss']['main_weight'],
                edge_weight=self.config['loss']['edge_weight']
            )
        elif loss_name == 'bcedice':
            from .losses.dice_loss import BCEDiceLoss
            return BCEDiceLoss()
        elif loss_name == 'dice':
            from .losses.dice_loss import DiceLoss
            return DiceLoss()
        elif loss_name == 'edge':
            from .losses.edge_loss import EdgeLoss
            return EdgeLoss()
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")
    
    def _create_metrics(self):
        """
        创建评估指标计算器
        
        Returns:
            指标计算器字典
        """
        from .metrics.iou_metrics import IoUMetric, PD_FA, ROCMetric
        
        metrics = {
            'iou': IoUMetric(),
            'pd_fa': PD_FA()
        }
        
        if self.config.get('metrics', {}).get('use_roc', True):
            metrics['roc'] = ROCMetric()
        
        return metrics
    
    def train(self, resume_from=None):
        """
        训练模型
        
        Args:
            resume_from: 可选的恢复训练的检查点路径
        """
        # 如果提供了恢复路径，加载检查点
        if resume_from is not None:
            self.start_epoch = load_checkpoint(self.model, resume_from, self.optimizer)
            print(f"从检查点恢复训练，起始轮次: {self.start_epoch}")
        
        # 训练总轮次
        epochs = self.config['train']['epochs']
        
        # 主训练循环
        for epoch in range(self.start_epoch, epochs):
            # 训练一个轮次
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证 - 添加默认的val_interval值
            val_interval = self.config['train'].get('val_interval', 1)  # 默认每个轮次都验证
            if self.val_loader is not None and (epoch + 1) % val_interval == 0:
                val_metrics = self._validate_epoch(epoch)
                self.val_metrics.append(val_metrics)
                
                # 保存最佳模型
                current_metric = val_metrics['iou']
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    print(f"更新最佳模型，IoU: {self.best_metric:.4f}")
                
                # 保存检查点
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pkl")
                save_checkpoint(self.model, self.optimizer, epoch + 1, checkpoint_path, is_best)
                
                # 可视化一些验证样本
                if self.config.get('visualization', {}).get('enabled', True):
                    self._visualize_predictions(epoch)
            
            # 调整学习率
            if self.lr_scheduler is not None:
                self.lr_scheduler(self.optimizer, epoch + 1)
            
            # 输出当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.6f}")
        
        print(f"训练完成，最佳IoU: {self.best_metric:.4f}")
        return self.best_metric
    
    def _train_epoch(self, epoch):
        """
        训练一个完整的轮次
        
        Args:
            epoch: 当前训练轮次
            
        Returns:
            平均训练损失
        """
        self.model.train()
        epoch_loss = 0.0
        
        # 使用tqdm显示进度条
        pbar = tqdm(total=len(self.train_loader), desc=f"训练轮次 {epoch+1}/{self.config['train']['epochs']}")
        
        for i, batch in enumerate(self.train_loader):
            # 解析批次数据
            images, masks, edges = batch['image'], batch['mask'], batch['edge']
            
            # 前向传播
            predictions, edge_outputs = self.model(images, edges)
            
            # 计算损失
            loss_result = self.criterion(predictions, edge_outputs, masks, edges)
            
            # 处理不同类型的损失返回值
            if isinstance(loss_result, tuple):
                # 如果是元组，则使用第一个元素作为总损失
                total_loss = loss_result[0]
                loss_info = f"{total_loss.item():.4f}"
                if len(loss_result) > 2:
                    main_loss = loss_result[1]
                    edge_loss = loss_result[2]
                    loss_info = f"Total: {total_loss.item():.4f}, Main: {main_loss.item():.4f}, Edge: {edge_loss.item():.4f}"
            else:
                # 如果不是元组，则直接使用
                total_loss = loss_result
                loss_info = f"{total_loss.item():.4f}"
            
            # 反向传播和优化 - Jittor风格
            # 直接使用optimizer.step(loss)，它会自动完成zero_grad和backward
            self.optimizer.step(total_loss)
            
            # 累计损失
            epoch_loss += total_loss.item()
            
            # 更新进度条
            pbar.set_postfix({"loss": loss_info})
            pbar.update(1)
        
        pbar.close()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(self.train_loader)
        print(f"轮次 {epoch+1} 训练损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def _validate_epoch(self, epoch):
        """
        在验证集上验证模型
        
        Args:
            epoch: 当前训练轮次
            
        Returns:
            包含各项评估指标的字典
        """
        self.model.eval()
        
        # 重置所有指标
        for metric in self.metrics.values():
            metric.reset()
        
        # 使用tqdm显示进度条
        pbar = tqdm(total=len(self.val_loader), desc=f"验证轮次 {epoch+1}")
        
        val_loss = 0.0
        val_samples = 0
        
        with jt.no_grad():
            for i, batch in enumerate(self.val_loader):
                # 解析批次数据
                images, masks, edges = batch['image'], batch['mask'], batch['edge']
                val_samples += images.shape[0]
                
                # 前向传播
                predictions, edge_outputs = self.model(images, edges)
                
                # 计算验证集损失并显示详细信息
                loss_result = self.criterion(predictions, edge_outputs, masks, edges)
                
                # 处理不同类型的损失返回值
                if isinstance(loss_result, tuple):
                    # 如果是元组，则使用第一个元素作为总损失
                    total_loss = loss_result[0]
                    loss_info = f"{total_loss.item():.4f}"
                    if len(loss_result) > 2:
                        main_loss = loss_result[1]
                        edge_loss = loss_result[2]
                        loss_info = f"Total: {total_loss.item():.4f}, Main: {main_loss.item():.4f}, Edge: {edge_loss.item():.4f}"
                else:
                    # 如果不是元组，则直接使用
                    total_loss = loss_result
                    loss_info = f"{total_loss.item():.4f}"
                
                val_loss += total_loss.item() * images.shape[0]
                
                # 应用sigmoid激活用于评估指标
                predictions = jt.sigmoid(predictions)
                edge_outputs = jt.sigmoid(edge_outputs)
                
                # 更新指标
                for name, metric in self.metrics.items():
                    metric.update(predictions, masks)
                
                # 更新进度条
                pbar.set_postfix({"val_loss": loss_info})
                pbar.update(1)
        
        pbar.close()
        
        # 计算所有指标
        metrics_results = {}
        for name, metric in self.metrics.items():
            result = metric.get()
            if isinstance(result, dict):
                for k, v in result.items():
                    metrics_results[f"{name}_{k}"] = v
            else:
                metrics_results[name] = result
        
        # 打印指标
        print(f"轮次 {epoch+1} 验证结果:")
        for name, value in metrics_results.items():
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
        
        return metrics_results
    
    def _visualize_predictions(self, epoch):
        """
        可视化一些验证样本的预测结果
        
        Args:
            epoch: 当前训练轮次
        """
        self.model.eval()
        
        # 获取一些验证样本
        num_samples = min(4, len(self.val_loader))
        samples = []
        
        with jt.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                
                # 解析批次数据
                images, masks, edges = batch['image'], batch['mask'], batch['edge']
                
                # 前向传播
                predictions, edge_outputs = self.model(images, edges)
                
                # 应用sigmoid激活
                predictions = jt.sigmoid(predictions)
                edge_outputs = jt.sigmoid(edge_outputs)
                
                # 添加到样本列表
                for j in range(min(1, len(images))):
                    samples.append({
                        'image': images[j],
                        'mask': masks[j],
                        'prediction': predictions[j],
                        'edge': edge_outputs[j]
                    })
                
                if len(samples) >= num_samples:
                    break
        
        # 创建可视化网格
        images = [s['image'] for s in samples]
        masks = [s['mask'] for s in samples]
        predictions = [s['prediction'] for s in samples]
        edges = [s['edge'] for s in samples]
        
        fig = create_comparison_grid(images, masks, predictions, edges)
        
        # 保存可视化结果
        save_path = os.path.join(self.visualization_dir, f"epoch_{epoch+1}.png")
        save_visualization(fig, save_path)
    
    def predict(self, image, edge=None, return_edge=False):
        """
        对单张图像进行预测
        
        Args:
            image: 输入图像，形状为[1, C, H, W]的Jittor张量
            edge: 可选的边缘图，形状为[1, C, H, W]的Jittor张量
            return_edge: 是否返回边缘预测
            
        Returns:
            预测的分割掩码，如果return_edge为True，则同时返回边缘预测
        """
        self.model.eval()
        
        # 如果未提供边缘图，创建一个零张量
        if edge is None:
            edge = jt.zeros((1, 3, image.shape[2], image.shape[3]))
        
        with jt.no_grad():
            # 前向传播
            predictions, edge_outputs = self.model(image, edge)
            
            # 应用sigmoid激活
            predictions = jt.sigmoid(predictions)
            edge_outputs = jt.sigmoid(edge_outputs)
        
        if return_edge:
            return predictions, edge_outputs
        else:
            return predictions
    
    def evaluate(self, test_loader):
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            包含各项评估指标的字典
        """
        self.model.eval()
        
        # 重置所有指标
        for metric in self.metrics.values():
            metric.reset()
        
        # 使用tqdm显示进度条
        pbar = tqdm(total=len(test_loader), desc="测试中")
        
        with jt.no_grad():
            for i, batch in enumerate(test_loader):
                # 解析批次数据
                images, masks, edges = batch['image'], batch['mask'], batch['edge']
                
                # 前向传播
                predictions, edge_outputs = self.model(images, edges)
                
                # 应用sigmoid激活
                predictions = jt.sigmoid(predictions)
                edge_outputs = jt.sigmoid(edge_outputs)
                
                # 更新指标
                for name, metric in self.metrics.items():
                    metric.update(predictions, masks)
                
                # 更新进度条
                pbar.update(1)
        
        pbar.close()
        
        # 计算所有指标
        metrics_results = {}
        for name, metric in self.metrics.items():
            result = metric.compute()
            if isinstance(result, dict):
                for k, v in result.items():
                    metrics_results[f"{name}_{k}"] = v
            else:
                metrics_results[name] = result
        
        # 打印指标
        print("测试结果:")
        for name, value in metrics_results.items():
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
        
        return metrics_results
    
    def plot_training_progress(self, save_path=None):
        """
        绘制训练进度图表
        
        Args:
            save_path: 可选的保存路径，如果提供则保存图表
        """
        if not self.train_losses:
            print("无训练历史可绘制")
            return
        
        # 创建图表
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 绘制训练损失
        epochs = range(1, len(self.train_losses) + 1)
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('训练损失', color='tab:red')
        ax1.plot(epochs, self.train_losses, 'tab:red', marker='o', label='训练损失')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        # 如果有验证指标，绘制到第二个y轴
        if self.val_metrics:
            # 提取验证轮次和IoU值
            val_epochs = [self.config['train']['val_interval'] * (i + 1) for i in range(len(self.val_metrics))]
            val_ious = [metrics['iou'] if 'iou' in metrics else metrics.get('iou_iou', 0) 
                       for metrics in self.val_metrics]
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('验证IoU', color='tab:blue')
            ax2.plot(val_epochs, val_ious, 'tab:blue', marker='s', label='验证IoU')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        # 添加网格和图例
        ax1.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        if self.val_metrics:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        else:
            ax1.legend(loc='upper center')
        
        # 保存图表如果提供了路径
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"训练进度图表已保存到: {save_path}")
        
        plt.show()
