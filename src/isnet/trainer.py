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
from .utils.gradient_utils import GetGradientNopadding # Import GetGradientNopadding

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

        # 初始化梯度计算模块 (如果使用CombinedLoss且需要边缘监督)
        if self.config['loss']['name'].lower() == 'combined':
            self.get_gradient = GetGradientNopadding()
    
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
        epoch_loss = 0.0
        
        for metric_calculator in self.metrics.values():
            metric_calculator.reset()
            
        pbar = tqdm(total=len(self.val_loader), desc=f"验证轮次 {epoch+1}/{self.config['train']['epochs']}")
        
        with jt.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                images = batch['image']
                main_targets = batch['mask']
                model_edge_input = batch.get('edge_map', images)

                main_pred, edge_pred = self.model(images, model_edge_input)

                loss_val_info = ""
                if self.config['loss']['name'].lower() == 'combined':
                    edge_targets = self.get_gradient(main_targets)
                    total_loss, main_loss_val, edge_loss_val = self.criterion(main_pred, main_targets, edge_pred, edge_targets)
                    loss = total_loss
                    loss_val_info = f"Total: {loss.item():.4f}, Main: {main_loss_val.item():.4f}, Edge: {edge_loss_val.item():.4f}"
                else:
                    loss = self.criterion(main_pred, main_targets)
                    loss_val_info = f"{loss.item():.4f}"
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss_val_info)

                main_pred_sigmoid = jt.sigmoid(main_pred)
                for metric_calculator in self.metrics.values():
                    metric_calculator.update(main_pred_sigmoid, main_targets)
                
                pbar.update(1)
        
        pbar.close()
        avg_loss = epoch_loss / len(self.val_loader)
        print(f"轮次 {epoch+1} 验证损失: {avg_loss:.4f}")

        metrics_summary = {}
        for name, metric_calculator in self.metrics.items():
            metric_value = metric_calculator.get()
            if isinstance(metric_value, dict):
                 for k, v in metric_value.items(): metrics_summary[f"{name}_{k}"] = v
            else: metrics_summary[name] = metric_value
            print(f"  {name}: {metric_value}")
        return metrics_summary

    def _visualize_predictions(self, epoch):
        """
        可视化一些验证样本的预测结果
        
        Args:
            epoch: 当前训练轮次
        """
        self.model.eval()
        num_samples_to_viz = self.config.get('visualization', {}).get('num_samples', 4)
        
        viz_data = []
        with jt.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i * batch['image'].shape[0] >= num_samples_to_viz:
                    break
                images = batch['image']
                main_targets = batch['mask']
                model_edge_input = batch.get('edge_map', images)
                
                main_pred, edge_pred = self.model(images, model_edge_input)
                main_pred_sigmoid = jt.sigmoid(main_pred)
                edge_pred_sigmoid = jt.sigmoid(edge_pred) # Assuming edge_pred also needs sigmoid for viz
                
                for j in range(images.shape[0]):
                    if len(viz_data) < num_samples_to_viz:
                        viz_data.append({
                            'image': images[j].numpy(),
                            'mask': main_targets[j].numpy(),
                            'prediction': main_pred_sigmoid[j].numpy(),
                            'edge_gt': self.get_gradient(main_targets[j:j+1])[0].numpy() if hasattr(self, 'get_gradient') else None,
                            'edge_pred': edge_pred_sigmoid[j].numpy()
                        })
                    else:
                        break
        
        if not viz_data:
            print("没有样本可供可视化。")
            return

        fig = create_comparison_grid(viz_data, self.config.get('visualization', {}))
        save_path = os.path.join(self.visualization_dir, f"epoch_{epoch+1}.png")
        save_visualization(fig, save_path)
        print(f"可视化结果已保存到: {save_path}")

    def evaluate(self, test_loader):
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            包含各项评估指标的字典
        """
        self.model.eval()
        epoch_loss = 0.0 # For consistency, can also track loss during evaluation

        for metric_calculator in self.metrics.values():
            metric_calculator.reset()
            
        pbar = tqdm(total=len(test_loader), desc="评估中")
        
        with jt.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                images = batch['image']
                main_targets = batch['mask']
                model_edge_input = batch.get('edge_map', images)

                main_pred, edge_pred = self.model(images, model_edge_input)

                loss_val_info = ""
                if self.config['loss']['name'].lower() == 'combined' and hasattr(self, 'get_gradient'):
                    edge_targets = self.get_gradient(main_targets)
                    total_loss, main_loss_val, edge_loss_val = self.criterion(main_pred, main_targets, edge_pred, edge_targets)
                    loss = total_loss
                    loss_val_info = f"Total: {loss.item():.4f}, Main: {main_loss_val.item():.4f}, Edge: {edge_loss_val.item():.4f}"
                elif self.config['loss']['name'].lower() != 'combined': # Handle non-combined losses
                    loss = self.criterion(main_pred, main_targets)
                    loss_val_info = f"{loss.item():.4f}"
                else: # Combined loss but no get_gradient (should not happen if configured correctly)
                    loss = self.criterion(main_pred, main_targets) # Fallback or specific handling
                    loss_val_info = f"{loss.item():.4f}"

                if isinstance(loss, jt.Var): # Ensure loss is a Jittor Var before .item()
                    epoch_loss += loss.item()
                pbar.set_postfix(loss=loss_val_info)

                main_pred_sigmoid = jt.sigmoid(main_pred)
                for metric_calculator in self.metrics.values():
                    metric_calculator.update(main_pred_sigmoid, main_targets)
                
                pbar.update(1)
        
        pbar.close()
        avg_loss = epoch_loss / len(test_loader) if len(test_loader) > 0 else 0
        print(f"评估平均损失: {avg_loss:.4f}")

        metrics_summary = {}
        print("评估结果:")
        for name, metric_calculator in self.metrics.items():
            metric_value = metric_calculator.get() # Use .get() for metrics
            if isinstance(metric_value, dict):
                 for k, v in metric_value.items(): metrics_summary[f"{name}_{k}"] = v
            else: metrics_summary[name] = metric_value
            print(f"  {name}: {metric_value}") # Or format nicely
        return metrics_summary

    def plot_training_progress(self, save_path=None):
        """
        绘制训练进度图表
        
        Args:
            save_path: 可选的保存路径，如果提供则保存图表
        """
        if not self.train_losses:
            print("无训练历史可绘制")
            return
        
        plt.figure(figsize=(12, 6))
        
        # 绘制训练损失
        epochs_axis = range(1, len(self.train_losses) + 1)
        plt.plot(epochs_axis, self.train_losses, 'r-', marker='o', label='训练损失')
        
        # 如果有验证指标，绘制IoU
        if self.val_metrics:
            val_epochs = []
            val_ious = []
            val_interval = self.config['train'].get('val_interval', 1)
            for i, metrics_dict in enumerate(self.val_metrics):
                # Attempt to find IoU, supporting different naming conventions from metrics
                iou_val = metrics_dict.get('iou', metrics_dict.get('IoUMetric', metrics_dict.get('iou_IoUMetric', None)))
                if iou_val is not None:
                    val_epochs.append((i + 1) * val_interval)
                    val_ious.append(iou_val)
            
            if val_ious: # Only plot if IoU values were found
                ax2 = plt.gca().twinx()
                ax2.plot(val_epochs, val_ious, 'b-', marker='s', label='验证 IoU')
                ax2.set_ylabel('验证 IoU', color='b')
                ax2.tick_params(axis='y', labelcolor='b')

        plt.xlabel('轮次')
        plt.gca().set_ylabel('训练损失', color='r')
        plt.gca().tick_params(axis='y', labelcolor='r')
        plt.title('训练进度')
        plt.grid(True)
        
        # 合并图例
        lines, labels = plt.gca().get_legend_handles_labels()
        if 'ax2' in locals() and ax2 is not None:
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, loc='best')
        else:
            plt.legend(loc='best')
            
        if save_path:
            plt.savefig(save_path)
            print(f"训练进度图表已保存到: {save_path}")
        else:
            plt.show()
