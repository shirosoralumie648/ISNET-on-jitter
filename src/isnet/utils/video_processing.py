import cv2
import numpy as np
import jittor as jt
import os
from tqdm import tqdm
from .visualization import make_overlay_image

def read_video_info(video_path):
    """
    读取视频基本信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        width: 视频宽度
        height: 视频高度
        fps: 帧率
        total_frames: 总帧数
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return width, height, fps, total_frames

def create_video_writer(output_path, width, height, fps):
    """
    创建视频写入器
    
    Args:
        output_path: 输出视频路径
        width: 视频宽度
        height: 视频高度
        fps: 帧率
        
    Returns:
        video_writer: OpenCV视频写入器对象
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return video_writer

def preprocess_frame(frame, target_size=None, normalize=True):
    """
    预处理视频帧以供模型输入
    
    Args:
        frame: 原始帧图像，BGR格式
        target_size: (width, height) 调整大小，如果为None则保持原始大小
        normalize: 是否归一化到[0,1]范围
        
    Returns:
        处理后的帧图像，Jittor张量格式[1, 3, H, W]
        original_frame: 原始帧图像，可能经过调整大小
    """
    # 保存原始帧以供可视化使用
    original_frame = frame.copy()
    
    # 调整大小如果需要
    if target_size is not None:
        frame = cv2.resize(frame, target_size)
        original_frame = cv2.resize(original_frame, target_size)
    
    # 转换为RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 归一化
    if normalize:
        frame = frame.astype(np.float32) / 255.0
    
    # 转换为Jittor张量格式[1, 3, H, W]
    frame = frame.transpose(2, 0, 1)  # [3, H, W]
    frame = np.expand_dims(frame, axis=0)  # [1, 3, H, W]
    frame = jt.array(frame)
    
    return frame, original_frame

def process_video(model, video_path, output_path, target_size=None, batch_size=1, show_progress=True, overlay_color=(0, 255, 0)):
    """
    处理视频并生成分割结果
    
    Args:
        model: 预加载的Jittor模型
        video_path: 输入视频路径
        output_path: 输出视频路径
        target_size: 可选的处理尺寸(width, height)
        batch_size: 批处理大小
        show_progress: 是否显示进度条
        overlay_color: 叠加颜色(BGR格式)
    """
    # 读取视频信息
    width, height, fps, total_frames = read_video_info(video_path)
    
    # 如果没有指定目标尺寸，使用原始视频尺寸
    if target_size is None:
        target_size = (width, height)
    
    # 创建视频写入器
    video_writer = create_video_writer(output_path, width, height, fps)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    
    # 创建进度条
    pbar = tqdm(total=total_frames) if show_progress else None
    
    # 设置模型为评估模式
    model.eval()
    
    frames_buffer = []
    original_frames_buffer = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理帧
        processed_frame, original_frame = preprocess_frame(frame, target_size)
        
        frames_buffer.append(processed_frame)
        original_frames_buffer.append(original_frame)
        frame_count += 1
        
        # 当缓冲区达到batch_size或最后一批时执行处理
        if len(frames_buffer) == batch_size or frame_count == total_frames:
            # 合并批处理
            batch_frames = jt.concat(frames_buffer, dim=0)
            
            # 推理
            with jt.no_grad():
                # 使用边缘图作为输入
                edge_maps = [jt.zeros((1, 3, f.shape[2], f.shape[3])) for f in frames_buffer]
                edge_batch = jt.concat(edge_maps, dim=0)
                
                # 模型预测
                predictions, edge_outputs = model(batch_frames, edge_batch)
                
                # 将预测结果转换为水平标签
                predictions = jt.sigmoid(predictions)
            
            # 处理每一帧的预测结果
            for i in range(len(frames_buffer)):
                # 获取当前帧的预测结果
                pred = predictions[i].numpy()  # [1, H, W]
                
                # 转换为适合显示的格式并二值化
                pred = pred[0]  # [H, W]
                pred_binary = (pred > 0.5).astype(np.uint8) * 255
                
                # 将预测结果调整到原始视频尺寸
                if pred_binary.shape[:2] != (height, width):
                    pred_binary = cv2.resize(pred_binary, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # 将处理后的原始帧调整到原始视频尺寸
                orig_frame = original_frames_buffer[i]
                if orig_frame.shape[:2] != (height, width):
                    orig_frame = cv2.resize(orig_frame, (width, height))
                
                # 叠加分割结果到原始帧
                overlay = make_overlay_image(orig_frame, pred_binary, color=overlay_color)
                
                # 写入输出视频
                video_writer.write(overlay)
                
                # 更新进度条
                if show_progress:
                    pbar.update(1)
            
            # 清空缓冲区
            frames_buffer = []
            original_frames_buffer = []
    
    # 关闭资源
    cap.release()
    video_writer.release()
    if show_progress:
        pbar.close()

def capture_camera_feed(model, camera_id=0, target_size=(640, 480), overlay_color=(0, 255, 0)):
    """
    从摄像头实时捕捉并处理视频流
    
    Args:
        model: 预加载的Jittor模型
        camera_id: 摄像头ID
        target_size: 处理尺寸(width, height)
        overlay_color: 叠加颜色(BGR格式)
    """
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise IOError(f"无法打开摄像头 ID: {camera_id}")
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])
    
    # 设置模型为评估模式
    model.eval()
    
    print("实时监测启动，按ESC键退出")
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break
        
        # 预处理帧
        processed_frame, original_frame = preprocess_frame(frame, target_size)
        
        # 使用边缘图作为输入
        edge_map = jt.zeros((1, 3, processed_frame.shape[2], processed_frame.shape[3]))
        
        # 推理
        with jt.no_grad():
            prediction, edge_output = model(processed_frame, edge_map)
            prediction = jt.sigmoid(prediction)
        
        # 处理预测结果
        pred = prediction[0].numpy()  # [1, H, W]
        pred = pred[0]  # [H, W]
        pred_binary = (pred > 0.5).astype(np.uint8) * 255
        
        # 叠加结果到原始帧
        overlay = make_overlay_image(original_frame, pred_binary, color=overlay_color)
        
        # 显示结果
        cv2.imshow("ISNet实时检测", overlay)
        
        # 检测键ESC退出
        if cv2.waitKey(1) == 27:  # ESC键
            break
    
    # 关闭资源
    cap.release()
    cv2.destroyAllWindows()
