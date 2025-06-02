import random
import numpy as np
import cv2
import jittor as jt

class Compose:
    """组合多个变换"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class Resize:
    """调整图像和掩码大小"""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return image, mask

class RandomCrop:
    """随机裁剪图像和掩码"""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        h, w = image.shape[:2]
        new_h, new_w = self.size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        image = image[top:top + new_h, left:left + new_w]
        if mask is not None:
            mask = mask[top:top + new_h, left:left + new_w]

        return image, mask

class RandomHorizontalFlip:
    """随机水平翻转"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask=None):
        if random.random() < self.prob:
            image = np.fliplr(image).copy()
            if mask is not None:
                mask = np.fliplr(mask).copy()
        return image, mask

class RandomVerticalFlip:
    """随机垂直翻转"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask=None):
        if random.random() < self.prob:
            image = np.flipud(image).copy()
            if mask is not None:
                mask = np.flipud(mask).copy()
        return image, mask

class RandomRotation:
    """随机旋转"""
    def __init__(self, degrees=10, prob=0.5):
        self.degrees = degrees
        self.prob = prob

    def __call__(self, image, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.degrees, self.degrees)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        return image, mask

class ColorJitter:
    """随机调整亮度、对比度、饱和度和色调"""
    def __init__(self, brightness=0.2, contrast=0.2, prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.prob = prob

    def __call__(self, image, mask=None):
        if random.random() < self.prob:
            # 亮度调整
            alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
            # 对比度调整
            beta = random.uniform(-self.contrast * 127, self.contrast * 127)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return image, mask

class Normalize:
    """归一化图像"""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, mask=None):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        if mask is not None:
            mask = mask.astype(np.float32) / 255.0
        return image, mask

class ToTensor:
    """将NumPy数组转换为Jittor张量"""
    def __call__(self, image, mask=None):
        # 调整通道顺序：HWC -> CHW
        image = image.transpose(2, 0, 1)
        image = jt.array(image)
        
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, ...]  # 添加通道维度
            else:
                mask = mask.transpose(2, 0, 1)
            mask = jt.array(mask)
            
        return image, mask
