import os
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
import cv2
import random
from ..utils.jittor_utils import sobel_edge_detection

class SirstDataset(Dataset):
    """SIRST dataset for infrared small target detection.
    
    This dataset loader supports both SIRST and IRSTD-1k datasets.
    """
    def __init__(self, root_dir, dataset_name='sirst', split='train', img_size=(256, 256), 
                 transform=None, use_edge=True, augmentations=True):
        """
        Args:
            root_dir (str): Root directory for the dataset
            dataset_name (str): 'sirst' or 'irstd1k'
            split (str): 'train' or 'test'
            img_size (tuple): Target image size (H, W)
            transform (callable, optional): Optional transform to be applied
            use_edge (bool): Whether to compute edge maps for edge supervision
            augmentations (bool): Whether to use data augmentations
        """
        super().__init__()
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.use_edge = use_edge
        self.augmentations = augmentations
        
        # Set dataset paths based on dataset_name
        if dataset_name == 'sirst':
            dataset_dir = os.path.join(root_dir, 'sirst-master')
            self.img_dir = os.path.join(dataset_dir, split, 'images')
            self.mask_dir = os.path.join(dataset_dir, split, 'masks')
        elif dataset_name == 'irstd1k':
            dataset_dir = os.path.join(root_dir, 'IRSTD-1k')
            self.img_dir = os.path.join(dataset_dir, split, 'images')
            self.mask_dir = os.path.join(dataset_dir, split, 'masks')
        else:
            raise ValueError(f"Dataset {dataset_name} not supported, use 'sirst' or 'irstd1k'")
        
        # Get file names
        self.img_names = sorted([f for f in os.listdir(self.img_dir) 
                               if f.endswith(('.jpg', '.png', '.bmp'))])
        
        self.total_len = len(self.img_names)
    
    def __len__(self):
        return self.total_len
    
    def _get_random_crop_params(self, img, mask):
        """Get random crop parameters"""
        h, w = img.shape[:2]
        th, tw = self.img_size
        
        if w == tw and h == th:
            return 0, 0, h, w
        
        # Ensure target has at least some nonzero pixels
        # Try several times to get a crop containing target
        for _ in range(10):
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            
            # Check if crop contains target
            mask_crop = mask[i:i+th, j:j+tw]
            if np.sum(mask_crop) > 0:
                return i, j, th, tw
        
        # If failed to find good crop, use center crop
        i = (h - th) // 2
        j = (w - tw) // 2
        return i, j, th, tw
    
    def _random_augmentation(self, img, mask):
        """Apply random augmentations"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        
        # Random vertical flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        # Random brightness and contrast adjustment
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-10, 10)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        return img, mask
    
    def __getitem__(self, idx):
        # Get image and mask paths
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # For SIRST, mask names might have different extensions
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            # Try other extensions
            for ext in ['.jpg', '.bmp']:
                alt_mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + ext)
                if os.path.exists(alt_mask_path):
                    mask_path = alt_mask_path
                    break
        
        # Read image and mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Read mask and ensure binary
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Apply augmentations in training mode
        if self.split == 'train' and self.augmentations:
            img, mask = self._random_augmentation(img, mask)
        
        # Random or center crop
        if self.split == 'train' and self.augmentations:
            i, j, h, w = self._get_random_crop_params(img, mask)
            img = img[i:i+h, j:j+w]
            mask = mask[i:i+h, j:j+w]
        else:
            # Resize to target size
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to [0,1] range
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Compute edge maps if needed
        if self.use_edge:
            # Use Sobel operator to get edges
            edge = sobel_edge_detection(img)
        else:
            edge = np.zeros_like(mask)
        
        # Convert to Jittor tensors
        img = jt.array(img.transpose(2, 0, 1))  # HWC to CHW
        mask = jt.array(mask[None, ...])  # Add channel dimension
        edge = jt.array(edge.transpose(2, 0, 1))  # HWC to CHW
        
        return {
            'image': img,
            'mask': mask,
            'edge': edge,
            'image_path': img_path
        }
