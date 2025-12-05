"""
BDD100K semantic segmentation dataset wrapper.

This Dataset:
- Loads RGB images from a given root folder.
- Loads corresponding segmentation masks (one class id per pixel).
- Applies transforms to images and masks (resize, flip, tensor conversion).
"""

import os
from glob import glob
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class BDD100KSegmentation(Dataset):
    def __init__(self, root: str, split: str = "train", image_size: Tuple[int, int] = (256, 256)):
        self.root = root
        self.split = split
        self.image_size = image_size

        images_dir = os.path.join(root, "images", "seg", split)
        masks_dir  = os.path.join(root, "labels", "seg", split)
        self.image_paths: List[str] = sorted(glob(os.path.join(images_dir, "*.jpg")))
        self.mask_paths: List[str] = sorted(glob(os.path.join(masks_dir, "*.png")))

        assert len(self.image_paths) == len(self.mask_paths), (
            f"Mismatch: {len(self.image_paths)} images vs "
            f"{len(self.mask_paths)} masks"
        )

        # ImageNet normalization (for pretrained ResNet backbone)
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        # Image: resize -> tensor -> normalize
        self.img_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Mask: resize (nearest) -> tensor -> (H, W) long
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
            transforms.PILToTensor(),                         # (1, H, W) uint8
            transforms.Lambda(lambda x: x.squeeze(0).long())  # (H, W)   int64
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index])

        # Convert to tensors
        image = self.img_transform(image)   # (3, H, W), float32
        mask  = self.mask_transform(mask)   # (H, W),   int64

        return image, mask
