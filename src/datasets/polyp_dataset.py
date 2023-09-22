import os
from PIL import Image, ImageFile
import torch
import numpy as np

from torch.utils.data import Dataset


class PolypDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
    ):
        self.data_dir = data_dir
        self.image_size = image_size

        # Get all the images in the data directory
        self.image_paths = [
            os.path.join(data_dir, "images", image_name)
            for image_name in os.listdir(os.path.join(data_dir, "images"))
        ]

        # Get all the masks in the data directory
        self.mask_paths = [
            os.path.join(data_dir, "masks", mask_name)
            for mask_name in os.listdir(os.path.join(data_dir, "masks"))
        ]

        # Sort the images and masks to ensure they correspond
        self.image_paths = sorted(self.image_paths)
        self.mask_paths = sorted(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Open the image and mask using PIL
        image = Image.open(image_path).resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image.convert("RGB"))

        mask = Image.open(mask_path).resize(
            (self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask.convert("L"))

        # Convert to Torch tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        # Threshold the mask to values 0 and 1
        mask = (mask > 0.5).float()

        return {
            "image": image,  # Shape: (3, H, W) = (3, image_size, image_size)
            "mask": mask  # Shape: (1, H, W) = (1, image_size, image_size)
        }
