import logging
from glob import glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
import os
from os.path import splitext
from natsort import natsorted


class BasicDataset(Dataset):
    def __init__(self, list_id, images_dir: str, masks_dir: str, img_size):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.size = img_size
        self.ids = natsorted(list_id)
        # self.img_path_list = natsorted(glob(str(images_dir) + "/*.jpg"))
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, is_mask, size):
        if is_mask:
            img = np.array(img, dtype=np.uint8)
            img = cv2.resize(img, (size, size))
            img = img / 255.0  # 255 -> 1
            img = np.expand_dims(img, axis=2)
        else:
            img = np.array(img, dtype=np.float32)
            mean = np.mean(img)
            std = np.std(img)
            # if std == 0:
            #     std = 1e-3
            img = (img - mean) / std  # 标准化
            img = cv2.resize(img, (size, size))
        return np.transpose(img, (2, 0, 1))

    @classmethod
    def load(cls, filename, is_mask):
        if is_mask:
            return Image.open(filename).convert('L')
        else:
            return Image.open(filename).convert('RGB')

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = os.path.join(self.images_dir, name + '.jpg')
        mask_path = os.path.join(self.masks_dir, name + '.png')
        # img_path = self.img_path_list[idx]
        # mask_path = img_path.replace('img_dir', 'ann_dir')
        # mask_path = img_path.replace('img_dir', 'ann_dir').replace('.jpg', '.png')

        img = self.load(img_path, is_mask=False)
        mask = self.load(mask_path, is_mask=True)

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, is_mask=False, size=self.size)
        mask = self.preprocess(mask, is_mask=True, size=self.size)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
