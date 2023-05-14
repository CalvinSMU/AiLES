import logging
from glob import glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
from os.path import splitext
from natsort import natsorted


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, img_size):
        self.images_dir = Path(images_dir)
        self.size = img_size
        self.ids = [splitext(file)[0].replace('\\', '/').split('/')[-1]
                    for file in natsorted(glob(str(images_dir) + "/*.jpg"))
                    if not file.startswith('.')]
        self.img_path_list = natsorted(glob(str(images_dir) + "/*.jpg"))
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, size):
        img = np.array(img, dtype=np.float32)
        mean = np.mean(img)
        std = np.std(img)
        if std == 0:
            std = 1e-3
        img = (img - mean) / std  # 标准化
        img = cv2.resize(img, (size, size))
        return np.transpose(img, (2, 0, 1))

    @classmethod
    def load(cls, filename):
        return Image.open(filename).convert('RGB')

    def __getitem__(self, idx):
        name = self.ids[idx]

        img_path = self.img_path_list[idx]
        img = self.load(img_path)

        img = self.preprocess(img, size=self.size)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous()
        }
