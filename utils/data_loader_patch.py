import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
from natsort import natsorted

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str,img_size):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.size = img_size
        self.ids = [splitext(file)[0] for file in natsorted(listdir(images_dir)) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, is_mask, size):
        if is_mask:
            img = np.array(img, dtype=np.uint8)
            img = cv2.resize(img, (size,size))

            img = img/255.0

            img = np.expand_dims(img, axis=2)
        else:
            img = np.array(img, dtype=np.float32)
            mean = np.mean(img)
            std = np.std(img)
            img = (img-mean)/std
            img = cv2.resize(img, (size,size))
        img = np.transpose(img, (2, 0, 1))

        return img

    @classmethod
    def load(cls, filename,is_mask):
        ext = splitext(filename)[1]
      
        if ext in ['.npz', '.npy']:
           
            return Image.fromarray(np.uint8(np.load(filename)))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            if is_mask:
                return Image.open(filename).convert('L')
            else:
                return Image.open(filename).convert('RGB')

    def __getitem__(self, idx):
        name = self.ids[idx]
        num = name.split('_', 2)[1]
     
        #mask_file = list(self.masks_dir.glob('mask_'+num + '.*'))
        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0], is_mask=True)
      
        img = self.load(img_file[0], is_mask=False)
       
        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, is_mask=False, size=self.size)
        mask = self.preprocess(mask, is_mask=True, size=self.size)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, img_size):
        super().__init__(images_dir, masks_dir, img_size)
