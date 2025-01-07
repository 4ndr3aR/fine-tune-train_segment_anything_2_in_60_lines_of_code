import os

import cv2

import random

import numpy as np

from torch.utils.data import Dataset

from dbgprint import dbgprint
from dbgprint import *

# Dataset class for LabPicsV1 dataset
class LabPicsDataset(Dataset):
    def __init__(self, data_dir, split="Train"):
        self.data_dir = data_dir
        self.split = split
        self.data = []
        for idx, name in enumerate(os.listdir(os.path.join(data_dir, "Simple", self.split, "Image"))):
            if idx % 1000 == 0:
                print('.', end='', flush=True)
            self.data.append({
                "image": os.path.join(data_dir, "Simple", self.split, "Image", name),
                "annotation": os.path.join(data_dir, "Simple", self.split, "Instance", name[:-4] + ".png")
            })
        dbgprint(dataloader, LogLevel.INFO, f"\nLoaded {len(self.data)} images for {self.split} set")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]
        Img = cv2.imread(ent["image"])[..., ::-1]
        ann_map = cv2.imread(ent["annotation"])

        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

        Img = np.pad(Img, ((0, 1024 - Img.shape[0]), (0, 1024 - Img.shape[1]), (0, 0)), 'constant')
        ann_map = np.pad(ann_map, ((0, 1024 - ann_map.shape[0]), (0, 1024 - ann_map.shape[1]), (0, 0)), 'constant')

        mat_map = ann_map[:, :, 0]
        ves_map = ann_map[:, :, 2]
        mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)

        inds = np.unique(mat_map)[1:]
        if len(inds) > 0:
            ind = np.random.choice(inds)
            mask = (mat_map == ind).astype(np.uint8)
            coords = np.argwhere(mask > 0)
            yx = coords[np.random.randint(len(coords))]
            point = [[yx[1], yx[0]]]
        else:
            mask = np.zeros_like(mat_map, dtype=np.uint8)
            point = [[0, 0]]  # Provide a default point

        dbgprint(dataloader, LogLevel.TRACE, f'{type(Img)} {type(mask)} {type(point)}')
        dbgprint(dataloader, LogLevel.TRACE, f"Input points len: {len(point)}")
        dbgprint(dataloader, LogLevel.TRACE, f"Input points: {point}")
        return Img, mask, point

