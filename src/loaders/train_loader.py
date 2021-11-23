import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.config import config_dict
from utils.text_line import TextLine
from utils.transform import transform_train


class TrainSet(Dataset):

    def __init__(self, dataset_name, short_side):
        config = config_dict[dataset_name]
        self.img_dir, self.cache_dir = config[0], config[2]
        self.img_names = sorted(os.listdir(self.img_dir))
        if not os.path.isdir(self.cache_dir):
            raise FileNotFoundError('Please run \'cache.py -name DATASET_NAME\' first.')

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.short_side = short_side

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img = self.get_img(item)
        p_pc_pe_all, ignored_vertices_all = self.get_p_pc_pe_all_ignored_vertices_all(item)
        img_width, img_height = img.size

        # Generate tls
        tls = []
        for p_pc_pe in p_pc_pe_all:
            tls.append(TextLine(img_width, img_height, p_pc_pe=p_pc_pe, ignore=False))
        for ignored_vertices in ignored_vertices_all:
            tls.append(TextLine(img_width, img_height, vertices=ignored_vertices, ignore=True))

        # Transform img & tls
        img, tls = transform_train(img, tls, self.short_side)

        # Get img_tensor, label_tensor, and ignored_mask
        img_tensor = self.normalize(self.to_tensor(img))  # Returns a torch.float32 type channel-first tensor.
        label_array = np.zeros((6, 512, 512), dtype='float64')
        ignored_mask = np.zeros((512, 512), dtype='float64')
        for tl in tls:
            if tl.ignore:
                ignored_mask = tl.draw_ignored_mask(ignored_mask)
            else:
                label_array = tl.draw_label(label_array)
        label_tensor = torch.from_numpy(label_array).float()  # To torch.float32 tensor.
        ignored_mask = torch.from_numpy(ignored_mask).float()  # To torch.float32 tensor.
        return img_tensor, label_tensor, ignored_mask

    def get_img(self, item):
        return Image.open(os.path.join(self.img_dir, self.img_names[item]))

    def get_p_pc_pe_all_ignored_vertices_all(self, item):
        p_pc_pe_all, ignored_vertices_all = [], []
        item_dir = os.path.join(self.cache_dir, str(item))
        cache_names = os.listdir(item_dir)
        for cache_name in cache_names:
            if cache_name[-11:-4] == 'ignored':
                ignored_vertices_all.append(np.load(os.path.join(item_dir, cache_name)))
            else:
                p_pc_pe_all.append(np.load(os.path.join(item_dir, cache_name)))
        return p_pc_pe_all, ignored_vertices_all


class TrainLoader(DataLoader):

    def __init__(self, dataset_name, batch_size, short_side, num_workers=32):
        super().__init__(dataset=TrainSet(dataset_name=dataset_name, short_side=short_side),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True)
