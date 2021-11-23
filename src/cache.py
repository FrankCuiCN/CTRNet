import argparse
import os
import shutil

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.config import config_dict
from utils.exceptions import InvalidVerticesError
from utils.plg_mapper import plg_mapper


class PlgCaching(Dataset):
    """
    Calculate and store polygon info:
    1. All integral points within each polygon
    2. Corresponding skeleton points & edge points
    3. 'ignored_vertices_all'
    """

    def __init__(self, dataset_name, short_side):
        self.img_dir, self.label_dir, self.cache_dir, self.label_decoder = config_dict[dataset_name]
        self.img_names, self.label_names = sorted(os.listdir(self.img_dir)), sorted(os.listdir(self.label_dir))
        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.mkdir(self.cache_dir)

        self.short_side = short_side

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        # Basic settings
        vertices_all, corner_idxs_all, ignored_vertices_all = self.get_label(item)
        scale = self.get_scale(item)

        # Create subdirectory
        os.mkdir(os.path.join(self.cache_dir, str(item)))

        # Solve and store vertices_all.
        for idx, vertices in enumerate(vertices_all):
            # Solve polygon
            try:
                p_pc_pe_all = plg_mapper(vertices * scale, corner_idxs_all[idx]) / scale
            except InvalidVerticesError:  # If the input vertices are invalid for plg_mapper.
                ignored_vertices_all.append(vertices)  # The invalid polygons shall also be ignored.
            else:
                # Store the result
                np.save(os.path.join(self.cache_dir, str(item), str(idx)), p_pc_pe_all)

        # Store ignored_vertices_all.
        for idx, ignored_vertices in enumerate(ignored_vertices_all):
            np.save(os.path.join(self.cache_dir, str(item), str(idx) + '_ignored'), ignored_vertices)

    def get_label(self, item):
        return self.label_decoder(os.path.join(self.label_dir, self.label_names[item]))

    def get_scale(self, item):
        img_path = os.path.join(self.img_dir, self.img_names[item])
        width, height = Image.open(img_path).size
        maximum_resize_rate, super_sampling_rate = 1.25, 1.5
        return super_sampling_rate * maximum_resize_rate * self.short_side / min(width, height)


def cache_plg(dataset_name, short_side, num_workers=32):
    def collate_fn(x):
        return x

    p_pc_pe_cache_loader = DataLoader(dataset=PlgCaching(dataset_name, short_side),
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn,
                                      pin_memory=True,
                                      drop_last=False)

    print('Caching data...')
    for idx, p_pc_pe_all in enumerate(p_pc_pe_cache_loader):
        print('Current progress:', str(idx + 1) + '/' + str(len(p_pc_pe_cache_loader)), '<<<<<', end='\r')
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str)
    parser.add_argument('-short_side', type=float, default=640.)
    parser.add_argument('-num_workers', type=int, default=32)
    args = parser.parse_args()

    cache_plg(args.name, args.short_side, args.num_workers)
