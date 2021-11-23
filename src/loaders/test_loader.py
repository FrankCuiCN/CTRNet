import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.config import config_dict
from utils.exceptions import InvalidVerticesError
from utils.text_line import TextLine
from utils.transform import transform_test


class TestSet(Dataset):

    def __init__(self, dataset_name, short_side):
        config = config_dict[dataset_name]
        self.img_dir, self.label_dir, self.label_decoder = config[0], config[1], config[3]
        self.img_names, self.label_names = sorted(os.listdir(self.img_dir)), sorted(os.listdir(self.label_dir))

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.short_side = short_side

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img = self.get_img(item)
        img_width, img_height = img.size
        vertices_all, ignored_vertices_all = self.get_vertices_all_ignored_vertices_all(item)

        # Generate tls
        tls = []
        for vertices in vertices_all:
            tls.append(TextLine(img_width, img_height, vertices=vertices, ignore=False))
        for ignored_vertices in ignored_vertices_all:
            tls.append(TextLine(img_width, img_height, vertices=ignored_vertices, ignore=True))

        # Transform img & tls
        img, tls = transform_test(img, tls, self.short_side)

        # Get out_x, out_y, and ignored_plgs
        img_tensor = self.normalize(self.to_tensor(img))  # Returns a torch.float32 type channel-first tensor.
        y_plgs = []
        ignored_plgs = []
        for tl in tls:
            if tl.ignore:
                try:
                    ignored_plgs.append(tl.get_polygon())
                except InvalidVerticesError:
                    pass
            else:
                try:
                    y_plgs.append(tl.get_polygon())
                except InvalidVerticesError:
                    pass
        return img_tensor, y_plgs, ignored_plgs

    def get_img(self, item):
        return Image.open(os.path.join(self.img_dir, self.img_names[item]))

    def get_vertices_all_ignored_vertices_all(self, item):
        return self.label_decoder(os.path.join(self.label_dir, self.label_names[item]), test_mode=True)


class TestLoader(DataLoader):

    def __init__(self, dataset_name, short_side, num_workers=32):
        super().__init__(dataset=TestSet(dataset_name=dataset_name, short_side=short_side),
                         batch_size=1,
                         shuffle=False,
                         num_workers=num_workers,
                         collate_fn=self.collate_fn,
                         pin_memory=True,
                         drop_last=False)

    @staticmethod
    def collate_fn(data):
        x = data[0][0][None]
        y_plgs = data[0][1]
        ignored_masks = data[0][2]
        return x, y_plgs, ignored_masks
