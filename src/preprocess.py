import argparse
import os

from PIL import Image

from utils.config import config_dict


def preprocess(dataset_name):
    img_dir = config_dict[dataset_name][0]
    for idx, img_name in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        Image.open(img_path).convert('RGB').save(img_path)  # convert to RGB and clear Exif data
        print(idx, '<<<<<', end='\r')
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str)
    args = parser.parse_args()

    preprocess(args.name)
