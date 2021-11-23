import numpy as np
from PIL import Image
from torchvision import transforms

random_color = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)


def test_resize(img, tls, short_side):
    width, height = img.size

    # Get new_width, new_height.
    scale = short_side / min(width, height)
    new_width, new_height = int(64. * np.round(scale * width / 64.)), int(64. * np.round(scale * height / 64.))

    # Perform resizing.
    img = img.resize((new_width, new_height), resample=Image.BILINEAR)
    for tl in tls:
        tl.resize(new_height / height, new_width / width)
    return img, tls


def random_resize(img, tls, short_side):
    width, height = img.size

    # Get new_width, new_height.
    scale = np.random.choice((0.8, 1., 1.25)) * short_side / min(width, height)
    new_width, new_height = int(np.round(scale * width)), int(np.round(scale * height))

    # Perform resizing.
    img = img.resize((new_width, new_height), resample=Image.BILINEAR)
    for tl in tls:
        tl.resize(new_height / height, new_width / width)
    return img, tls


def random_horizontal_flip(img, tls):
    # Perform random flipping.
    if np.random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        for tl in tls:
            tl.horizontal_flip()
    return img, tls


def random_rotate(img, tls):
    # Get angle.
    angle = np.random.uniform(-10., 10.)

    # Perform rotation.
    img = img.rotate(angle, resample=Image.BILINEAR)
    for tl in tls:
        tl.rotate(angle)
    return img, tls


def random_crop(img, tls):
    # Get the min/max i/j for i_start/j_start.
    width, height = img.size
    ij_min_max_all, h_, w_ = [], height - 512, width - 512
    for tl in tls:
        if tl.ignore:
            continue
        i_tmp, j_tmp = tl.p_pc_pe.T
        ij_min_max_all += [[i_tmp.min(), j_tmp.min(), i_tmp.max(), j_tmp.max()]]
    if len(ij_min_max_all) == 0:
        i_min, j_min, i_max, j_max = 0, 0, h_, w_
    else:
        tmp_1, tmp_2, tmp_3, tmp_4 = np.array(ij_min_max_all, dtype='int64').T
        ij_min_max = np.array([tmp_1.min(), tmp_2.min(), tmp_3.max(), tmp_4.max()], dtype='int64') - 256
        i_min, j_min, i_max, j_max = np.clip(ij_min_max, 0, (h_, w_, h_, w_))

    # Get i/j start/stop
    i_start, j_start = np.random.randint(i_min, i_max + 1), np.random.randint(j_min, j_max + 1)
    i_stop, j_stop = i_start + 512, j_start + 512

    # Perform cropping.
    img = img.crop((j_start, i_start, j_stop, i_stop))
    for tl in tls:
        tl.shift(-i_start, -j_start)
    return img, tls


def transform_train(img, tls, short_side):
    img, tls = random_resize(img, tls, short_side)
    img, tls = random_horizontal_flip(img, tls)
    img, tls = random_rotate(img, tls)
    img, tls = random_crop(img, tls)
    img = random_color(img)
    return img, tls


def transform_test(img, tls, short_side):
    img, tls = test_resize(img, tls, short_side)
    return img, tls
