import cv2
import numpy as np

from utils.exceptions import InvalidVerticesError
from utils.simple_polygon import SimplePolygon


def draw_mask(cart_list, width, height, color, resolution):
    mask = np.zeros((height, width), dtype='float64')
    i1, j1, i2, j2 = cart_list.T
    i21, j21 = i2 - i1, j2 - j1
    for shrinkage in np.linspace(0., 1., resolution):
        mask[np.round(i1 + shrinkage * i21).astype('int64'),
             np.round(j1 + shrinkage * j21).astype('int64')] = color
    return mask


def morph_close(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)
    return mask


def merge(mask):
    return cv2.connectedComponents(morph_close(mask))[1]


def pred_decoder(pred, decoder_settings, require_picture=False):
    """
    Input type (pred): numpy array (float64)
    Input shape (pred): C x H x W
    """
    pred = pred.copy()
    height, width = pred.shape[1:3]
    conf_threshold, shrinkage = decoder_settings

    # Process pred and get v21, center_list.
    mask_conf = pred[0] >= conf_threshold
    pi, pj, a, oc, oe = pred[1:, mask_conf]
    pi, pj = np.where(pi >= 0.5, 1., -1.), np.where(pj >= 0.5, 1., -1.)
    a[:], oc[:], oe[:] = a.clip(0., 90.) * np.pi / 180., oc.clip(0.), oe.clip(0.)  # 'pred' shall be manipulated.
    v21_unit = np.stack((pi * np.cos(a), pj * np.sin(a)), axis=1)
    v21 = v21_unit * (oc + oe)[:, None]
    center_list = v21_unit * oc[:, None] + np.indices((height, width), dtype='float64').transpose((1, 2, 0))[mask_conf]

    # Get cart_list.
    cart_list = np.concatenate((center_list - v21, center_list + v21), axis=1)
    cart_list = cart_list.clip(0., (height - 1., width - 1., height - 1., width - 1.)).round()

    # Get cart_list_skel.
    v21_shrunk = shrinkage * v21
    cart_list_skel = np.concatenate((center_list - v21_shrunk, center_list + v21_shrunk), axis=1)
    cart_list_skel = cart_list_skel.clip(0., (height - 1., width - 1., height - 1., width - 1.)).round()

    # Get instance_map (i.e. clustering).
    resolution = 20
    mask_skel = draw_mask(cart_list_skel, width, height, color=1., resolution=resolution)
    category = merge(mask_skel)[tuple(center_list.clip(0., (height - 1., width - 1.)).round().astype('int64').T)]
    instance_map = draw_mask(cart_list, width, height, color=category, resolution=resolution)

    # Get polygons and their info.
    plg_num = int(instance_map.max())
    pred_plgs, pred_plgs_info = [], -np.ones((plg_num, 5), dtype='float64')
    for idx in range(plg_num):
        mask = instance_map == idx + 1

        # Validate and get polygon.
        contours_all = cv2.findContours(morph_close(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(contours_all) == 0:
            continue
        try:
            plg = SimplePolygon(max(contours_all, key=cv2.contourArea)[:, 0, ::-1])
        except InvalidVerticesError:
            continue
        pred_plgs.append(plg)

        # Get the polygon info.
        pred_masked = pred[:, mask]
        conf_mean = pred_masked[0].mean()
        height = 4. * pred_masked[4:6].mean()
        width = plg.area() / height if height != 0. else 0.
        aspect_ratio = width / height if height != 0. else 0.
        rot_std = pred_masked[3].std()
        pred_plgs_info[idx] = [conf_mean, height, width, aspect_ratio, rot_std]
    pred_plgs_info = pred_plgs_info[pred_plgs_info[:, 0] != -1.]

    if require_picture:
        return pred_plgs, pred_plgs_info, [*pred, mask_skel, instance_map]
    else:
        return pred_plgs, pred_plgs_info
