import cv2
import numpy as np

from utils.exceptions import InvalidVerticesError
from utils.simple_polygon import SimplePolygon


def rotate_pts(pts, rot_degree, rot_center):
    pts = pts.copy()  # Avoid mutation.
    radian = np.pi * rot_degree / 180.
    i_s, j_s = np.split(pts, 2, axis=-1)
    i_c, j_c = rot_center
    j_s, j_c = -j_s, -j_c
    i_s -= i_c
    j_s -= j_c
    j_s_new = -j_s * np.cos(radian) + i_s * np.sin(radian) - j_c
    i_s_new = j_s * np.sin(radian) + i_s * np.cos(radian) + i_c
    return np.concatenate((i_s_new, j_s_new), axis=-1)


class TextLine:
    def __init__(self, img_width, img_height, p_pc_pe=None, vertices=None, ignore=False):
        """
        Parameters:
            img_width: int
            img_height: int
            p_pc_pe: float64
            vertices: float64
            ignore: bool
        """
        self.img_width, self.img_height = map(float, (img_width, img_height))
        self.p_pc_pe = p_pc_pe
        self.vertices = vertices
        self.ignore = ignore

    def rotate(self, rot_degree):
        rot_center = (self.img_height / 2., self.img_width / 2.)
        if self.vertices is not None:
            self.vertices = rotate_pts(self.vertices, rot_degree, rot_center)
        if self.p_pc_pe is not None:
            self.p_pc_pe = rotate_pts(self.p_pc_pe, rot_degree, rot_center)

    def resize(self, scale_i, scale_j):
        self.img_height *= scale_i
        self.img_width *= scale_j
        if self.vertices is not None:
            self.vertices *= [scale_i, scale_j]
        if self.p_pc_pe is not None:
            self.p_pc_pe *= [scale_i, scale_j]

    def horizontal_flip(self):
        if self.vertices is not None:
            tmp = self.vertices[..., 1]
            tmp[:] = self.img_width - tmp
        if self.p_pc_pe is not None:
            tmp = self.p_pc_pe[..., 1]
            tmp[:] = self.img_width - tmp

    def vertical_flip(self):
        if self.vertices is not None:
            tmp = self.vertices[..., 0]
            tmp[:] = self.img_height - tmp
        if self.p_pc_pe is not None:
            tmp = self.p_pc_pe[..., 0]
            tmp[:] = self.img_height - tmp

    def shift(self, shift_i, shift_j):
        if self.vertices is not None:
            self.vertices += [shift_i, shift_j]
        if self.p_pc_pe is not None:
            self.p_pc_pe += [shift_i, shift_j]

    def get_polygon(self):
        """InvalidVerticesError will be raised if the given vertices are invalid."""
        return SimplePolygon(self.vertices)

    def draw_label(self, label_array):
        # Get p, v1, v2.
        p, pc, pe = self.p_pc_pe
        v1, v2 = pc - p, pe - p
        mask_range = ((p >= 0.) & (p <= 511.)).all(axis=-1)
        if not mask_range.any():
            return label_array
        p, mask_unique = np.unique(np.round(p[mask_range]).astype('int64'), axis=0, return_index=True)
        v1, v2 = v1[mask_range][mask_unique], v2[mask_range][mask_unique]

        # Get v21_unit.
        v21 = v1 - v2
        v21_unit = v21.copy()
        l2_norm = np.linalg.norm(v21, axis=-1)
        mask = l2_norm > 1e-5
        v21_unit[mask] = v21[mask] / l2_norm[mask, None]
        v21_unit[~mask] = [1., 0.]

        # Get features.
        i_v21_unit, j_v21_unit = v21_unit.T
        pi, pj = np.sign(i_v21_unit).clip(0.), np.sign(j_v21_unit).clip(0.)
        a = np.arccos(np.abs(i_v21_unit)) * 180. / np.pi
        oc, oe = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)

        # Draw label.
        i_p, j_p = p.T
        label_array[0, i_p, j_p] = 1.
        label_array[1:6, i_p, j_p] = np.stack((pi, pj, a, oc, oe))
        return label_array

    def draw_ignored_mask(self, ignored_mask):
        try:
            contours = self.get_polygon().contours()
        except InvalidVerticesError:
            return ignored_mask
        return cv2.drawContours(ignored_mask, [np.round(contour).astype('int64')[:, ::-1] for contour in contours],
                                contourIdx=-1, color=1., thickness=-1)
