import numpy as np
from bentley_ottmann.base import edges_intersect

from utils.exceptions import InvalidVerticesError
from .utils.fem_solve import fem_solve
from .utils.linear_near_interp import linear_near_interp
from .utils.rasterize_plg import rasterize_plg


def merge_same(vertices, corner_idxs):
    mask = np.append((vertices[1:] != vertices[:-1]).any(axis=-1), True)
    vertices = vertices[mask]
    corner_mask = np.zeros((len(mask),), dtype=bool)
    corner_mask[corner_idxs] = True
    p_mask = corner_mask & (~mask)
    np_mask = mask | p_mask
    corner_idxs = np.where(corner_mask[np_mask][np.append(True, ~p_mask[np_mask][:-1])])[0]
    return vertices, corner_idxs


def to_ccw(vertices, corner_idxs):
    xs, ys = vertices.T.copy()
    xs -= np.roll(xs, 1)
    ys += np.roll(ys, 1)
    return (vertices, corner_idxs) if (xs * ys).sum() < 0. else (vertices[::-1], len(vertices) - 1 - corner_idxs[::-1])


def is_invalid(vertices, corner_idxs):
    cond1 = edges_intersect([(float(x), float(y)) for (x, y) in vertices], accurate=True)  # Self-intersection check.
    cond2 = len(corner_idxs) != 4
    return cond1 or cond2


def get_w_c_h_c(f_inv, n=500):
    """sample_num == n ** 2"""
    pts_grid_c_mid = np.stack(np.meshgrid(np.linspace(0., 1., n, dtype='float64'),
                                          np.linspace(0., 1., n, dtype='float64')),
                              axis=2).reshape((-1, 2))
    pts_grid_s = f_inv(pts_grid_c_mid).reshape([n, n, 2])
    w_c = np.sum(np.linalg.norm(pts_grid_s[:, 1:] - pts_grid_s[:, :-1], axis=-1), axis=1).mean()
    h_c = np.sum(np.linalg.norm(pts_grid_s[1:] - pts_grid_s[:-1], axis=-1), axis=0).mean()
    return w_c, h_c


def get_c_mid_skel_edge(pts: np.ndarray, w_c, h_c):
    scale_factor = [w_c, h_c]
    pts = (scale_factor * pts).copy()
    w_r, h_r = min(h_c / 2., w_c / 2.), h_c / 2.

    # Get pts_skel.
    pts_skel = pts.copy()
    x_skel, y_skel = pts_skel.T
    x_skel[:], y_skel[:] = x_skel.clip(w_r, w_c - w_r), h_r

    # Get pts_edge
    pts_edge = pts.copy()
    l_inf_norm = np.linalg.norm([1. / w_r, 1. / h_r] * (pts - pts_skel), ord=np.inf, axis=1)
    mask = l_inf_norm > 1e-5
    pts_edge[mask] = pts_skel[mask] + (pts - pts_skel)[mask] / l_inf_norm[mask, None]
    pts_edge[~mask] = pts_skel[~mask] + [0., h_r]
    return pts_skel / scale_factor, pts_edge / scale_factor


def plg_mapper(vertices, corner_idxs):
    """
    Get all the integral points as well as their corresponding skel/edge points.

    Parameters:
        vertices: A numpy array (float64) with shape (-1, 2).
        corner_idxs: A numpy array (int64) with shape (4,).

    Returns:
        A numpy array (float64) with shape (3, -1, 2).
    """
    # Regulate and validate the inputs.
    vertices, corner_idxs = to_ccw(*merge_same(vertices, corner_idxs))
    if is_invalid(vertices, corner_idxs):
        raise InvalidVerticesError('The given vertices are invalid.')

    # Solve (fem) the laplace equation and perform linear_near interpolation.
    pts_mesh_s, pts_mesh_c_mid = fem_solve(vertices, corner_idxs)
    f, f_inv = map(linear_near_interp, (pts_mesh_s, pts_mesh_c_mid), (pts_mesh_c_mid, pts_mesh_s))

    # Get the results for all the integral points within the polygon.
    pts_all_s = rasterize_plg(vertices)
    return np.stack((pts_all_s, *map(f_inv, get_c_mid_skel_edge(f(pts_all_s), *get_w_c_h_c(f_inv)))))
