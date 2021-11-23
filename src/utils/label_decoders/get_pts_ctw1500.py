import numpy as np


def get_pts_ctw1500(f_path, test_mode=False):
    """
    Decode ctw1500 labels.
    Returns:
        vertices_all: a list of numpy arrays (float64), each with shape (14, 2).
        corner_idxs_all: a list of numpy arrays (int64), each with shape (4,).
        ignored_vertices_all: an empty list.
    """
    with open(f_path) as f:
        pts_strings: 'list' = f.readlines()

    line_num = len(pts_strings)
    for idx in range(line_num):
        pts_strings[idx] = pts_strings[idx].split(',')

    pts = np.array(pts_strings, dtype='float64').reshape([line_num, -1, 2])
    vertices_all = (pts[:, 0:1] + pts[:, 2:])[..., ::-1]

    ignored_vertices_all = []
    if test_mode:
        return map(list, (vertices_all, ignored_vertices_all))
    else:
        corner_idxs_all = np.zeros((len(vertices_all), 4), dtype='int64')
        corner_idxs_all[:] = [0, 6, 7, 13]
        return map(list, (vertices_all, corner_idxs_all, ignored_vertices_all))
