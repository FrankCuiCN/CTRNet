import numpy as np
from matplotlib import path


def rasterize_plg(vertices, eps=1e-5):
    """
    Get all the integral points within a given polygon.
    Note that the input vertices need to be counterclockwise.
    The eps is used for loosening.

    Parameters:
        vertices: a numpy array (float64) with shape (-1, 2). The vertices need to be counterclockwise.
        eps: a float number.

    Returns:
        A numpy array (float64) with shape (-1, 2). All of the integral points within the given polygon.
    """
    tolerance = 5.
    x_min, x_max = vertices[:, 0].min() - tolerance, vertices[:, 0].max() + tolerance
    y_min, y_max = vertices[:, 1].min() - tolerance, vertices[:, 1].max() + tolerance

    points_all = np.stack(np.indices((int(x_max - x_min), int(y_max - y_min)), dtype='float64')).T.reshape((-1, 2))
    points_all += np.array([x_min, y_min], dtype='float64')
    return points_all[path.Path(vertices).contains_points(points_all, radius=eps)]
