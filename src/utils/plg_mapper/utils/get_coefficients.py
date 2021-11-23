import numpy as np


def section(input_array, start, stop, axis=0):
    """
    View input_array as a ring and return a slice of it.
    """
    return np.roll(input_array, -start, axis=axis)[:stop - start]


def normal_vector(x, y, z):
    """
    Find the normal vector of (x, y, z) s.t. z = -1.. If it doesn't exist, returns a zero vector.

    Parameters:
        x: a float number.
        y: a float number.
        z: a float number.

    Returns:
        A numpy array (float64) with shape (3,)
    """
    if x != 0.:
        return np.array([(z - y) / x, 1., -1.], dtype='float64')
    elif y != 0.:
        return np.array([1., (z - x) / y, -1.], dtype='float64')
    else:
        return np.zeros((3,), dtype='float64')


def get_length_ratios(pts):
    """
    Get length_ratios for piecewise linear interpolation.

    Parameters:
        pts: a numpy arrays (float64) with shape (2,). They're 2D points.

    Returns:
        A boundary function describing the the 2D line segment.
    """
    piecewise_lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    return np.append(0., piecewise_lengths / piecewise_lengths.sum()).cumsum()


def get_lin2d(p1, p2):
    """
    Find a linear function f(x, y) = a * x + b * y + c s.t.:
    1. f(x1, y1) = z1
    2. f(x2, y2) = z2
    Return the coefficients (a, b, c)

    Parameters:
        p1: a numpy array (float64) with shape (3,). It's a 3D point.
        p2: a numpy array (float64) with shape (3,). It's a 3D point.
    """
    u = normal_vector(*(p2 - p1))
    return u[0], u[1], -np.dot(p1, u)


vertices_c_mid = np.array([[0., 0.],
                           [1., 0.],
                           [1., 1.],
                           [0., 1.]], dtype='float64')


def get_coefficients(vertices, corner_idxs):
    coefficients_x = np.zeros((len(vertices), 3), dtype='float64')
    coefficients_y = coefficients_x.copy()
    counter = 0
    for idx_1 in range(4):
        polyline_s = section(vertices, corner_idxs[idx_1], corner_idxs[(idx_1 + 1) % 4] + 1)
        polyline_s_c_mid = section(vertices_c_mid, idx_1, (idx_1 + 1) % 4 + 1)

        length_ratios = get_length_ratios(polyline_s)
        length_x, length_y = polyline_s_c_mid[1] - polyline_s_c_mid[0]
        x_c_mid = polyline_s_c_mid[0, 0] + length_ratios * length_x
        y_c_mid = polyline_s_c_mid[0, 1] + length_ratios * length_y

        pts_x = np.concatenate((polyline_s, x_c_mid[:, None]), axis=1)
        pts_y = np.concatenate((polyline_s, y_c_mid[:, None]), axis=1)

        for idx_2 in range(len(pts_x) - 1):
            coefficients_x[counter] = get_lin2d(pts_x[idx_2], pts_x[idx_2 + 1])
            coefficients_y[counter] = get_lin2d(pts_y[idx_2], pts_y[idx_2 + 1])
            counter += 1
    return coefficients_x, coefficients_y
