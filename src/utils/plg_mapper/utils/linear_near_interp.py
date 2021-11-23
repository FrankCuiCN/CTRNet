import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def linear_near_interp(points, values):
    """
    Parameters:
        points: a numpy array (float64) with shape (-1, 2).
        values: a numpy array (float64) with shape (-1, 2).
    """
    f_linear = LinearNDInterpolator(points, values)
    f_near = NearestNDInterpolator(points, values)

    def f(x):
        """
        Parameters:
            x: a numpy array (float64) with shape (-1, 2).
        """
        y = f_linear(x)
        mask_nan = np.isnan(y).all(axis=1)
        y[mask_nan] = f_near(x[mask_nan])
        return y

    return f
