# File containing the function GaussKernel which computes the Gauss kernel regression estimate

import numpy as np
import numpy.matlib as npm
from scipy.spatial import distance_matrix


def GaussKernel2D(x_extended, xs_extended, ys, bandwidth=1, nan2zeros=False):
    """
    Compute Gauss kernel regression estimate

    :param x: 1d-array, x-location where the values have to be estimated
    :param xs: 1d-array, x-value of the data points
    :param ys: 1d-array, y-value of the data points
    :param bandwidth: float>0, smoothing parameter

    :return: 1d-array, estimated values at x
    """


    ys_extended = npm.repmat(ys, len(x_extended), 1)

    def K(u):
        """
        The kernel
        """
        return np.exp(-np.square(u) / 2) / (np.square(2 * np.pi))

    # Return the regression estimate
    result = np.sum(ys_extended.T * K(distance_matrix(xs_extended, x_extended) / bandwidth), axis=0) / np.sum(
        K(distance_matrix(xs_extended, x_extended) / bandwidth), axis=0)

    return result

