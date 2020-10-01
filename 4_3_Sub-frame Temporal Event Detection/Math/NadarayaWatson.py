# File containing the function GaussKernel which computes the Gauss kernel regression estimate

import numpy as np
import numpy.matlib as npm


def GaussKernel(x, xs, ys, bandwidth=1, nan2zeros=False):
    """
    Compute Gauss kernel regression estimate

    :param x: 1d-array, x-location where the values have to be estimated
    :param xs: 1d-array, x-value of the data points
    :param ys: 1d-array, y-value of the data points
    :param bandwidth: float>0, smoothing parameter

    :return: 1d-array, estimated values at x
    """

    def K(u):
        """
        The kernel
        """
        return np.exp(-np.square(u) / 2) / (np.square(2 * np.pi))

    x_extended = npm.repmat(x, len(xs), 1)
    xs_extended = npm.repmat(xs, len(x), 1)
    ys_extended = npm.repmat(ys, len(x), 1)

    # Return the regression estimate
    if nan2zeros:
        result = np.sum(ys_extended * K((xs_extended - x_extended.T) / bandwidth), axis=1) / np.sum(
            K((xs_extended - x_extended.T) / bandwidth), axis=1)
        result[np.isnan(result)] = 0
        return result
    else:
        return np.sum(ys_extended * K((xs_extended - x_extended.T) / bandwidth), axis=1) / np.sum(
            K((xs_extended - x_extended.T) / bandwidth), axis=1)

