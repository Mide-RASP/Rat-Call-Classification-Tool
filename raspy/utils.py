from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import itertools

import numpy as np
import numba as nb
import scipy.signal as signal
import time

# import llvmlite.binding as llvm
# llvm.set_option('', '--debug-only=loop-vectorize')

#from numpy.core.defchararray import lower

@nb.njit
def zero_pad(mat, pad_shape, mat_shape):
    even_transforms = (pad_shape + 1) % 2
    temp = pad_shape + mat_shape - even_transforms  #This should be given to np.zeros as the size parameter, but Numba throws a fit
    pad_shape //= 2
    answer = np.zeros((temp[0], temp[1]), dtype=mat.dtype)
    answer[pad_shape[0]:mat.shape[0] + pad_shape[0], pad_shape[1]:mat.shape[1] + pad_shape[1]] = mat
    return answer

@nb.njit
def conv2d(mat, kernel):
    output_shape = (mat.shape[0] - kernel.shape[0] + 1, mat.shape[1] - kernel.shape[1] + 1) # temp = tuple(mat_shape - kernel_shape + 1)
    conv_input_shape = kernel.shape + output_shape
    strides = (mat.strides[0], mat.strides[1], mat.strides[0], mat.strides[1])
    subM = np.lib.stride_tricks.as_strided(mat, shape=conv_input_shape, strides=strides)

    # All the code from here to the end of this function is to replicate this:  return np.einsum('ij,ijkl->kl', f, subM)
    answer = np.zeros(output_shape, mat.dtype)
    for k in xrange(conv_input_shape[2]):
        for l in xrange(conv_input_shape[3]):
            for i in xrange(conv_input_shape[0]):
                for j in xrange(conv_input_shape[1]):
                    answer[k, l] += kernel[i, j] * subM[i, j, k, l]

    return answer

def makeDir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)


def hysteresis1d(data, low, high):
    data = np.asarray(data)
    # State aliases (for legibility)
    LOW_STATE = 0
    MID_STATE = 1
    HIGH_STATE = 2

    states = (data >= high).astype(np.uint8) + (data >= low).astype(np.uint8)
    # Transitions = changes in state taking place *on* data indices (excludes
    #   default LOW edge states)
    transitions = np.concatenate(([0], np.flatnonzero(np.diff(states)) + 1))
    transitionValues = states[transitions]

    # Regions = changes in state, inclusive of the default LOW edge states on
    #   the far left & right)
    regionValues = np.concatenate(([LOW_STATE], transitionValues, [LOW_STATE]))
    regionIsHigh = (regionValues == HIGH_STATE)

    midFilter = (transitionValues == MID_STATE)
    transitionValues[midFilter] = HIGH_STATE * (regionIsHigh[:-2][midFilter]
                                                | regionIsHigh[2:][midFilter])

    filteredData = np.empty_like(data, dtype=bool)
    # TODO find a way to circumvent using an explicit Python-implemented loop
    for i, j, val in itertools.izip_longest(transitions, transitions[1:],
                                            transitionValues):
        filteredData[i: j] = bool(val)

    return filteredData


def hysteresisNd(data, low_threshold, high_threshold):
    """ Consider data as a graph.  Each node has a value and a state.  All
        nodes are initially inactive.  Create a queue of each node with a
        value above high, which are known to be 'on'.  Those nodes are all marked
        as 'on' and added to a queue.  Iterate through each node in the queue.
        Process each adjacent node.  If that node has not been checked and it
        is in the middle of the hysteresis band, turn it 'on' and add it to
        the queue.  When the queue is empty, return the 'on' array.
    """
    data = np.asarray(data)

    def infinity_ball_coords(dims, dist):
        """ Generates integer coordinates for all points within a given
            distance under the L-infinity metric (see link below).

            https://en.wikipedia.org/wiki/Chebyshev_distance
        """
        return tuple(np.ravel(a) for a in np.broadcast_arrays(*(
            np.arange(-dist, dist + 1).reshape(-1, *(axis * [1])) #np.r_[-1, axis * [1]]
            for axis in range(dims)
        )))

    COORD_DELTAS = infinity_ball_coords(dims=data.ndim, dist=1)

    # Make binary maps for "low"s & "high"s
    bordered_shape = tuple(i + 2 for i in data.shape)
    highs = np.zeros(bordered_shape, dtype=np.bool)
    lows = np.ones(bordered_shape, dtype=np.bool)

    centerSlice = tuple(slice(1, -1) for dim in data.shape)
    lows[centerSlice] = (data < low_threshold)

    # Spreads "high" values to neighboring "mid" values
    coords = tuple(indices + 1 for indices in np.nonzero(data >= high_threshold))
    while highs[coords].size > 0:
        highs[coords] = True
        shiftedCoords = tuple((indices[:, np.newaxis] + delta).flatten()
                              for indices, delta in zip(coords, COORD_DELTAS))
        indicesFilter = ~(lows[shiftedCoords] | highs[shiftedCoords])
        coords = tuple(shiftedIndices[indicesFilter]
                       for shiftedIndices in shiftedCoords)
        coords = np.unravel_index(
            np.unique(np.ravel_multi_index(coords, highs.shape)),
            highs.shape
        )

    return highs[centerSlice]


def hysteresis(data, low, high, padding=None):
    """ Run a hysteresis filter on a 1d or 2d array. """
    if type(data) is not np.array:
        data = np.array(data, dtype='double')

    if data.ndim == 1:
        return hysteresis1d(data, low, high)
    elif data.ndim == 2:
        result = hysteresisNd(data, low, high)

        if padding is None:
            return result
        else:
            import scipy.signal as signal
            return signal.convolve(result, padding, mode='same') > 0

    else:
        raise RuntimeError


@nb.njit #This might not actually be faster since it's such a small amount of code
def gauss(xdata, sigma, mu, amp):
    """ A gaussian distribution function to fit with. """

    # Speed things up, hopefully, by defining all these ahead of time, locally
    var = sigma**2
    pi = np.pi
    const = amp #amp/np.sqrt(2*pi*var)

    return const*np.exp(((xdata-mu)**2)/(-2*var))


@nb.njit
def rotated_for_gabor(x, y, theta):
    return (
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta)
    )

@nb.njit
def gabor(width, height, lambdA, theta, psi, sigma, gamma):
    x = np.linspace(-1, 1, width).reshape(width, 1)
    y = np.linspace(-1, 1, height)

    xPrime, yPrime = rotated_for_gabor(x, y, -theta)

    return (
            np.exp(-(xPrime ** 2 + (gamma * yPrime) ** 2) / (2 * sigma ** 2))
            * np.cos(2 * np.pi * xPrime / lambdA + psi)
    )



@nb.njit
def compiled_convolve(raw_spectrogram, out_shape, noise_shape, size=10, nconv=4, gabor_mode=True):
    """ Convolve a kernel or set of kernels over the spectrogram and return
        the magnitude of the activation vector.
        @param [in] size the size (squared) of the kernel to use
        @param [in] nconv the number of convolutions to use
        @param [in] mode the type of kernel to use during convolution
                    'gabor' indicates a gabor filter """
    convolvedSignals = np.zeros(out_shape, dtype=np.float32)
    size = int(size)

    for i in xrange(0, nconv):
        angle = np.pi * (0.75 * i / nconv - 0.375)
        if gabor_mode:
            kernel = gabor(size, size, 0.75, angle, 0, 0.5, 0.5)
        convolvedSignals[:, :] += np.power(
            np.maximum(
                conv2d(
                    zero_pad(raw_spectrogram, np.array(kernel.shape), np.array(raw_spectrogram.shape)),
                    kernel),
                np.zeros(noise_shape)),
            2)

    return np.sqrt(convolvedSignals / nconv) * (10 ** 0.5)