import numpy as np

def pad_with_zeros(X, pad):
    """Applies padding to height and width with zeros.

    :param X: input matrix
    :type X: np.array, (num of images, height, width, num of channels)
    :param pad: amount of padding
    :type pad: int

    :return: padding aplied X, X_padded
    :rtype: np.array
    """

    X_padded = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))

    return X_padded


def convolution_step(prev_a_slice, W, b):
    """Applies one step of convolution with W and b parameters to a slice.

    :param prev_a_slice: one slice of a from previous layer
    :type prev_a_slice: np.array(filter size, filter size, num of previous channels)
    :param W: weight parameter
    :type W: np.array(filter size, filter size, num of previous channels)
    :param b: bias parameter
    :type b: np.array(1, 1, 1)
    """

    Z_lesser = np.sum(np.multiply(prev_a_slice, W))
    
    b = np.squeeze(b)
    Z = Z_lesser + b

    return Z


def forward_convolution(prev_A, W, b, hyperparam):
    """Forwardpropagates with convolution

    :param prev_A: input data from previous layer
    :type prev_A: np.array(num of examples, height, width, num of channels)
    :param W: weight matrix
    :type W: np.array(filter size, filter size, num of prev channels, num of channels)
    :param b: bias vector
    :type b: (1, 1, 1, num of channels)
    :param hyperparam: dictionary containing padding and stride levels
    :type hyperparam: dictionary

    :return: output of convolution Z, cache containing prev_A, W, b, hyperparam
    :rtype: np.array, tuple
    """

    (m, n_height_prev, n_width_prev, C_prev) = prev_A.shape
    (f, f, C_prev, C) = W.shape

    stride = hyperparam["stride"]
    padding = hyperparam["padding"]

    height = int((n_height_prev + 2*padding - f)/stride) + 1
    width = int((n_width_prev + 2*padding - f)/stride) + 1

    Z = np.zeros((m, height, width, C))

    prev_A_padded = pad_with_zeros(prev_A, padding)

    for i in range(m):

        prev_a_padded = prev_A_padded[i]

        for h in range(height):

            vertical_first = stride*h
            vertical_last = vertical_first + f

            for w in range(width):

                horizontal_first = stride*w
                horizontal_last = horizontal_first + f

                for c in range(C):

                    prev_a_slice = prev_a_padded[vertical_first:vertical_last, horizontal_first:horizontal_last, :]
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = convolution_step(prev_a_slice, weights, biases)

    cache = (prev_A, W, b, hyperparam)

    return Z, cache


def forward_pooling(prev_A, hyperparam, mode):
    """Pool your inputs forwardly

    :param prev_A: input data from previous layer
    :type prev_A: np.array(num of examples, height, width, num of channels)
    :param hyperparam: dictionary containing stride and padding hyperparameters
    :type hyperparam: dictionary
    :param mode: pooling modes of max and average
    :type mode: string

    :return: pooled A and cache containing input and hyperparameters
    :rtype: np.array, tuple
    """

    (m, n_height_prev, n_width_prev, C_prev) = prev_A.shape

    f = hyperparam["f"]
    stride = hyperparam["stride"]

    height = int((n_height_prev - f)/stride) + 1
    width = int((n_width_prev - f)/stride) + 1
    C = C_prev

    A = np.zeros((m, height, width, C))

    for i in range(m):

        prev_a_slice = prev_A[i]

        for h in range(height):

            vertical_first = stride * h
            vertical_last = vertical_first + f

            for w in range(width):

                horizontal_first = stride * w
                horizontal_last = horizontal_first + f

                for c in range(C):

                    prev_a_slice_part = prev_a_slice[vertical_first:vertical_last, horizontal_first:horizontal_last, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(prev_a_slice_part)

                    elif mode == "average":
                        A[i, h, w, c] = np.mean(prev_a_slice_part)

    
    cache = (prev_A, hyperparam)
    assert(A.shape == (m, height, width, C))

    return A, cache


def backward_convolution(dZ, cache):
    """Backpropagation with convolution

    :param dZ: gradient of the pre-activation parameter
    :type dZ: np.array(m, height, width, num of channels)
    :param cache: dictionary containing linear cache

    :return: gradient of activated parameters of previous layer, weights and biases
    :rtype: np.array
    """

    (prev_A, W, b, hyperparam) = cache
    (m, n_height_prev, n_width_prev, C_prev) = prev_A.shape
    (f, f, C_prev, C) = W.shape

    stride = hyperparam["stride"]
    padding = hyperparam["padding"]

    (m, height, width, C) = dZ.shape

    dprev_A = np.zeros(prev_A.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    prev_A_padded = pad_with_zeros(prev_A, padding)
    dprev_A_padded = pad_with_zeros(dprev_A, padding)

    for i in range(m):

        prev_a_padded = prev_A_padded[i]
        dprev_a_padded = dprev_A_padded[i]

        for h in range(height):

            for w in range(width):

                for c in range(C):

                    vertical_first = stride*h
                    vertical_end = vertical_first + f
                    horizontal_first = stride*w
                    horizontal_last = horizontal_first + f

                    a_slice = prev_a_padded[vertical_first:vertical_end, horizontal_first:horizontal_last, :]
                    dprev_a_padded[vertical_first:vertical_end, horizontal_first:horizontal_last, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice*dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    
    dprev_A[i, :, :, :] = dprev_a_padded[padding : -padding, padding : -padding, :]
    assert(dprev_A.shape == (m, n_height_prev, n_width_prev, C_prev))

    return dprev_A, dW, db


def masking(X):
    """masks to identify max of X, it is for backpass of max pooling

    :param X: input data
    :type X: np.array(a, a)

    :return: mask containing True at the max position
    :rtype: np.array
    """

    mask = (X == np.max(X))

    return mask


def value_distribution(dZ, shape):
    """distributes values according to shape, it is for average pooling

    :param dZ: derivative of pre-activateed parameters
    :type dZ: np.array
    :param shape: shape of array
    :type shape: np.array

    :return: a, array
    :rtype: np.array(height, width)
    """

    (height, width) = shape
    averages = np.prod(shape)

    a = (dZ/averages) * np.ones(shape)

    return a


def backward_pooling(dA, cache, mode):
    """Backpropagation of pooling

    :param dA: gradient of activated parameter
    :type dA: np.array
    :param cache: cache containing input and parameters
    :type cache: tuple
    :param mode: pooling type of max or average
    :type mode: string

    :return: gradient of the previous activated parameter
    :rtype: np.array
    """

    (prev_A, hyperparameter) = cache

    stride = hyperparameter["stride"]
    f = hyperparameter["f"]

    m, n_height_prev, n_width_prev, C_prev = prev_A.shape
    m, height, width, C = dA.shape

    dprev_A = np.zeros(prev_A.shape)

    for i in range(m):

        prev_a = prev_A[i, :, :, :]

        for h in range(height):
            for w in range(width):
                for c in range(C):

                    vertical_first = stride*h
                    vertical_last = vertical_first + f
                    horizontal_first = stride * w
                    horizontal_last = horizontal_first + f

                    if mode == "max":

                        prev_a_slice = prev_a[vertical_first:vertical_last, horizontal_first:horizontal_last, c]
                        mask = masking(prev_a_slice)
                        dprev_A[i, vertical_first:vertical_last, horizontal_first:horizontal_last, c] += mask * dA[i, h, w, c]
                    
                    elif mode == "average":

                        da = dA[i, h, w, c]
                        shape = (f,f)
                        dprev_A[i, vertical_first:vertical_last, horizontal_first:horizontal_last, c] += value_distribution(da, shape)

    assert(dprev_A.shape == prev_A.shape)

    return dprev_A

