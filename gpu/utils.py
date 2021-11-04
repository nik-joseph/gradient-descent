import numpy as np
from tqdm import trange
from .functions import Function
from numba import cuda


def predict(x, w, b, function):
    """
    :param x: input row
    :param w: model weights
    :param b: bias
    :param function: an class of type Function having activation function and its derivative
    :return: prediction y hat
    """
    x_ar, w_ar = np.array(x, dtype=float), np.array(w, dtype=float)
    if x_ar.shape != w_ar.shape:
        raise Exception("Input and weight not same size in prediction")
    return function.function(b + function().dot(x_ar, w_ar))


def stochastic_coordinate(train_data_X, train_data_y,
                          learning_rate=None, epochs=None, function=Function, func_args=None):
    """
    :param train_data_X: input data attributes
    :param train_data_y: input data y hat
    :param learning_rate: learning rate
    :param epochs: number of iterations to train
    :param function: an class of type Function having activation function and its derivative
    :param func_args: holds an arguments required for weight and bias update
    :return: weight and bias of the model w, b
    """
    if func_args is None:
        func_args = {}

    # Initialize weight and bias to 0
    w, b = np.zeros(len(train_data_X[0])), 0

    ww = np.zeros(len(train_data_X[0]) + 1, dtype=np.float64)

    # Raise required errors
    if not learning_rate:
        raise Exception("Learning rate not set!")
    if not epochs:
        raise Exception("Number of epochs not set!")
    if not issubclass(function, Function):
        raise Exception("Given function not instance of Function!")

    l2 = func_args.pop('l2')
    learning_rate = learning_rate / (ww.shape[0])

    # Repeat learning rate and l2 to size of ww
    learning_rate_v2 = np.repeat(learning_rate, ww.shape)
    l2 = np.concatenate((np.repeat(l2, w.shape), np.array([0])))

    # Create XX from train data
    xx = np.c_[train_data_X, np.ones(train_data_X.shape[0])]

    # Copy data to gpu
    train_data_X_gpu = cuda.to_device(xx)
    train_data_y_gpu = cuda.to_device(train_data_y)
    learning_rate_gpu = cuda.to_device(learning_rate_v2)
    l2_gpu = cuda.to_device(l2)
    ww_gpu = cuda.to_device(ww)

    # Create grad array in gpu
    w_grad_gpu = cuda.to_device(np.zeros(ww.shape))

    # Start epoch
    for _ in trange(epochs):
        function().cuda_train(
            train_data_X_gpu, train_data_y_gpu, ww_gpu, learning_rate_gpu, l2_gpu, w_grad_gpu
        )
        cuda.synchronize()

    # Copy weight and bias to cpu
    ww = ww_gpu.copy_to_host()

    return ww[:-1], ww[-1]
