import numpy as np


class Function:
    @staticmethod
    def function(*args, **kwargs):
        raise Exception('Abstract class Function called!')

    @staticmethod
    def partial_derivative_w(*args, **kwargs):
        raise Exception('Abstract class Function called!')

    @staticmethod
    def partial_derivative_b(*args, **kwargs):
        raise Exception('Abstract class Function called!')


class SigmoidLogisticRegression(Function):
    @staticmethod
    def function(y):
        return 1/(1 + np.exp(-y))

    @staticmethod
    def partial_derivative_w(input_x, input_y, y_hat, *args, **kwargs):
        error = input_y - y_hat
        return - error * (y_hat * (1 - y_hat)) * input_x

    @staticmethod
    def partial_derivative_b(input_y, y_hat):
        error = input_y - y_hat
        return - error * (y_hat * (1 - y_hat))


class RidgeRegression(Function):
    @staticmethod
    def function(y):
        return y

    @staticmethod
    def partial_derivative_w(input_x, input_y, y_hat, w, l2, *args, **kwargs):
        return - (2 * input_x * (input_y - y_hat)) + (2 * l2 * w)

    @staticmethod
    def partial_derivative_b(input_y, y_hat):
        return -2 * (input_y - y_hat)
