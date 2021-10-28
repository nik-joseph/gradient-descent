import numpy as np

from .utils import stochastic_gradient, coordinate_gradient, predict
from .functions import RidgeRegression as ActivationFunction


class RidgeRegression:
    def __init__(self, function='gradient', l2=0.01):
        self.w = None
        self.b = None
        self.function = function
        self.functions = {
            'gradient': stochastic_gradient,
            'coordinate': coordinate_gradient
        }
        self.activation_function = ActivationFunction
        self.l2 = l2

    def fit(self, train_data_X, train_data_y, learning_rate, epochs):
        gradient_function = self.functions.get(self.function, stochastic_gradient)
        self.w, self.b = gradient_function(
            train_data_X, train_data_y, learning_rate, epochs, function=self.activation_function,
            func_args={'l2': self.l2}
        )
        return self

    def predict(self, X_test):
        if X_test.ndim != 2:
            raise Exception('Number of dimensions not 2!')

        return np.array([
            round(predict(x, self.w, self.b, function=self.activation_function))
            for x in X_test
        ], dtype=float)

    def score(self, X_test, y_test):
        y_hat = self.predict(X_test)
        return (1 - (y_hat - y_test) ** 2).mean()
