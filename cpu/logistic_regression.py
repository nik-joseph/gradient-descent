import numpy as np

from .utils import stochastic_gradient, coordinate_gradient, predict


class LogisticRegression:
    def __init__(self, function='gradient'):
        self.w = None
        self.b = None
        self.function = function
        self.functions = {
            'gradient': stochastic_gradient,
            'coordinate': coordinate_gradient
        }

    def fit(self, train_data_X, train_data_y, learning_rate, epochs):
        gradient_function = self.functions.get(self.function, stochastic_gradient)
        self.w, self.b = gradient_function(train_data_X, train_data_y, learning_rate, epochs)
        return self

    def predict_single(self, x):
        return round(predict(x, self.w, self.b))

    def predict(self, X_test):
        if X_test.ndim != 2:
            raise Exception('Number of dimensions not 2!')

        return np.array([
            self.predict_single(row)
            for row in X_test
        ], dtype=float)

    def score(self, X_test, y_test):
        y_hat = self.predict(X_test)
        return (1 - (y_hat - y_test) ** 2).mean()
