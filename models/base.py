import numpy as np
from tqdm import trange


class Model:
    def __init__(self, batch_size):
        self.train_y = []
        self.train_X = []
        self.batch_size = batch_size
        self.initial_variables_set = False

    def fit(self, train_X, train_y, epochs, *args, **kwargs):
        self.initial_variables_set = False
        self.train_X, self.train_y = train_X, train_y
        for _ in trange(epochs, desc='Epochs'):
            for X, y in self.__get_data__():
                self.__fit__(X, y, *args, **kwargs)

        return self

    def predict(self, X):
        if (dim := X.ndim) == 2:
            return np.array([
                self.__predict__(single_x)
                for single_x in X
            ], dtype=float)

        elif dim == 1:
            return self.__predict__(X)

        raise Exception(f"Invalid number of dimensions{dim}!")

    def score(self, X_test, y_test):
        y_hat = self.predict(X_test)
        return (1 - (y_hat - y_test) ** 2).mean()

    def __get_data__(self):
        raise Exception("Abstract __get_data__ called!")

    def __fit__(self, *args, **kwargs):
        raise Exception("Abstract __fit__ called!")

    def __predict__(self, x):
        raise Exception("Abstract __predict__ called!")
