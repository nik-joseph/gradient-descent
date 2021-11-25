import datetime

import numpy as np
from tqdm import trange, tqdm


class Model:
    def __init__(self, batch_size, benchmark):
        self.train_y = []
        self.train_X = []
        self.batch_size = batch_size
        self.epochs = None
        self.benchmark = benchmark
        self.benchmarks = {
            'time': [],
            'weights': [],
        }
        self.timer = None
        self.w, self.b = None, None

    def fit(self, train_X, train_y, epochs, *args, **kwargs):
        self.train_X, self.train_y = train_X, train_y
        self.epochs = epochs
        self.timer = datetime.datetime.now()

        # Initialize class with required variables
        self.__initialize_variables__(*args, **kwargs)

        for _ in trange(self.epochs, desc='Epochs'):
            self.__checkpoint__()
            for X, y in self.__get_data__():
                self.__fit__(X, y, *args, **kwargs)
            if self.benchmark:
                self.__benchmark__((self.w.copy(), self.b.copy()))

        return self

    def __initialize_variables__(self, *args, **kwargs):
        pass

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

    def __get_benchmark_results__(self, weights, X, y):
        raise Exception("Abstract __get_benchmark_results__ called!")

    def __checkpoint__(self):
        current_time = datetime.datetime.now()
        time_taken = (current_time - self.timer).total_seconds()
        return time_taken

    def __benchmark__(self, weights):
        time_taken = self.__checkpoint__()
        self.benchmarks['time'].append(time_taken)
        self.benchmarks['weights'].append(weights)

    def get_benchmark_results(self, X, y):
        print("\nRunning Benchmarks")

        self.__get_benchmark_results__(self.benchmarks['weights'][-1], X, y)
        data = [
            (time, self.__get_benchmark_results__(weights, X, y))
            for time, weights in tqdm(zip(self.benchmarks['time'], self.benchmarks['weights']), total=self.epochs)
        ]
        return data
