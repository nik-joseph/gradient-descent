import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver
from tqdm import trange

from models.base import Model
from gpu.cuda_functions import dot, cuda_subtract, cuda_compute_weights_v2, cuda_add


class RidgeRegression(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = None
        self.b = None

    def __get_data__(self):
        indices = list(np.random.randint(self.train_X.shape[0])for _ in range(self.batch_size))
        return zip(self.train_X[indices], self.train_y[indices])

    def __fit__(self, X, y, learning_rate, l2, *args, **kwargs):
        # Initialize weight and bias to 0 if both w and b are None
        if not self.initial_variables_set:
            self.w, self.b = np.zeros(len(self.train_X[0])), 0
            self.initial_variables_set = True

        # Compute prediction
        y_hat = self.predict(X)

        # Get random attribute to descent
        index = np.random.randint(0, self.w.shape[0] + 1)

        if index < self.w.shape[0]:
            # Update weight on learning rate and calculate error for current weight based on index
            delta_w = - (2 * X[index] * (y - y_hat)) + (2 * l2 * self.w[index])
            self.w[index] = self.w[index] - learning_rate * delta_w
        else:
            delta_b = -2 * (y - y_hat)
            self.b = self.b - learning_rate * delta_b

    def __predict__(self, X):
        if X.shape != self.w.shape:
            raise Exception("Input and weight not same size in prediction")
        return self.b + np.dot(X, self.w.transpose())


class GPURidgeRegression(RidgeRegression):
    THREADS_PER_BLOCK = 1
    MIN_BLOCKS = 96

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ww = None
        self.epochs = None
        self.current_epoch = None

    def fit(self, train_X, train_y, epochs, *args, **kwargs):
        self.train_X, self.train_y = train_X, train_y
        self.epochs = epochs

        # Initialize class with required variables
        self.__initialize_variables__(*args, **kwargs)

        for _ in trange(epochs, desc='Epochs'):
            self.__fit__(*args, **kwargs)

        self.ww = self.ww_gpu.copy_to_host()
        cuda.synchronize()

        self.w, self.b = self.ww[:-1], self.ww[-1]
        return self

    def __get_data_batch__(self):
        indices = list(np.random.randint(self.train_X.shape[0]) for _ in range(self.batch_size))

        # Create XX from train data and GPU arrays
        self.train_XX = np.c_[self.train_X, np.ones(self.train_X.shape[0])]
        self.og_train_XX_gpu = cuda.to_device(self.train_XX)
        self.og_train_y_gpu = cuda.to_device(self.train_y)

        return self.train_XX[indices], self.train_y[indices]

    def __get_data__(self):
        return zip(*self.__get_data_batch__())

    def __initialize_variables__(self, learning_rate, l2, *args, **kwargs):
        # Moved here for isolation
        self.__get_data_batch__()

        # Initialize weight and bias to 0
        self.ww = np.zeros(self.train_XX.shape[1], dtype=np.float64)

        # Repeat learning rate and l2 to size of ww
        self.learning_rate_rep = np.repeat(learning_rate, self.ww.shape)
        self.l2 = np.concatenate((np.repeat(l2, self.ww.shape), np.array([0])))

        # Copy data to gpu
        self.learning_rate_gpu = cuda.to_device(self.learning_rate_rep)
        self.l2_gpu = cuda.to_device(self.l2)
        self.ww_gpu = cuda.to_device(self.ww)

        # Create gpu array to store y_hat
        self.y_hat_gpu = cuda.to_device(np.zeros(self.train_y.shape[0]))

        # Create grad array in gpu
        self.w_grad_gpu = cuda.to_device(np.zeros(self.ww.shape))

        # Create weight rng states for complete epochs
        self.w_rng_array = cuda.to_device(
            np.random.randint(self.ww_gpu.shape[0], size=(self.epochs, self.train_XX.shape[0])))

        # Create train data rng states for complete epochs
        self.data_rng_array = cuda.to_device(
            np.random.randint(self.train_XX.shape[0], size=(self.epochs, self.batch_size))
        )

        # Create epoch gpu counter
        self.current_epoch_gpu = cuda.to_device(np.zeros((1,), dtype=int))

    def __fit__(self, learning_rate, l2, *args, **kwargs):
        # Update current epoch gpu counter
        cuda_add[1 + self.MIN_BLOCKS, 1](self.current_epoch_gpu)

        # Wait for sync
        cuda.synchronize()

        # Compute y_hat and store to gpu
        self.__cuda_dot_vector()

        # Set grad to zero
        driver.device_memset(self.w_grad_gpu, 0, self.w_grad_gpu.size * 8)

        # Wait for sync
        cuda.synchronize()

        # Wait for sync
        cuda.synchronize()

        # Compute weight grads
        cuda_compute_weights_v2[
            int(
                np.ceil(self.train_XX.shape[0] / self.THREADS_PER_BLOCK) + self.MIN_BLOCKS
            ), self.THREADS_PER_BLOCK
        ](self.og_train_XX_gpu, self.og_train_y_gpu, self.y_hat_gpu,
          self.learning_rate_gpu, self.l2_gpu, self.ww_gpu, self.w_grad_gpu,
          self.w_rng_array, self.data_rng_array, self.current_epoch_gpu)

        # Wait for sync
        cuda.synchronize()

        # Subtract grads from weight
        cuda_subtract[
            int(np.ceil(self.ww_gpu.shape[0] / self.THREADS_PER_BLOCK) + self.MIN_BLOCKS), self.THREADS_PER_BLOCK
        ](self.ww_gpu, self.w_grad_gpu)

        # Wait for sync
        cuda.synchronize()

    def __cuda_dot_vector(self):
        # Empty out_gpu
        driver.device_memset(self.y_hat_gpu, 0, self.y_hat_gpu.size * 8)

        # Wait for sync
        cuda.synchronize()

        # Compute blocks per grid
        blocks_per_grid_x = int(np.ceil(self.train_XX.shape[0] + self.MIN_BLOCKS / self.THREADS_PER_BLOCK))
        blocks_per_grid_y = int(np.ceil(self.train_XX.shape[1] + self.MIN_BLOCKS / self.THREADS_PER_BLOCK))

        # Compute dot product
        dot[
            (blocks_per_grid_x, blocks_per_grid_y), (self.THREADS_PER_BLOCK, self.THREADS_PER_BLOCK)
        ](self.og_train_XX_gpu, self.ww_gpu, self.y_hat_gpu)

        # Wait for sync
        cuda.synchronize()
