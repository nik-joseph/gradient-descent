import numpy as np
from numba import cuda, guvectorize, vectorize
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_next

THREADS_PER_BLOCK = 1


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

    @staticmethod
    def cuda_train(*args, **kwargs):
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

    @staticmethod
    def cuda_train(X, y, w, b, learning_rate, l2, *args, **kwargs):
        y_hat = np.array([np.dot(single_x, w.transpose()) for single_x in X], dtype=np.float32)

        # Copy to GPU
        X_gpu = cuda.to_device(X)
        y_gpu = cuda.to_device(y)
        y_hat_gpu = cuda.to_device(y_hat)
        learning_rate_gpu = cuda.to_device([learning_rate])
        l2_gpu = cuda.to_device([l2])
        w_gpu = cuda.to_device(w)
        w_grad_gpu = cuda.to_device(np.zeros(w.shape))

        # Compute blocks per grid
        blocks_per_grid = int(X.shape[0] / THREADS_PER_BLOCK)

        # Create weight rng states
        w_rng = create_xoroshiro128p_states(blocks_per_grid * THREADS_PER_BLOCK, seed=np.random.random())

        # Compute weights
        cuda_compute_weights[blocks_per_grid, THREADS_PER_BLOCK](
            X_gpu, y_gpu, y_hat_gpu, learning_rate_gpu, l2_gpu, w_gpu, w_grad_gpu, w_rng
        )

        # Wait for sync
        cuda.synchronize()

        # Copy updated weights to cpu weights
        w_grad = w_grad_gpu.copy_to_host()

        w = w - w_grad
        return w, b


@cuda.jit()
def cuda_compute_weights(X, y, y_hat, learning_rate, l2, w, w_grad, w_rng):
    i = cuda.grid(1)
    if i < w.shape[0]:
        w_index = xoroshiro128p_next(w_rng, i) % w.shape[0]

        # Compute partial w_grad
        partial_w_grad = - learning_rate[0] * ((2 * X[i][w_index] * (y[i] - y_hat[i])) + (2 * l2[0] * w[w_index]))

        # Update w_grad
        cuda.atomic.add(
            w_grad,
            w_index,
            partial_w_grad
        )
