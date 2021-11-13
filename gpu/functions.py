import numpy as np
from numba import cuda
from .cuda_functions import THREADS_PER_BLOCK, MIN_BLOCKS, create_xoroshiro128p_states
from .cuda_functions import cuda_compute_weights, cuda_dot, dot, cuda_subtract, cuda_zeros


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

    @staticmethod
    def dot(x, w):
        # Create output variable
        out = np.zeros([1])

        # Copy all to gpu
        x_gpu = cuda.to_device(x)
        w_gpu = cuda.to_device(w)
        out_gpu = cuda.to_device(out)

        # Compute blocks per grid
        blocks_per_grid = int(np.ceil(x.shape[0] / THREADS_PER_BLOCK)) + MIN_BLOCKS

        # Compute dot product
        cuda_dot[blocks_per_grid, THREADS_PER_BLOCK](x_gpu, w_gpu, out_gpu)

        # Wait for sync
        cuda.synchronize()

        # Copy output back to cpu and return
        return out_gpu.copy_to_host()[0]

    @staticmethod
    def cuda_train_v2(*args, **kwargs):
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

    def cuda_train(self, X, y, ww, learning_rate, l2, w_grad_gpu, batch=None, og_X=None, og_y=None):

        if batch is not None:
            X = cuda.to_device(og_X[batch])
            y = cuda.to_device(og_y[batch])

        # Compute y_hat and store to gpu
        y_hat_gpu = cuda_dot_vector(X, ww)

        # Set grad to zero
        cuda_zeros[int(np.ceil(ww.shape[0]/THREADS_PER_BLOCK) + MIN_BLOCKS), THREADS_PER_BLOCK](w_grad_gpu)

        # Wait for sync
        cuda.synchronize()

        # Create weight rng states
        w_rng = create_xoroshiro128p_states(
            int(np.ceil(X.shape[0] / THREADS_PER_BLOCK) + MIN_BLOCKS) * THREADS_PER_BLOCK, seed=np.random.random())

        # Wait for sync
        cuda.synchronize()

        # Compute weight grads
        cuda_compute_weights[int(np.ceil(X.shape[0] / THREADS_PER_BLOCK) + MIN_BLOCKS), THREADS_PER_BLOCK](
            X, y, y_hat_gpu, learning_rate, l2, ww, w_grad_gpu, w_rng
        )

        # Wait for sync
        cuda.synchronize()

        # Subtract grads from weight
        cuda_subtract[int(np.ceil(ww.shape[0]/THREADS_PER_BLOCK) + MIN_BLOCKS), THREADS_PER_BLOCK](ww, w_grad_gpu)

        # Wait for sync
        cuda.synchronize()

        return ww


def cuda_dot_vector(x, w):
    out = np.zeros(x.shape[0])
    out_gpu = cuda.to_device(out)

    # Compute blocks per grid
    blocks_per_grid_x = int(np.ceil(x.shape[0] + MIN_BLOCKS / THREADS_PER_BLOCK))
    blocks_per_grid_y = int(np.ceil(x.shape[1] + MIN_BLOCKS / THREADS_PER_BLOCK))

    # Compute dot product
    dot[(blocks_per_grid_x, blocks_per_grid_y), (THREADS_PER_BLOCK, THREADS_PER_BLOCK)](x, w, out_gpu)

    # Wait for sync
    cuda.synchronize()

    # return gpu stored dot product
    return out_gpu
