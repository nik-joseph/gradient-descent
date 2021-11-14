from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_next

THREADS_PER_BLOCK = 1

MIN_BLOCKS = 96


@cuda.jit()
def cuda_compute_weights(X, y, y_hat, learning_rate, l2, w, w_grad, w_rng):
    i = cuda.grid(1)
    if i < w.shape[0]:
        w_index = w_rng[i]

        # Compute partial w_grad
        partial_w_grad = - learning_rate[w_index] * (
                (2 * X[i][w_index] * (y[i] - y_hat[i])) + (2 * l2[w_index] * w[w_index])
        )

        # Update w_grad
        cuda.atomic.add(
            w_grad,
            w_index,
            partial_w_grad
        )


@cuda.jit()
def cuda_compute_weights_v2(X, y, y_hat, learning_rate, l2, w, w_grad, w_rng, data_rng, current_epoch):
    i = cuda.grid(1)
    if i < w.shape[0]:
        w_index = w_rng[current_epoch[0]][i]
        data_index = data_rng[current_epoch[0]][i]

        # Compute partial w_grad
        partial_w_grad = - learning_rate[w_index] * (
                (2 * X[data_index][w_index] * (y[data_index] - y_hat[data_index])) + (2 * l2[w_index] * w[w_index])
        )

        # Update w_grad
        cuda.atomic.add(
            w_grad,
            w_index,
            partial_w_grad
        )


@cuda.jit
def cuda_dot(x, w, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        cuda.atomic.add(out, 0, x[i] * w[i])


@cuda.jit
def dot(x_, w_, out_):
    i, j = cuda.grid(2)
    if i < x_.shape[0] and j < x_.shape[1]:
        cuda.atomic.add(out_, i, x_[i][j] * w_[j])


@cuda.jit
def cuda_subtract(w, w_grad):
    i = cuda.grid(1)
    if i < w.shape[0]:
        cuda.atomic.sub(w, i, w_grad[i])


@cuda.jit
def cuda_zeros(x):
    i = cuda.grid(1)
    if i < x.shape[0]:
        x[i] = 0


@cuda.jit
def cuda_add(x):
    i = cuda.grid(1)
    if i < x.shape[0]:
        cuda.atomic.add(x, 0, 1)
