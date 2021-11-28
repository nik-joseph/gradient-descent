import numpy as np
import math
from numba import cuda
from materials.utils import repeat

THREADS_PER_BLOCK = 16, 16       # threads per block


@cuda.jit
def kernel_product(mat_a, mat_b, out):
    x, y = cuda.grid(2)
    if x < out.shape[0] and y < out.shape[1]:
        partial_product = 0
        for k in range(mat_a.shape[1]):
            partial_product += mat_a[x, k] * mat_b[k, y]
        out[x, y] = partial_product


def matmul(mat_a, mat_b):
    return np.array([
        [
            sum([
                mat_a[x, k] * mat_b[k, y]
                for k in range(mat_a.shape[1])
            ])
            for y in range(mat_b.shape[1])
        ]
        for x in range(mat_a.shape[0])
    ])


def matrix_product(mat_a, mat_b, no_cuda=False):
    if mat_a.shape[1] != mat_b.shape[0]:
        raise Exception("Matrices have invalid dimensions to multiply")
    if no_cuda:
        return repeat(lambda: matmul(mat_a, mat_b))

    a_device = cuda.to_device(mat_a)
    b_device = cuda.to_device(mat_b)
    out_device = cuda.device_array(shape=(mat_a.shape[0], mat_b.shape[1]), dtype=np.float32)

    blocks_per_grid_x = math.ceil(out_device.shape[0] / THREADS_PER_BLOCK[0])
    blocks_per_grid_y = math.ceil(out_device.shape[1] / THREADS_PER_BLOCK[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    repeat(lambda: kernel_product[blocks_per_grid, THREADS_PER_BLOCK](a_device, b_device, out_device))

    return out_device


def gpu_matrix_product(mat_a, mat_b):
    a_device = cuda.to_device(mat_a)
    b_device = cuda.to_device(mat_b)
    out_device = cuda.device_array(shape=(mat_a.shape[0], mat_b.shape[1]), dtype=np.float32)

    kernel_product[(48, 48), THREADS_PER_BLOCK](a_device, b_device, out_device)

    cuda.synchronize()

    return out_device.copy_to_host()
