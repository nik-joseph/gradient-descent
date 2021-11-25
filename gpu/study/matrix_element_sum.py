from numba import cuda
from materials.utils import repeat
import math

THREADS_PER_BLOCK = 16, 16     # threads per block


@cuda.jit
def kernel_sum(mat_a, mat_b, out):
    x, y = cuda.grid(2)
    if x < out.shape[0] and y < out.shape[1]:
        out[x, y] = mat_a[x, y] + mat_b[x, y]


def element_wise_sum(mat_a, mat_b, no_cuda=False):
    if mat_a.shape != mat_b.shape:
        raise Exception("Matrices are of different dimensions")
    if no_cuda:
        return repeat(lambda: mat_a + mat_b)

    a_device = cuda.to_device(mat_a)
    b_device = cuda.to_device(mat_b)
    out_device = cuda.device_array_like(mat_a)

    blocks_per_grid_x = math.ceil(out_device.shape[0] / THREADS_PER_BLOCK[0])
    blocks_per_grid_y = math.ceil(out_device.shape[1] / THREADS_PER_BLOCK[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    repeat(lambda: kernel_sum[blocks_per_grid, THREADS_PER_BLOCK](a_device, b_device, out_device))

    return out_device.copy_to_host()


def gpu_matrix_sum(mat_a, mat_b):
    a_device = cuda.to_device(mat_a)
    b_device = cuda.to_device(mat_b)
    out_device = cuda.device_array_like(mat_a)

    blocks_per_grid_x = math.ceil(out_device.shape[0] / THREADS_PER_BLOCK[0])
    blocks_per_grid_y = math.ceil(out_device.shape[1] / THREADS_PER_BLOCK[1])
    blocks_per_grid = (blocks_per_grid_x + 48, blocks_per_grid_y + 48)

    kernel_sum[blocks_per_grid, THREADS_PER_BLOCK](a_device, b_device, out_device)

    cuda.synchronize()
    return out_device
