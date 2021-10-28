from numba import cuda
import numpy as np

from utils import repeat


@cuda.jit
def add_kernel(x, y, out):
    tx = cuda.threadIdx.x  # this is the unique thread ID within a 1D block
    ty = cuda.blockIdx.x  # Similarly, this is the unique block ID within the 1D grid

    block_size = cuda.blockDim.x  # number of threads per block
    grid_size = cuda.gridDim.x  # number of blocks in the grid

    start = tx + ty * block_size
    stride = block_size * grid_size

    # assuming x and y inputs are same length
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]


@cuda.jit
def add_kernel_simplified(x, y, out):
    start = cuda.grid(1)        # 1 = one dimensional thread grid, returns a single value
    stride = cuda.gridsize(1)   # ditto

    # assuming x and y inputs are same length
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]


def set_1():
    n = 100000
    x = np.arange(n).astype(np.float32)
    y = 2 * x
    out = np.empty_like(x)

    threads_per_block = 128
    blocks_per_grid = 30

    add_kernel_simplified[blocks_per_grid, threads_per_block](x, y, out)
    print(out[:10])

    print("I/O from CPU")
    repeat(lambda: add_kernel_simplified[blocks_per_grid, threads_per_block](x, y, out))

    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    out_device = cuda.device_array_like(x)
    print("I/O from GPU")
    repeat(lambda: add_kernel_simplified[blocks_per_grid, threads_per_block](x_device, y_device, out_device))

    # CPU input/output arrays, implied synchronization for memory copies
    print("implied synchronization")
    repeat(lambda: add_kernel[blocks_per_grid, threads_per_block](x, y, out), count=1)

    # GPU input/output arrays, no synchronization (but force sync before and after)
    cuda.synchronize()
    print("no synchronization (but force sync before and after)")
    repeat(lambda: add_kernel[blocks_per_grid, threads_per_block](x_device, y_device, out_device), count=1)
    cuda.synchronize()

    # GPU input/output arrays, include explicit synchronization in timing
    def func_wth_sync():
        add_kernel[blocks_per_grid, threads_per_block](x_device, y_device, out_device)
        cuda.synchronize()

    print("include explicit synchronization in timing")
    repeat(func_wth_sync, count=1)


# set_1()


@cuda.jit
def thread_counter_race_condition(global_counter):
    global_counter[0] += 1  # This is bad


@cuda.jit
def thread_counter_safe(global_counter):
    cuda.atomic.add(global_counter, 0, 1)  # Safely add 1 to offset 0 in global_counter array


def set_2():
    # This gets the wrong answer
    global_counter = cuda.to_device(np.array([0], dtype=np.int32))
    thread_counter_race_condition[64, 64](global_counter)

    print('Should be %d:' % (64 * 64), global_counter.copy_to_host())

    # This works correctly
    global_counter = cuda.to_device(np.array([0], dtype=np.int32))
    thread_counter_safe[64, 64](global_counter)

    print('Should be %d:' % (64*64), global_counter.copy_to_host())


# set_2()

# Exercise
# For this exercise, create a histogram kernel.
# This will take an array of input data, a range and a number of bins,
# and count how many of the input data elements land in each bin.
# Below is an example CPU implementation of histogram:


def cpu_histogram(x, x_min, x_max, histogram_out):
    """Increment bin counts in histogram_out, given histogram range [x_min, x_max)."""
    # Note that we don't have to pass in n_bins explicitly, because the size of histogram_out determines it
    n_bins = histogram_out.shape[0]
    bin_width = (x_max - x_min) / n_bins

    # This is a very slow way to do this with NumPy, but looks similar to what you will do on the GPU
    for element in x:
        bin_number = np.int32((element - x_min) / bin_width)
        if 0 <= bin_number < histogram_out.shape[0]:
            # only increment if in range
            histogram_out[bin_number] += 1


@cuda.jit
def cuda_histogram(x, x_min, x_max, histogram_out):
    """Increment bin counts in histogram_out, given histogram range [x_min, x_max)."""
    # Cuda stuff
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n_bins = histogram_out.shape[0]
    bin_width = (x_max - x_min) / n_bins

    for i in range(start, x.shape[0], stride):
        bin_number = np.int32((x[i] - x_min) / bin_width)
        if 0 <= bin_number < histogram_out.shape[0]:
            # only increment if in range
            cuda.atomic.add(histogram_out, bin_number, 1)


def set_3():
    x = np.random.normal(size=10000, loc=0, scale=1).astype(np.float32)
    x_min = np.float32(-4.0)
    x_max = np.float32(4.0)
    histogram_out = np.zeros(shape=10, dtype=np.int32)
    histogram_out_gpu_device = cuda.to_device(np.zeros(shape=10, dtype=np.int32))

    print("CPU Histogram\n")
    repeat(lambda: cpu_histogram(x, x_min, x_max, histogram_out), count=100)

    threads_per_block = 128
    blocks_per_grid = 30
    print("GPU Histogram\n")
    repeat(lambda: cuda_histogram[blocks_per_grid, threads_per_block](x, x_min, x_max, histogram_out_gpu_device), count=100)

    # Just to see if what am doing is right
    histogram_out_gpu = histogram_out_gpu_device.copy_to_host()
    assert all(histogram_out == histogram_out_gpu)


# set_3()
