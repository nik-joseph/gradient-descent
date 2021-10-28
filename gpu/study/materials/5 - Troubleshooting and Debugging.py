import numpy as np

from numba import cuda
from pdb import set_trace

# Kinda skipped this class, too boring and didn't understand


@cuda.jit
def histogram(x, x_min, x_max, histogram_out):
    n_bins = histogram_out.shape[0]
    bin_width = (x_max - x_min) / n_bins

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Could not get this to work
    if start == 0:
        set_trace()

    for i in range(start, x.shape[0], stride):
        bin_number = np.int32((x[i] - x_min) / bin_width)
        if 0 <= bin_number < histogram_out.shape[0]:
            cuda.atomic.add(histogram_out, bin_number, 1)


def set_1():
    x = np.random.normal(size=50, loc=0, scale=1).astype(np.float32)
    x_min = np.float32(-4.0)
    x_max = np.float32(4.0)
    histogram_out = np.zeros(shape=10, dtype=np.int32)

    histogram[64, 64](x, x_min, x_max, histogram_out)

    print('input count:', x.shape[0])
    print('histogram:', histogram_out)
    print('count:', histogram_out.sum())


set_1()
