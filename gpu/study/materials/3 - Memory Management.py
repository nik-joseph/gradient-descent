import math
import numpy as np
from matplotlib import pyplot as plt
from numba import vectorize, cuda

from utils import repeat


@vectorize(['float32(float32, float32)'], target='cuda')
def add_func(x, y):
    return x + y


def set_1():
    n = 100000
    x = np.arange(n).astype(np.float32)
    y = 2 * x

    print("GPU input data from CPU")
    repeat(lambda: add_func(x, y))

    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)

    # print(x_device)
    # print(x_device.shape)
    # print(x_device.dtype)

    print("GPU input data from GPU")
    repeat(lambda: add_func(x_device, y_device))

    out_device = cuda.device_array(shape=(n,), dtype=np.float32)  # does not initialize the contents, like np.empty()

    # Must be cause of what we are running but didn't see much improvement here
    print("GPU input data from GPU, output in GPU")
    repeat(lambda: add_func(x_device, y_device, out=out_device))

    out_host = out_device.copy_to_host()
    print(out_host[:10])


# set_1()

# Exercise
@vectorize(['float32(float32, float32, float32)'], target='cuda')
def make_pulses(i, period, amplitude):
    return max(math.sin(i / period) - 0.3, 0.0) * amplitude


def set_2():
    n = 100000
    noise = (np.random.normal(size=n) * 3).astype(np.float32)
    t = np.arange(n, dtype=np.float32)
    period = n / 23

    pulses = make_pulses(t, period, 100.0)
    print("GPU i/o data from CPU performance")
    repeat(lambda: add_func(pulses, noise))

    print("GPU i/o data from GPU performance")
    pulses_device = cuda.to_device(pulses)
    noise_device = cuda.to_device(noise)
    waveform_cuda = cuda.device_array(shape=(n,), dtype=np.float32)
    # Here I see really good results
    repeat(lambda: add_func(pulses_device, noise_device, out=waveform_cuda))
    waveform = waveform_cuda.copy_to_host()

    plt.plot(waveform)
    plt.show()


set_2()
