import math
import scipy.stats  # for definition of gaussian distribution
import numpy as np
from numba import vectorize, cuda
from matplotlib import pyplot as plt
from utils import repeat

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = np.arange(4*4).reshape((4, 4))
b_col = b[:, np.newaxis]


@vectorize(['int32(int32, int32)'], target='cuda')
def add_func(x, y):
    return x + y


def set_1():
    repeat(lambda: np.add(b_col, c))       # CPU

    # Much slower ?? Probably wont finish btw
    repeat(lambda: add_func(b_col, c))     # GPU


# set_1()


def set_2():
    import math  # Note that for the CUDA target, we need to use the scalar functions from the math module, not NumPy

    SQRT_2PI = np.float32(
        (2 * math.pi) ** 0.5)  # Precompute this constant as a float32.  Numba will inline it at compile time.

    @vectorize(['float32(float32, float32, float32)'], target='cuda')
    def gaussian_pdf(x_, mean_, sigma_):
        """ Compute the value of a Gaussian probability density function at x with given mean and sigma. """
        return math.exp(-0.5 * ((x_ - mean_) / sigma_) ** 2) / (sigma_ * SQRT_2PI)

    # Evaluate the Gaussian a million times!
    x = np.random.uniform(-3, 3, size=1000000).astype(np.float32)
    mean = np.float32(0.0)
    sigma = np.float32(1.0)

    norm_pdf = scipy.stats.norm
    print("norm_pdf")
    repeat(lambda: norm_pdf.pdf(x, loc=mean, scale=sigma))

    print("gaussian_pdf")
    repeat(lambda: gaussian_pdf(x, mean, sigma))


# set_2()


def set_3():
    @cuda.jit(device=True)
    def polar_to_cartesian(rho, theta):
        x = rho * math.cos(theta)
        y = rho * math.sin(theta)
        return x, y  # This is Python, so let's return a tuple

    @vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
    def polar_distance(rho1, theta1, rho2, theta2):
        x1, y1 = polar_to_cartesian(rho1, theta1)
        x2, y2 = polar_to_cartesian(rho2, theta2)

        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    n = 1000000
    rho1 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
    theta1 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    rho2 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
    theta2 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)

    # This is really fast
    print(polar_distance(rho1, theta1, rho2, theta2))


# set_3()

def set_4():
    n = 100000
    noise = np.random.normal(size=n) * 3
    pulses = np.maximum(np.sin(np.arange(n) / (n / 23)) - 0.3, 0.0)
    waveform = ((pulses * 300) + noise).astype(np.int16)
    plt.plot(waveform)
    plt.show()

    @vectorize(['int16(int16, int16)'], target='cuda')
    def zero_suppress(waveform_value, threshold):
        return waveform_value if waveform_value >= threshold else 0

    # the noise on the baseline should disappear when zero_suppress is implemented
    plt.plot(zero_suppress(waveform, 15.0))
    # that's cool
    plt.show()


# set_4()
