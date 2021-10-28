from numba import jit
import numpy as np
import math

from utils import repeat


@jit
def hypot(x, y):
    # Implementation from https://en.wikipedia.org/wiki/Hypot
    x = abs(x)
    y = abs(y)
    t = min(x, y)
    x = max(x, y)
    t = t / x
    return x * math.sqrt(1 + t * t)


# Equivalent to
# def hypot(x, y):
#     x = abs(x);
#     y = abs(y);
#     t = min(x, y);
#     x = max(x, y);
#     t = t / x;
#     return x * math.sqrt(1+t*t)
#
# hypot = jit(hypot)

def set_1():

    print("CPU")
    repeat(lambda: hypot.py_func(3.0, 4.0))

    print("GPU")
    repeat(lambda: hypot(3.0, 4.0))


# set_1()


def set_2():
    # This is designed to fail
    @jit(nopython=True)
    def cannot_compile(x):
        return x['key']

    print(cannot_compile(dict(key='value')))

# set_2()


def set_3():
    @jit(nopython=True)
    def ex1(x, y, o):
        for i in range(x.shape[0]):
            o[i] = hypot(x[i], y[i])

    in1 = np.arange(10, dtype=np.float64)
    in2 = 2 * in1 + 1
    out = np.empty_like(in1)

    print('in1:', in1)
    print('in2:', in2)

    ex1(in1, in2, out)

    print('out:', out)
    # This test will fail until you fix the ex1 function
    np.testing.assert_almost_equal(out, np.hypot(in1, in2))


# set_3()
