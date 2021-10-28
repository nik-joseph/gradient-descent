import numpy as np
import numba.types
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba import guvectorize
import math
from utils import repeat


@cuda.jit
def compute_pi(rng_states, iterations, out):
    """Find the maximum value in values and store in result[0]"""
    thread_id = cuda.grid(1)

    # Compute pi by drawing random (x, y) points and finding what
    # fraction lie inside a unit circle
    inside = 0
    for i in range(iterations):
        x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        y = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if x ** 2 + y ** 2 <= 1.0:
            inside += 1

    out[thread_id] = 4.0 * inside / iterations


def set_1():
    # Random Numbers Topic
    threads_per_block = 64
    blocks = 24
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
    out = np.zeros(threads_per_block * blocks, dtype=np.float32)
    compute_pi[blocks, threads_per_block](rng_states, 10000, out)
    print('pi:', out.mean())


set_1()

# Shared Memory Topic

TILE_DIM = 32
BLOCK_ROWS = 8


@cuda.jit
def transpose(a_in, a_out):
    x = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.y

    for j in range(0, TILE_DIM, BLOCK_ROWS):
        a_out[x, y + j] = a_in[y + j, x]


def set_2():
    size = 1024
    a_in = cuda.to_device(np.arange(size * size, dtype=np.int32).reshape((size, size)))
    a_out = cuda.device_array_like(a_in)

    grid_shape = (int(size / TILE_DIM), int(size / TILE_DIM))

    def func():
        transpose[grid_shape, (TILE_DIM, BLOCK_ROWS)](a_in, a_out)
        cuda.synchronize()

    repeat(func)


# set_2()


TILE_DIM_PADDED = TILE_DIM + 1  # Read Mark Harris' blog post to find out why this improves performance!


@cuda.jit
def tile_transpose(a_in, a_out):
    # THIS CODE ASSUMES IT IS RUNNING WITH A BLOCK DIMENSION OF (TILE_SIZE x TILE_SIZE)
    # AND INPUT IS A MULTIPLE OF TILE_SIZE DIMENSIONS
    tile = cuda.shared.array((TILE_DIM, TILE_DIM_PADDED), numba.types.int32)

    x = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.y

    for j in range(0, TILE_DIM, BLOCK_ROWS):
        tile[cuda.threadIdx.y + j, cuda.threadIdx.x] = a_in[y + j, x]  # transpose tile into shared memory

    cuda.syncthreads()  # wait for all threads in the block to finish updating shared memory

    # Compute transposed offsets
    x = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.x
    y = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.y

    for j in range(0, TILE_DIM, BLOCK_ROWS):
        a_out[y + j, x] = tile[cuda.threadIdx.x, cuda.threadIdx.y + j]


def set_3():
    size = 1024
    a_in = cuda.to_device(np.arange(size * size, dtype=np.int32).reshape((size, size)))
    a_out = cuda.device_array_like(a_in)

    grid_shape = (int(size / TILE_DIM), int(size / TILE_DIM))
    a_out = cuda.device_array_like(a_in)  # replace with new array

    def func():
        tile_transpose[grid_shape, (TILE_DIM, BLOCK_ROWS)](a_in, a_out)
        cuda.synchronize()

    repeat(func)


# set_3()

# Generalized U funcs

@guvectorize(['(float32[:], float32[:])'],  # have to include the output array in the type signature
             '(i)->()',                 # map a 1D array to a scalar output
             target='cuda')
def l2_norm(vec, out):
    acc = 0.0
    for value in vec:
        acc += value**2
    out[0] = math.sqrt(acc)


def set_4():
    angles = np.random.uniform(-np.pi, np.pi, 10)
    cords = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    print(cords)
    print(l2_norm(cords))


# set_4()
