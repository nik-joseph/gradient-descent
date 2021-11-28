import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from matrix_element_sum import gpu_matrix_sum
from matrix_product import gpu_matrix_product


def timeit(func, *args, **kwargs):
    start = datetime.datetime.now()
    func(*args, **kwargs)
    return (datetime.datetime.now() - start).total_seconds()


def simple_sum(a, b):
    return np.array([
        [
            i + j for i, j in zip(row_a, row_b)
        ] for row_a, row_b in zip(a, b)
    ])


def simple_product(a, b):
    return np.array([
        [
            sum([
                x * y
                for x, y in zip(i_row, j_row)
            ])
            for j_row in zip(*b)
        ]
        for i_row in a
    ])


def get_report(cpu_func, gpu_func, title, matrix_sizes=(10, 1000, 10)):
    matrices_list = [
        np.random.randint(matrix_size, size=[2, matrix_size, matrix_size])
        for matrix_size in range(*matrix_sizes)
    ]
    cpu_report = [
        timeit(cpu_func, *matrices)
        for matrices in tqdm(matrices_list)
    ]
    gpu_report = [
        timeit(gpu_func, *matrices)
        for matrices in tqdm(matrices_list)
    ]

    plt.plot(range(*matrix_sizes), cpu_report, label='CPU')
    plt.plot(range(*matrix_sizes), gpu_report, label='GPU')

    plt.title(title)
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution time(seconds)')
    plt.legend()
    plt.show()


get_report(simple_sum, gpu_matrix_sum, 'Matrix Sum', matrix_sizes=(10, 1500, 10))
# get_report(simple_product, gpu_matrix_product, 'Matrix Product', matrix_sizes=(10, 250, 10))
# get_report(np.matmul, gpu_matrix_product, 'Matrix Product')

