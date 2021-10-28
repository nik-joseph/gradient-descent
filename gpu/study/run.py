import numpy as np
from matrix_element_sum import element_wise_sum
from matrix_product import matrix_product


def sum_check():
    size = 1024
    mat_a = np.arange(size * size, dtype=np.int32).reshape((size, size))
    mat_b = np.arange(size * size, dtype=np.int32).reshape((size, size))

    print("Element wise sum")
    print("CPU\n")
    cpu_response = element_wise_sum(mat_a, mat_b, no_cuda=True)
    print("GPU\n")
    gpu_response = element_wise_sum(mat_a, mat_b, no_cuda=False)

    assert (cpu_response == gpu_response).all()


def product_check():
    size = 32
    mat_a = np.arange(size * size, dtype=np.float32).reshape((size, size))
    mat_b = np.arange(size * size, dtype=np.float32).reshape((size, size))

    # mat_a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(5, 2)
    # mat_b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(2, 5)

    print("Matrix product")
    print("CPU\n")
    cpu_response = matrix_product(mat_a, mat_b, no_cuda=True)
    print("GPU\n")
    gpu_response = matrix_product(mat_a, mat_b, no_cuda=False)

    assert np.allclose(cpu_response, gpu_response)


print("Sum Check\n")
sum_check()
print("Product Check\n")
product_check()
