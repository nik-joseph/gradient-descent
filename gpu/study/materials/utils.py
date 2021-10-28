from tqdm import trange
from datetime import datetime
from numba import cuda


class Timer:
    def __init__(self):
        self.start = None

    def __enter__(self):
        self.start = datetime.now()
        return self

    def checkpoint(self):
        end = datetime.now()
        diff = end - self.start
        self.start = end
        return diff

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def repeat(func, count=1000):
    times = []
    with Timer() as timer:
        for _ in trange(count):
            func()
            cuda.synchronize()
            times.append(timer.checkpoint())

    times = list(map(lambda x: x.total_seconds(), times))

    print(f"\nMax runtime {max(times)}")
    print(f"Min runtime {min(times)}")
    print(f"Total runtime {sum(times)}\n")

    return func()
