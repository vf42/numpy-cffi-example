import numpy as np
import time

import myutils as my


"""
Comparing the speed of row_reduce implementation in Python and C.
Benchmark process:
1. For each matrix size,
2. Generate 100 random matrices of that size.
3. Run each implementation on each matrix.
4. Report stats.
"""


def timed_sample(f):
    start_time = time.monotonic_ns()
    f()
    return time.monotonic_ns() - start_time


def print_stats_header():
    print("version   matrix      count      average       median         90th         99th")


ns_in_ms = 1000000


def print_stats(version, matrix_size, samples):
    count = len(samples)
    avg = np.average(samples) / ns_in_ms
    median = np.median(samples) / ns_in_ms
    percentile90 = np.percentile(samples, 90) / ns_in_ms
    percentile99 = np.percentile(samples, 99) / ns_in_ms
    print(f"{version:7} {matrix_size:8d} {count:10d} {avg:12.4f} "
          f"{median:12.4f} {percentile90:12.4f} {percentile99:12.4f}")


if __name__ == '__main__':
    # Warm up the library, otherwise will get one slow call.
    my.row_reduce_c(np.array([[1, 2, 3], [4, 5, 6]]))
    print_stats_header()
    for i in (5, 10, 100, 200, 500, 1000):
        pysamples = []
        csamples1 = []
        csamples2 = []
        for j in range(100):
            m = np.random.rand(i, i)
            pysamples.append(timed_sample(lambda: my.row_reduce_py(m)))
            csamples1.append(timed_sample(lambda: my.row_reduce_c(m)))
            csamples2.append(timed_sample(lambda: my.row_reduce_c2(m)))
        print_stats("Python", i, pysamples)
        print_stats("C1", i, csamples1)
        print_stats("C2", i, csamples2)
