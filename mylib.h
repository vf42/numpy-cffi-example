#ifndef MYLIB_H
#define MYLIB_H

#include <stddef.h>

/*
 * Implementation of Gaussian Elimination, editing data in-place.
 */
void row_reduce(double* m, const size_t rows, const size_t cols);

/*
 * Implementation of Gaussian Elimination, returning a copy of the input data.
 */
double* row_reduce_copy(
    const double* in_m, const size_t rows, const size_t cols);

/*
 * Status flags for kernel function.
 */
typedef enum kernel_result_status kernel_result_status;
enum kernel_result_status {
    KERNEL_OK,
    KERNEL_NOT_RREF,
    KERNEL_HAS_ZERO_ROWS,
};

/*
 * Kernel function result data.
 */
typedef struct kernel_result kernel_result;
struct kernel_result {
    double* m; // Array containing the vectors in the kernel.
    size_t count; // Number of vectors in the kernel.
    kernel_result_status status;
};

/*
 * Return the kernel of the input matrix.
 */
kernel_result kernel(const double* m, size_t rows, const size_t cols);

#endif
