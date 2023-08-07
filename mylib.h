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

#endif