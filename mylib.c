#include "mylib.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// Using same epsilon value as in python.
bool is_zero(double x) { return fabs(x) < 2.220446049250313e-15; }

// Find the row with the largest value in the pivot column.
extern inline void find_max_pivot_row(double* m, size_t rows, size_t cols,
    size_t pivot_row, size_t pivot_col, size_t* max_row, double* pivot)
{
    for (size_t i = pivot_row + 1; i < rows; ++i) {
        if (fabs(m[i * cols + pivot_col]) > fabs(*pivot)) {
            *max_row = i;
            *pivot = m[i * cols + pivot_col];
        }
    }
}

// Swap two rows.
extern inline void swap_rows(
    double* m, size_t rows, size_t cols, size_t row1, size_t row2)
{
    if (row1 != row2) {
        // TODO: Use memcpy?
        for (size_t i = 0; i < cols; ++i) {
            double temp = m[row2 * cols + i];
            m[row2 * cols + i] = m[row1 * cols + i];
            m[row1 * cols + i] = temp;
        }
    }
}

// Multiply a row by a scalar.
extern inline void multiply_row(double* m, size_t cols, size_t row, double x)
{
    for (size_t i = 0; i < cols; ++i) {
        m[row * cols + i] *= x;
    }
}

// Subtract the pivot row from a given row, achieving 0 in the pivot column.
extern inline void subtract_pivot_row(
    double* m, size_t cols, size_t pivot_row, size_t pivot_col, size_t i)
{
    double factor = m[i * cols + pivot_col];
    for (size_t j = pivot_col; j < cols; ++j) {
        m[i * cols + j] -= factor * m[pivot_row * cols + j];
    }
}

/*
 * Implementation of Gaussian Elimination, editing data in-place.
 */
void row_reduce(double* m, const size_t rows, const size_t cols)
{
    if (!m || !rows || !cols) {
        return;
    }
    size_t pivot_row = 0;
    size_t pivot_col = 0;
    while (pivot_row < rows && pivot_col < cols) {
        size_t max_row = pivot_row;
        double pivot = m[pivot_row * cols + pivot_col];
        find_max_pivot_row(
            m, rows, cols, pivot_row, pivot_col, &max_row, &pivot);
        if (is_zero(pivot)) {
            // The pivot column is all zeros, so we can't reduce it any further.
            ++pivot_col;
            continue;
        }
        // Swap the pivot row with the max row.
        swap_rows(m, rows, cols, pivot_row, max_row);
        // Reduce the pivot row.
        multiply_row(m, cols, pivot_row, 1 / pivot);
        // Subtract multiples of the pivot row from all other rows.
        for (size_t i = 0; i < rows; ++i) {
            if (i == pivot_row) {
                continue;
            }
            subtract_pivot_row(m, cols, pivot_row, pivot_col, i);
        }
        ++pivot_row;
        ++pivot_col;
    }
}

/*
 * Implementation of Gaussian Elimination, returning a copy of the input data.
 */
double* row_reduce_copy(
    const double* in_m, const size_t rows, const size_t cols)
{
    double* m = malloc(rows * cols * sizeof(double));
    memcpy(m, in_m, rows * cols * sizeof(double));
    row_reduce(m, rows, cols);
    return m;
}

