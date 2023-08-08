#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mylib.h"

bool is_zero(double x);

bool cmparr(double* a, double* b, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (!is_zero(a[i] - b[i])) {
            return false;
        }
    }
    return true;
}

void test_row_reduce_1()
{
    const size_t rows = 4, cols = 3;
    double m[12] = { 3, -3, 0, 1, 2, 3, 7, -5, 2, 3, -1, 2 };
    double expected[12] = { 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0 };
    row_reduce(m, rows, cols);
    assert(cmparr(m, expected, rows * cols));
}

void test_row_reduce_2()
{
    const size_t rows = 4, cols = 4;
    double m[16] = { 1., 0, 2, 0, 1, 1, 0, 0, 1, 2, 0, 1, 1, 1, 1, 1 };
    double expected[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    row_reduce(m, rows, cols);
    assert(cmparr(m, expected, rows * cols));
}

kernel_result_status validate_kernel_input(
    const double* m, const size_t rows, const size_t cols);

void test_validate_kernel_input()
{
    double m1[9] = { 1., 0, 0, 0, 1, 0, 0, 0, 1 };
    assert(validate_kernel_input(m1, 3, 3) == KERNEL_OK);
    double m2[9] = { 1., 0, 0, 1, 1, 0, 0, 0, 1 };
    assert(validate_kernel_input(m2, 3, 3) == KERNEL_NOT_RREF);
    double m3[9] = { 1., 0, 1, 0, 1, 0, 0, 0, 0 };
    assert(validate_kernel_input(m3, 3, 3) == KERNEL_HAS_ZERO_ROWS);
    double m4[8] = {1.,   0.,   0.,   1.,0.,   1., -0.5, -0.5};
    assert(validate_kernel_input(m4, 2, 4) == KERNEL_OK);
}

void test_kernel_1()
{
    const size_t rows = 3, cols = 5;
    double m[15] = { 1., 3, 0, 0, 3, 0, 0, 1, 0, 9, 0, 0, 0, 1, -4 };
    double expected[10] = { 3., -1, 0, 0, 0, 3., 0, 9, -4, -1 };
    kernel_result result = kernel(m, rows, cols);
    assert(2 == result.count);
    assert(cmparr(result.m, expected, 4));
    assert(cmparr(result.m + 4, expected + 4, 4));
}

void test_kernel_2()
{
    const size_t rows = 2, cols = 2;
    double m[15] = { 1, 0, 0, 1 };
    kernel_result result = kernel(m, rows, cols);
    assert(0 == result.count);
    assert(result.m == NULL);
}

int main()
{
    test_row_reduce_1();
    test_row_reduce_2();
    test_kernel_1();
    test_kernel_2();
    test_validate_kernel_input();
    printf("OK\n");

    return EXIT_SUCCESS;
}
