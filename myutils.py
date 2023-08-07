import numpy as np


import _my.lib as mylib
from cffi import FFI
ffi = FFI()

tolerance = np.finfo(float).eps * 10


def is_zero(x):
    """
    Check if x is effectively zero.
    """
    return np.abs(x) < tolerance


def row_reduce_py(input_m):
    """
    Row reduce a matrix to reduced row echelon form using Gaussian elimination.
    Python version.
    """
    m = np.array(
        input_m, dtype=float)  # Working with a copy to preserve input.
    rows, cols = m.shape
    pivot_row = 0
    pivot_col = 0
    while pivot_row < rows and pivot_col < cols:
        # Find the row with the largest value in the pivot column.
        max_row = np.argmax(np.abs(m[pivot_row:, pivot_col])) + pivot_row
        if is_zero(m[max_row, pivot_col]):
            # No nonzero pivot in this column, move to the next column.
            pivot_col += 1
            continue
        # Swap the pivot row with max_row.
        if pivot_row != max_row:
            m[[pivot_row, max_row]] = m[[max_row, pivot_row]]
        # Get the pivot value.
        pivot = m[pivot_row, pivot_col]
        # Divide the pivot row by the pivot value.
        m[pivot_row, :] /= pivot
        # Subtract multiples of the pivot row from all other rows.
        for i in range(rows):
            if i != pivot_row:
                m[i, :] -= m[i, pivot_col] * m[pivot_row, :]
        # Move to the next pivot row and column.
        pivot_row += 1
        pivot_col += 1
    return m


def row_reduce_c(input_m):
    """
    Row reduce a matrix to reduced row echelon form using Gaussian elimination.
    """
    # Make a copy of the input matrix.
    m = np.array(input_m, dtype=float)
    # Get a pointer to the numpy array data.
    d = ffi.from_buffer("double[]", m)
    mylib.row_reduce(d, m.shape[0], m.shape[1])
    return m


def row_reduce_c2(input_m):
    """
    Row reduce a matrix to reduced row echelon form using Gaussian elimination.
    """
    # Get a pointer to the numpy array data.
    ind = ffi.from_buffer("double[]", input_m)
    # This will return a pointer to the raw data, which we have to process.
    outd = mylib.row_reduce_copy(ind, input_m.shape[0], input_m.shape[1])
    # Wrap the raw data to the buffer object.
    b = ffi.buffer(outd, np.dtype("float").itemsize * input_m.size)[:]
    # Feed it to numpy.
    m = np.frombuffer(b, dtype=float)
    return m.reshape(input_m.shape)