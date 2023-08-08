from cffi import FFI
ffibuilder = FFI()

# Copy the required definitions from the header file.
ffibuilder.cdef("""
void free(void *ptr); // Required to ensure the allocated objects are freed.
void row_reduce(double* m, size_t rows, size_t cols);
double* row_reduce_copy(
    const double* in_m, const size_t rows, const size_t cols);
                
typedef enum {
    KERNEL_OK,
    KERNEL_NOT_RREF,
    KERNEL_HAS_ZERO_ROWS,
} kernel_result_status;
typedef struct {
    double* m;
    size_t count;
    kernel_result_status status;
} kernel_result;
kernel_result kernel(double* m, size_t rows, const size_t cols);
""")

ffibuilder.set_source("_my",  # name of the output C extension
                      """
                      #include "mylib.h"
                      """,
                      sources=['mylib.c'],
                      libraries=['m'])

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
