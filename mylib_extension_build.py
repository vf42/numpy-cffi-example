from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("""
void row_reduce(double* m, size_t rows, size_t cols);
""")

ffibuilder.set_source("_my",  # name of the output C extension
                      """
                      #include "mylib.h"
                      """,
                      sources=['mylib.c'],
                      libraries=['m'])

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
