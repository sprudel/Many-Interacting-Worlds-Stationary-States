from libc.math cimport cosh, tanh
cdef int n = 6
cdef double F(double x):
    return -n*(n+1.)*tanh(x)/cosh(x)**2
cdef double V(double x):
    return -n*(n+1)/2./cosh(x)**2