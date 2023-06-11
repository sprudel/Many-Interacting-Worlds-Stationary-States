from libc.math cimport sin, cos, acos, exp, sqrt, fabs

import numpy as np
cimport numpy as np
#define gauß Kernel functions
cdef double norm = sqrt(2)/(2*sqrt(np.pi))

cdef inline np.float64_t K(np.float64_t x):
    return norm*exp(-x**2/2)
cdef inline np.float64_t Kdiff(np.float64_t x):
    return -norm*x*exp(-x**2/2)
cdef inline np.float64_t Kdiff2(np.float64_t x):
    return norm*(x**2*exp(-x**2/2) - exp(-x**2/2))
cdef inline np.float64_t Kdiff3(np.float64_t x):
    return norm*(-x**3*exp(-x**2/2)+ 3*x*exp(-x**2/2))