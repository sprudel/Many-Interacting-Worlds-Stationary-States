"""This file implements the Groundstate Algorthim for polyinterpolation of cumulative F values"""
include "MIWBase.pxi"
include "PotentialForce.pxi"
include "Kernel.pxi"
from libc.math cimport sin, cos, acos, exp, sqrt, fabs


#not consistent in code, has to be checked if changed  in whole code
cdef double hbar =1
cdef double m = 1


import numpy as np
cimport numpy as np



cdef class Pgaussinterpolation(MIWBase_pval):
    cdef np.float64_t [:] pbetween
    cdef np.float64_t [:] pos_between
    cdef np.int_t[:] phase
    cdef np.float64_t [:] hs

    cdef size_t n_kernel
    cdef size_t recursive_interpolation
    
    cdef int n_nodes
    
    property pbetween:
        def __get__(self):
            return np.asarray(self.pbetween)
    property phase:
        def __get__(self):
            return np.asarray(self.phase)
    property hs:
        def __get__(self):
            return np.asarray(self.hs)
        
    def __init__(self, worlds, phase=None,recursive_interpolation=5,filter_indices=None):
        if phase is None:
            self.phase = np.zeros_like(worlds, dtype=np.int)
        else:
            self.phase = phase
        
        #saving phasjumps
        self.n_nodes = np.sum(np.diff(np.asarray(self.phase))!=0)
        print(self.n_nodes)
        
        self.pbetween = np.zeros(worlds.shape[0]-1, dtype=np.float64)
        self.pos_between = np.zeros(worlds.shape[0]-1, dtype=np.float64)
        self.hs = np.ones(worlds.shape[0]-1, dtype=np.float64)
        self.n_kernel = worlds.shape[0]-1-2*self.n_nodes
        self.recursive_interpolation = recursive_interpolation
               
        MIWBase_pval.__init__(self,worlds,filter_indices)
            
    cdef inline np.float64_t p_eval(self, np.float64_t x):
        cdef np.float64_t tmp = 0.
        cdef size_t i
        for i in range(self.N-1):
            tmp += 1./(<np.float64_t>self.n_kernel)*(K((x-self.pos_between[i])/self.hs[i])/self.hs[i])
        return tmp
    
    def p_eval_vec(self, x):
        return np.vectorize(lambda y: self.p_eval(y))(x)
    
    cdef inline np.float64_t single_kernel(self, np.float64_t x, size_t i):
        return 1./(<np.float64_t>self.n_kernel)*(K((x-self.pos_between[i])/self.hs[i])/self.hs[i])
    
    def get_all_K(self,x):
        tmp = []
        for i in range(self.N-1):
            tmpf = np.vectorize(lambda y: self.single_kernel(y,i))
            tmp.append(tmpf(x))
        return tmp
    
    cdef inline np.float64_t pdiff_eval(self, np.float64_t x):
        cdef np.float64_t tmp = 0.
        cdef size_t i
        for i in range(self.N-1):
            tmp += 1./(<np.float64_t>self.n_kernel)*(Kdiff((x-self.pos_between[i])/self.hs[i])/self.hs[i]**2)
        return tmp
    
    cdef inline np.float64_t pdiff2_eval(self, np.float64_t x):
        cdef np.float64_t tmp = 0.
        cdef size_t i
        for i in range(self.N-1):
            tmp += 1./(<np.float64_t>self.n_kernel)*(Kdiff2((x-self.pos_between[i])/self.hs[i])/self.hs[i]**3)
        return tmp
    
    cdef inline np.float64_t pdiff3_eval(self, np.float64_t x):
        cdef np.float64_t tmp = 0.
        cdef size_t i
        for i in range(self.N-1):
            tmp += 1./(<np.float64_t>self.n_kernel)*(Kdiff3((x-self.pos_between[i])/self.hs[i])/self.hs[i]**4)       
        return tmp
    cdef inline update_between_values(self):
        cdef size_t i
        for i in range(self.N-1):
            self.pos_between[i] = (self.worlds[i+1]+self.worlds[i])/2.
        for i in range(self.N-1):
            self.pbetween[i] = 1./(<np.float64_t>self.N)/(self.worlds[i+1]-self.worlds[i])*delta(self.phase[i],self.phase[i+1])
    
    cdef int _update_pvals(self):
        #calculate p values between worlds
        self.update_between_values()
        #update h values for interpolation:
        cdef size_t i,j
        for i in range(self.recursive_interpolation):
            for j in range(self.N-1):
                self.hs[j] = fabs(K(0)/(self.pbetween[j]-self.p_eval(self.pos_between[j])+K(0)/self.hs[j]))*(-1+delta(self.phase[j],self.phase[j+1])*2)

        #interpolate dervivates with gauss interpolation
        for i in range(self.N):
            self.pvals[i,0] = self.p_eval(self.worlds[i])
            self.pvals[i,1] = self.pdiff_eval(self.worlds[i])
            self.pvals[i,2] = self.pdiff2_eval(self.worlds[i])
            self.pvals[i,3] = self.pdiff3_eval(self.worlds[i])

        return 0
    

    