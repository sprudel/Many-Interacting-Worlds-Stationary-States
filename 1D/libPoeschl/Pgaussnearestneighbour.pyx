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




cdef class Pgaussnearestneighbour(MIWBase_pval):
    cdef np.int_t[:] phase
    cdef np.float64_t [:] kernel_pos
    cdef np.float64_t [:] hs
    cdef size_t n_kernel
    cdef size_t recursive_interpolation
    cdef np.float64_t alpha
    cdef np.float64_t gamma
    cdef int n_nodes
    cdef np.int_t[:] node_indices
    cdef np.float64_t[:] node_pos
    
    property phase:
        def __get__(self):
            return np.asarray(self.phase)
    property hs:
        def __get__(self):
            return np.asarray(self.hs)
        
    def __init__(self, worlds, alpha=1., gamma=0.5, phase=None,filter_indices=None):
        self.alpha = alpha
        self.gamma = gamma
        if phase is None:
            self.phase = np.zeros_like(worlds, dtype=np.int)
        else:
            self.phase = phase
        
        #saving phasjumps
        self.n_nodes = np.sum(np.diff(np.asarray(self.phase))!=0)
        print(self.n_nodes)
        self.node_indices = np.zeros(self.n_nodes,dtype=np.int)
        #saves indices of neighbouring worlds at node,
        #ie index i -> between worlds[i]+worlds[i+1] 
        self.node_indices = np.where(np.diff(np.asarray(self.phase))!=0)[0]
        
        self.n_kernel = worlds.shape[0]-self.n_nodes
        
        self.N = worlds.shape[0]
        #additional kernels for node handling (-> negative h-vals)
        self.hs = np.ones(self.N+self.n_nodes, dtype=np.float64)
        self.kernel_pos = np.zeros(self.N + self.n_nodes, dtype=np.float64)
             
        MIWBase_pval.__init__(self,worlds,filter_indices)
            
    cdef inline np.float64_t p_eval(self, np.float64_t x):
        cdef np.float64_t tmp = 0.
        cdef size_t i
        for i in range(self.N+self.n_nodes):
            tmp += 1./(<np.float64_t>self.n_kernel)*(K((x-self.kernel_pos[i])/self.hs[i])/self.hs[i])
            #print("i: {}, kernelpos: {}, h: {}: val: {}".format(i,self.kernel_pos[i], self.hs[i],   1./(<np.float64_t>self.n_kernel)*(K((x-self.kernel_pos[i])/self.hs[i])/self.hs[i]))) 
        return tmp
    
    def p_eval_vec(self, x):
        return np.vectorize(lambda y: self.p_eval(y))(x)
    
    cdef inline np.float64_t single_kernel(self, np.float64_t x, size_t i):
        return 1./(<np.float64_t>self.n_kernel)*(K((x-self.kernel_pos[i])/self.hs[i])/self.hs[i])
    
    def get_all_K(self,x):
        tmp = []
        for i in range(self.N+self.n_nodes):
            tmpf = np.vectorize(lambda y: self.single_kernel(y,i))
            tmp.append(tmpf(x))
        return tmp
    
    cdef inline np.float64_t pdiff_eval(self, np.float64_t x):
        cdef np.float64_t tmp = 0.
        cdef size_t i
        for i in range(self.N+self.n_nodes):
            tmp += 1./(<np.float64_t>self.n_kernel)*(Kdiff((x-self.kernel_pos[i])/self.hs[i])/self.hs[i]**2)
        return tmp
    
    cdef inline np.float64_t pdiff2_eval(self, np.float64_t x):
        cdef np.float64_t tmp = 0.
        cdef size_t i
        for i in range(self.N+self.n_nodes):
            tmp += 1./(<np.float64_t>self.n_kernel)*(Kdiff2((x-self.kernel_pos[i])/self.hs[i])/self.hs[i]**3)
        return tmp
    
    cdef inline np.float64_t pdiff3_eval(self, np.float64_t x):
        cdef np.float64_t tmp = 0.
        cdef size_t i
        for i in range(self.N+self.n_nodes):
            tmp += 1./(<np.float64_t>self.n_kernel)*(Kdiff3((x-self.kernel_pos[i])/self.hs[i])/self.hs[i]**4)       
        return tmp
    
    cdef int _update_pvals(self):
        #update h values for interpolation:
        cdef size_t i,j
        cdef double dist
        cdef double tmp_p
        cdef double tmp_x
        for i in range(1,self.N-1):
            dist = min(self.worlds[i+1]-self.worlds[i],self.worlds[i]-self.worlds[i-1])
            self.hs[i] = dist**self.gamma*self.alpha
        self.hs[0] = (self.worlds[1]-self.worlds[0])**self.gamma*self.alpha
        self.hs[self.N-1] = (self.worlds[self.N-1]-self.worlds[self.N-2])**self.gamma*self.alpha
        #update kernel positions for worlds
        for i in range(self.N):
            self.kernel_pos[i] = self.worlds[i]
        #update node position
        #print(np.array(self.kernel_pos))
        for i in range(self.n_nodes):
            self.kernel_pos[self.N+i] = (self.worlds[self.node_indices[i]]+self.worlds[self.node_indices[i]+1])/2.
        #update node hs
        for i in range(self.n_nodes):
            tmp_p =0.
            tmp_x = self.kernel_pos[self.N+i]
            for j in range(self.N):
                tmp_p += 1./(<np.float64_t>self.n_kernel)*(K((tmp_x-self.kernel_pos[j])/self.hs[j])/self.hs[j])
            self.hs[self.N+i] = - norm/self.n_kernel/tmp_p 
    
            
        #interpolate dervivates with gauss interpolation
        for i in range(self.N):
            self.pvals[i,0] = self.p_eval(self.worlds[i])
            self.pvals[i,1] = self.pdiff_eval(self.worlds[i])
            self.pvals[i,2] = self.pdiff2_eval(self.worlds[i])
            self.pvals[i,3] = self.pdiff3_eval(self.worlds[i])

        return 0
    

    