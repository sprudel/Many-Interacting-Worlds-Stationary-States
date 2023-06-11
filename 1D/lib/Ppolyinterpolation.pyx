"""This file implements the Groundstate Algorthim for polyinterpolation of cumulative F values"""
include "MIWBase.pxi"
include "PotentialForce.pxi"

#not consistent in code, has to be checked if changed  in whole code
cdef double hbar =1
cdef double m = 1

import numpy as np
cimport numpy as np
cdef np.float64_t neville(np.float64_t[:] x,np.float64_t[:] y, np.float64_t x0):
    """Through any N points y[i] = f(x[i]), there is a unique
    polynomial P order N-1.  Neville's algorithm is used for finding
    interpolates of this unique polynomial at any point x0."""
    cdef int n = x.shape[0]
    cdef np.float64_t [:] p = y.copy()
    cdef size_t k,j
    for k in range(1,n):
        for j in range(n-k):
                p[j] = ((x0-x[j+k])*p[j]+(x[j]-x0)*p[j+1])/(x[j]-x[j+k])
    return p[0]

#maxmial 10 point interpolation
cdef np.float64_t [:] p = np.zeros(10)
cdef np.float64_t [:] pdiff = np.zeros(10)
cdef np.float64_t [:] pdiff2 = np.zeros(10)
cdef np.float64_t [:] pdiff3 = np.zeros(10)
cdef np.float64_t [:] result = np.zeros(3)

cdef np.float64_t[:] neville_diffs3(np.float64_t[:] x,np.float64_t[:] y, np.float64_t x0):
    """Through any N points y[i] = f(x[i]), there is a unique
    polynomial P order N-1.  Neville's algorithm is used for finding
    interpolates of this unique polynomial at any point x0. This function calculates first to 3rd derivative"""
    cdef int n = x.shape[0]
    cdef size_t k,j
    for k in range(n):
        p[k] = y[k]
    for k in range(n):
        pdiff[k] = 0.
    for k in range(n):
        pdiff2[k] = 0.
    for k in range(n):
        pdiff3[k] = 0.   
    for k in range(1,n):
        for j in range(n-k):
                pdiff3[j] = (3*pdiff2[j]+(x0-x[j+k])*pdiff3[j]+(x[j]-x0)*pdiff3[j+1]-3*pdiff2[j+1])/(x[j]-x[j+k])
                pdiff2[j] = (2*pdiff[j]+(x0-x[j+k])*pdiff2[j]+(x[j]-x0)*pdiff2[j+1]-2*pdiff[j+1])/(x[j]-x[j+k])
                pdiff[j] = ((x0-x[j+k])*pdiff[j]+p[j]+(x[j]-x0)*pdiff[j+1]-p[j+1])/(x[j]-x[j+k])
                p[j] = ((x0-x[j+k])*p[j]+(x[j]-x0)*p[j+1])/(x[j]-x[j+k])
    result[0] = pdiff[0]
    result[1] = pdiff2[0]
    result[2] = pdiff3[0]
    return result

cdef np.float64_t [:] result2 = np.zeros(4)
cdef np.float64_t[:] neville_p_diffs3(np.float64_t[:] x,np.float64_t[:] y, np.float64_t x0):
    """Through any N points y[i] = f(x[i]), there is a unique
    polynomial P order N-1.  Neville's algorithm is used for finding
    interpolates of this unique polynomial at any point x0. This function calculates p and first to 3rd derivative"""
    cdef int n = x.shape[0]
    cdef size_t k,j
    for k in range(n):
        p[k] = y[k]
    for k in range(n):
        pdiff[k] = 0.
    for k in range(n):
        pdiff2[k] = 0.
    for k in range(n):
        pdiff3[k] = 0.   
    for k in range(1,n):
        for j in range(n-k):
                pdiff3[j] = (3*pdiff2[j]+(x0-x[j+k])*pdiff3[j]+(x[j]-x0)*pdiff3[j+1]-3*pdiff2[j+1])/(x[j]-x[j+k])
                pdiff2[j] = (2*pdiff[j]+(x0-x[j+k])*pdiff2[j]+(x[j]-x0)*pdiff2[j+1]-2*pdiff[j+1])/(x[j]-x[j+k])
                pdiff[j] = ((x0-x[j+k])*pdiff[j]+p[j]+(x[j]-x0)*pdiff[j+1]-p[j+1])/(x[j]-x[j+k])
                p[j] = ((x0-x[j+k])*p[j]+(x[j]-x0)*p[j+1])/(x[j]-x[j+k])
    result2[0] = p[0]
    result2[1] = pdiff[0]
    result2[2] = pdiff2[0]
    result2[3] = pdiff3[0]
    return result2

cdef class Ppolyinterpolation(MIWBase_pval):
    cdef int int_points
    def __init__(self, worlds,int_points=2,fixed_boundaries=0):
        self.int_points = int_points
        MIWBase_pval.__init__(self,worlds,fixed_boundaries=fixed_boundaries)
        
    cdef int _update_pvals(self):
        """updates all pvalues befor force calculation"""
        cdef size_t i
        #calculate p values
        for i in range(1,self.N-1):
            self.pvals[i,0] = 1./(self.N+1)/2.*(1./(self.worlds[i]-self.worlds[i-1])+1./(self.worlds[i+1]-self.worlds[i]))
        #special factor to stabilize boundaries
        self.pvals[0,0] = 1./(self.N+1)*(1./(self.worlds[1]-self.worlds[0]))*3./4.
        self.pvals[self.N-1,0] = 1./(self.N+1)*(1./(self.worlds[self.N-1]-self.worlds[self.N-2]))*3./4.
        
        for i in range(self.int_points,self.N-self.int_points):
            self.pvals[i,1:4] = neville_diffs3(self.worlds[i-self.int_points:i+self.int_points+1],self.pvals[i-self.int_points:i+self.int_points+1,0],self.worlds[i])
        #for boundary cases:
        for i in range(self.int_points):
            self.pvals[i,1:4]= neville_diffs3(self.worlds[0:self.int_points*2+1],self.pvals[0:self.int_points*2+1,0],self.worlds[i])
        for i in range(self.N-self.int_points, self.N):
            self.pvals[i,1:4] = neville_diffs3(self.worlds[self.N-self.int_points*2-1:self.N],self.pvals[self.N-self.int_points*2-1:self.N,0],self.worlds[i])
        return 0
    
    
cdef class Phalfstep_polyinterpolation(MIWBase_pval):
    cdef int int_points
    cdef np.float64_t [:] pbetween
    cdef np.float64_t [:] pos_between
    property pbetween:
        def __get__(self):
            return np.asarray(self.pbetween)
    cdef np.int_t[:] phase
    property phase:
        def __get__(self):
            return np.asarray(self.phase)
    def __init__(self, worlds, int_points=2, phase=None, fixed_boundaries=0):
        self.int_points = int_points
        if phase is None:
            self.phase = np.zeros_like(worlds, dtype=np.int)
        else:
            self.phase = phase
        self.pbetween = np.zeros(worlds.shape[0]-1, dtype=np.float64)
        self.pos_between = np.zeros(worlds.shape[0]-1, dtype=np.float64)
        MIWBase_pval.__init__(self,worlds,fixed_boundaries=fixed_boundaries)
        
    cdef int _update_pvals(self):
        #calculate p values between worlds
        cdef size_t i
        for i in range(0,self.N-1):
            self.pos_between[i] = (self.worlds[i+1]+self.worlds[i])/2.
            self.pbetween[i] = 1./(self.N+1)/(self.worlds[i+1]-self.worlds[i])*delta(self.phase[i],self.phase[i+1])
        #interpolate dervivates with polynomial interpolation
        for i in range(self.int_points, self.N-self.int_points):
            self.pvals[i] = neville_p_diffs3(self.pos_between[i-self.int_points:i+self.int_points],self.pbetween[i-self.int_points:i+self.int_points],self.worlds[i])
        #for boundary cases:
        for i in range(self.int_points):
            self.pvals[i] = neville_p_diffs3(self.pos_between[0:self.int_points*2],self.pbetween[0:self.int_points*2],self.worlds[i])
        #improving derivatives:
        #cdef np.float64_t [:] pos = np.zeros(self.int_points*2)
        #cdef np.float64_t [:] pvals = np.zeros(self.int_points*2)
        #pos[0] = self.worlds[0]-1./(self.N+1.)*2./self.pvals[0,0]
        #pos[1] = self.worlds[0]
        #pos[2:self.int_points*2] = self.pos_between[0:self.int_points*2-2]
        #pvals[0] = 0
        #pvals[1] = self.pvals[0,0]
        #pvals[2:self.int_points*2] = self.pbetween[0:self.int_points*2-2]
        #for i in range(self.int_points):
        #    self.pvals[i] = neville_p_diffs3(pos,pvals,self.worlds[i])
        
        for i in range(self.N-self.int_points,self.N):
            self.pvals[i] = neville_p_diffs3(self.pos_between[self.N-self.int_points*2:self.N],self.pbetween[self.N-self.int_points*2:self.N],self.worlds[i])
        return 0
    
from scipy.spatial import KDTree
def p_from_worlds(x,worlds):
    "convenient function to calculate density by corresponding method of interpolation"
    cdef size_t N = worlds.shape[0]
    cdef np.float64_t [:] pdiscrete = np.zeros_like(worlds)
    
    cdef size_t i
    for i in range(1.,N-1):
        pdiscrete[i] = 1./(N+1)/2.*(1./(worlds[i]-worlds[i-1])+1./(worlds[i+1]-worlds[i]))
    pdiscrete[0] = 1./(N+1)*(1./(worlds[1]-worlds[0]))*3./4.
    pdiscrete[N-1] = 1./(N+1)*(1./(worlds[N-1]-worlds[N-2]))*3./4.
    
    
    tree = KDTree(worlds[np.newaxis].T)
    cdef np.int64_t[:] nn_index = tree.query(x[np.newaxis].T)[1]
    cdef size_t Nx = x.shape[0]
    cdef np.float64_t [:] p = np.zeros_like(x)
    for i in range(Nx):
        if nn_index[i] <= 1:
            p[i] = neville(worlds[0:5],pdiscrete[0:5],x[i])
        elif nn_index[i] > N-3:
            p[i] = neville(worlds[N-5:N],pdiscrete[N-5:N],x[i])
        else:
            p[i] = neville(worlds[nn_index[i]-2:nn_index[i]+3],pdiscrete[nn_index[i]-2:nn_index[i]+3],x[i])
    return np.array(p)
           
            

    
    
    

"""
#convenient function to calculate forces quantum forces
def f_from_worlds(worlds):
    tmp = np.zeros_like(worlds)
    cdef int N = worlds.shape[0]
    tmp[0] = qmforce_left(worlds[0],worlds[1],worlds[2])
    tmp[1] = qmforce_left_inner(worlds[0],worlds[1],worlds[2],worlds[3])
    tmp[N-2] = qmforce_right_inner(worlds[N-4],worlds[N-3],worlds[N-2],worlds[N-1])
    tmp[N-1] = qmforce_right(worlds[N-3],worlds[N-2],worlds[N-1])
    cdef int j
    for j in range(N-4):
      tmp[j+2] = quantumforce(worlds[j],worlds[j+1],worlds[j+2],worlds[j+3],worlds[j+4])
    return tmp

#vectorized local potential
cdef double loc_potential_helper(double xm2,double xm1,double x,double xp1,double xp2):
    return potential_helperfunc(x,xp1,xp2)+potential_helperfunc(xm1,x,xp1)+potential_helperfunc(xm2,xm1,x)
def loc_potential(worlds):
    tmp = np.zeros_like(worlds)
    cdef int N = worlds.shape[0]
    tmp[0] = potential_helperfunc_left(worlds[0],worlds[1])+potential_helperfunc(worlds[0],worlds[1],worlds[2])
    tmp[1] = potential_helperfunc_left(worlds[0],worlds[1])+potential_helperfunc(worlds[0],worlds[1],worlds[2])+potential_helperfunc(worlds[1],worlds[2],worlds[3])
    cdef int j
    for j in range(N-4):
      tmp[j+2] = loc_potential_helper(worlds[j],worlds[j+1],worlds[j+2],worlds[j+3],worlds[j+4])
    tmp[-2] = potential_helperfunc(worlds[-4],worlds[-3],worlds[-2])+potential_helperfunc(worlds[-3],worlds[-2],worlds[-1])+potential_helperfunc_right(worlds[-2],worlds[-1])
    tmp[-1] = potential_helperfunc(worlds[-3],worlds[-2],worlds[-1])+potential_helperfunc_right(worlds[-2],worlds[-1])
    return tmp/3.
"""
