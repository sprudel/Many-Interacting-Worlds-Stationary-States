"""This file implements the Groundstate Algorthm for original MIW Algorithm as in the original paper"""
include "MIWBase.pxi"
include "PotentialForce.pxi"

#not consistent in code, has to be checked if changed  in whole code
cdef double hbar =1
cdef double m = 1

# Quantumforce as defined in the paper, boundary terms were calculated by respective limits to +- infinity
cdef inline double quantumforce(double xm2,double xm1,double x,double xp1,double xp2):
    return hbar**2/(4*m*(xp1 - x)**2)*(1/(xp2 - xp1) - 2/(xp1 - x) + 1/(x - xm1))- hbar**2/(4*m*(x - xm1)**2)*(1/(xp1 - x) - 2/(x - xm1) + 1/(xm1 - xm2))
cdef inline double qmforce_left(double x,double xp1,double xp2):
    return -(x - 3*xp1 + 2*xp2)/(4*(x - xp1)**3*(xp1 - xp2))
cdef inline double qmforce_left_inner(double xm1,double x,double xp1,double xp2):
    return  (-x**4 + x*(xm1**3 + xm1**2*(10*xp1 - 7*xp2) + 4*xm1*xp1*(xp1 - xp2) + 7*xp1**2*(xp1 - xp2)) + x**3*(3*xm1 + 7*xp1 - 6*xp2) - 3*x**2*(xm1 + xp1)*(xm1 + 3*xp1 - 3*xp2) + xm1**2*xp1*(-xp1 + xp2) + xm1*xp1**2*(-xp1 + xp2) + 2*xp1**3*(-xp1 + xp2) + xm1**3*(-3*xp1 + 2*xp2))/(4*(x - xm1)**3*(x - xp1)**3*(xp1 - xp2))
cdef inline double qmforce_right_inner(double xm2,double xm1,double x,double xp1):
    return (-x**4 - 2*xm1**4 + xm1**3*(2*xm2 - xp1) + xm1**2*(xm2 - xp1)*xp1 + xm1*(xm2 - 3*xp1)*xp1**2 + 2*xm2*xp1**3 - 3*x**2*(xm1 + xp1)* (3*xm1 - 3*xm2 + xp1) + x**3*(7*xm1 - 6*xm2 + 3*xp1) +  x*(7*xm1**3 + xp1**2*(-7*xm2 + xp1) + xm1**2*(-7*xm2 + 4*xp1) +  2*xm1*xp1*(-2*xm2 + 5*xp1)))/(4*(x - xm1)**3*(xm1 - xm2)*(x - xp1)**3)
cdef inline double qmforce_right(double xm2,double xm1,double x):
    return -((x - 3*xm1 + 2*xm2)/(4 *(x - xm1)**3 *(xm1 - xm2)))
# Part of Potential to calculate Empirical Energy
cdef inline double potential_helperfunc(double xm1, double x, double xp1):
    return (1./(xp1-x)-1./(x-xm1))**2
cdef inline double potential_helperfunc_left(double x, double xp1):
    return (1./(xp1-x)**2)
cdef inline double potential_helperfunc_right(double xm1, double x):
    return (-1./(x-xm1))**2
 
cdef class MIWorig(MIWBase):
    cdef np.float64_t[:] _pvals #save intermediate discrete pvals
    cdef np.float64_t[:] _fvals #save intermediate force vals calculated from pvals    
    cdef np.int_t[:] phase #save "phase" = +-1 -> 0;pi

    

    property p:
        def __get__(self):
            return p_from_worlds(np.array(self.worlds),np.array(self.worlds))
    property force:
        def __get__(self):
            return f_from_worlds(np.array(self.worlds))
    def __init__(self,worlds,filter_indices=None,phase=None):
        self._pvals = np.zeros(worlds.shape[0]+3)
        self._fvals = np.zeros(worlds.shape[0])
        MIWBase.__init__(self,worlds,filter_indices=filter_indices,)
        if phase is not None:
            self.phase = np.array(phase, dtype=np.int)
        else:
            self.phase = np.ones(self.N, dtype=np.int)
        self._update_pvals()
        self._update_fvals()
    cdef _update_pvals(self):
        cdef size_t i
        for i in range(2,self.N+1):
            self._pvals[i] = 1./(self.worlds[i-1]-self.worlds[i-2])*fabs((self.phase[i-1]+self.phase[i-2]))/2.
        #boundary cases
        ## p0minus = 0, pN+2minus = 0 -> do nothing already set to zero
 
    cdef _update_fvals(self):
        cdef size_t i
        for i in range(self.N):
            self._fvals[i] = (hbar**2/(4*m)*self._pvals[i+2]**2*(self._pvals[i+3]- 2*self._pvals[i+2] + self._pvals[i+1])-hbar**2/(4*m)*self._pvals[i+1]**2*(self._pvals[i+2]- 2*self._pvals[i+1] + self._pvals[i]))#*self.phase[i]
    cdef double get_max_force(self):
        """calculates maximal force value"""
        cdef double tmp_maxforce = 0.
        cdef size_t i 
        for i in range(self.N):
            if fabs(self._fvals[i] + F(self.worlds[i] )*self.filter_indices[i]> tmp_maxforce):
                tmp_maxforce = fabs(self._fvals[i] + F(self.worlds[i]))*self.filter_indices[i]
        return tmp_maxforce
    cdef double _step(self):
        self._update_pvals()
        self._update_fvals()
        #update all worlds
        cdef int j
        for j in range(self.N):
            self.worlds_buffer[j] = self.worlds[j] + self.dt**2/(2.)*(F(self.worlds[j])+self._fvals[j])
    cdef flush_worlds_buffer(self):
        MIWBase.flush_worlds_buffer(self)
        self._update_pvals()
        self._update_fvals()
    def p_eval_vec(self,x):
        """Calculates density estimate at arbitrary ponits given the world configuration"""
        return p_from_worlds(x,np.array(self.worlds),np.array(self.phase))
    cpdef double get_energy(self):
        cdef double tmp = 0
        cdef size_t i
        for i in range(self.N):
            tmp += V(self.worlds[i]) + (self._pvals[i+2]-self._pvals[i+1])**2*hbar**2/(8*m)
        return tmp/self.N
    cpdef get_output_p(self):
        cdef np.float64_t[:] output_p
        output_p = np.zeros(self.N)
        cdef size_t i
        for i in range(self.N):
            output_p[i] = (self._pvals[i+2]+self._pvals[i+1])/(2.*self.N)
        return output_p
    cpdef get_Qloc(self):
        tmp = np.zeros(self.N)
        cdef size_t i
        for i in range(self.N):
            tmp[i] = (self._pvals[i+2]-self._pvals[i+1])**2*hbar**2/(8*m)
        for i in range(self.N-1):
            tmp[i] = (self._pvals[i+3]-self._pvals[i+2])**2*hbar**2/(8*m)
        for i in range(self.N-1):
            tmp[i+1]= (self._pvals[i+2]-self._pvals[i+1])**2*hbar**2/(8*m)
        for i in range(self.N):
            tmp[i] /= 3.
        return tmp
        

### Calculate density from world distribution approximately
#height $h$ given by:
#$$ h(x) = \underbrace{\frac{1}{N+1} \frac{1}{|x_{n+1} - x_n|} }_{h_n} \text{  for } x\in (x_n,x_{n+1})$$
#$$ h(x_n)= 1/2(h_n+h_{n-1})$$

def p_from_worlds(x,worlds, phase=None):
    N = worlds.shape[0]
    if phase is None:
        phase = np.ones_like(worlds)
    p= np.zeros_like(x)
    boundary_left = worlds.min()
    boundary_right = worlds.max()
    supp_p = (x>=boundary_left)&(x<boundary_right) #index where p non-zero
    h = np.zeros_like(worlds)
    h[:-1] = 1/np.abs(np.diff(worlds))/(N)*np.abs(phase[1:]+phase[:-1])/2. #heights, see formula above, h[N-1]=0 here for boundary calculations
    interval_index = np.digitize(x[supp_p], worlds) -1 #index of interval between worlds
    p[supp_p]=h[interval_index]
    indexof_x_equal_worlds, indexof_worlds_equal_x = np.where(x[:,np.newaxis]==worlds) #looking for x values equal some world
    p[indexof_x_equal_worlds] = (h[indexof_worlds_equal_x]+h[indexof_worlds_equal_x-1])/2
    return p

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

