"""This file contains basic Ground State algorithm class MIWBase, which implements basic update structure with buffer,
 and output functions and automatically checks for crossings."""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs, sqrt

cdef inline double delta(int i, int j):
    if i==j:
        return 1.
    else:
        return 0.
cdef class MIWBase:
    cdef int N
    cdef double t
    cdef double dt
    #holds world configurations without boundary
    cdef np.float64_t[:] worlds
    cdef np.float64_t[:] worlds_buffer
    cdef np.int_t [:] filter_indices # if 1 -> do update, if 0 -> keep worlds fixed
    property worlds:
        def __get__(self):
            return np.asarray(self.worlds)
        def __set__(self,worlds):
            self.worlds = worlds
    property worlds_buffer:
        def __get__(self):
            return np.asarray(self.worlds_buffer)
    property filter_indices:
        def __get__(self):
            return np.asarray(self.filter_indices)
        def __set__(self, filter_indices):
            self.filter_indices = filter_indices.copy()
            
    def __init__(self,initial_config,filter_indices=None):
        self.t=0.
        self.worlds = initial_config.copy()
        self.worlds_buffer = initial_config.copy()
        if filter_indices is not None:
            self.filter_indices = filter_indices.copy()
        else:
            self.filter_indices = np.ones_like(initial_config, dtype=np.int)
        self.N = initial_config.shape[0]
    cdef double get_max_dt(self):
        """calculates maximal dt value such that no crossings occurs
            should be owerwritten in class for specfic model"""
        cdef double tmp_maxforce = self.get_max_force()
        cdef double tmp_mindist = self.worlds[self.N-1] - self.worlds[0]
        cdef double dist = 0.
        cdef size_t i
        if tmp_maxforce<=0.:
            return 0.
        else:        
            for i in range(self.N-1):
                dist = self.worlds[i+1]-self.worlds[i]
                if dist < tmp_mindist:
                    tmp_mindist = dist
            return sqrt(dist/2./tmp_maxforce)
            
        
    cdef double get_max_force(self):
        """calculates maximal force value
            should be owerwritten in class for specfic model"""
        return 0.
    cdef int exist_crossings(self):
        """returns 1 if there is a crossing, 0 otherwise"""
        cdef int i
        for i in range(self.N-1):
            if (self.worlds_buffer[i+1]-self.worlds_buffer[i]<0):
                return 1
        return 0
    cdef flush_worlds_buffer(self):
        cdef int i
        for i in range(self.N):
            if self.filter_indices[i]==1:
                self.worlds[i]=self.worlds_buffer[i]
        self.t = self.t + self.dt**2
        
    cdef double _step(self):
        """exectues one integration step and writes result to worlds_buffer
            without checking crossings"""
        #example integration procedure
        cdef int i 
        for i in range(self.N):
            self.worlds_buffer[i]=self.worlds[i]+self.dt
    
    cdef double get_energy(self):
        return 0.
    cpdef get_output_p(self):
        """returns array of p values at worlds for output error calculation"""
        return 0.
    cpdef get_Qloc(self):
        """returns local quantum potential values for error calculation"""
        return 0.
    def update_minimizer(self, int NumIterations=1, double control_factor = 0.9):
        cdef size_t i,j
        cdef np.float64_t[:] tmp_worlds
        cdef np.float64_t[:] tmp_force
        cdef double dt
        cdef double t
        cdef double tmp_energy
        cdef double force_squared
        tmp_worlds = np.zeros(self.N,dtype=np.float64)
        tmp_force = np.zeros(self.N, dtype=np.float64)
        output_energy = []
        output_t = []
        for i in range(NumIterations):
            tmp_energy = self.get_energy()
            t = self.t
            ##make copy of worlds
            for j in range(self.N):
                tmp_worlds[j] = self.worlds[j]
            
            #extract force (THIS IS A HACK, new interface neccesary)!
            force_squared = 0.
            self._rawupdate(1.)
            for j in range(self.N):
                tmp_force[j] = self.worlds_buffer[j]-self.worlds[j]
                force_squared = tmp_force[j]**2
            
            #start with maximal dt
            dt = self.get_max_dt()
            # use backtracking condition to determine maximal dt
            while self.get_energy()-tmp_energy> - force_squared*dt**2*control_factor:
                for j in range(self.N):
                    self.worlds_buffer[j] = tmp_worlds[j] + dt**2* tmp_force[j]
                self.t = t #write old value before update by flush
                self.dt = dt #write new dt value befor updat of t by flush
                self.flush_worlds_buffer()
                #halbiere dt falls für den fall das ba
                dt = dt/2.
            output_energy.append(self.get_energy())
            output_t.append(self.t)
        return np.array(output_t), np.array(output_energy)
            
            
            
            
           
    cpdef _rawupdate(self, double dt):
        self.dt = dt
        self._step()
            
        
    def update(self, double dt ,int NumIterations=1,int OutputEvery=1, check_crossing=True,double flex_factor_dt = 0., double flex_max_dt = 1e-1):
        """updates world configuration by time step dt and repeats this for <NumIterations>,
            returns a numpy array of world configuartion every <OutputEvery>
            Automatically detects crossing and repeats step with dt/2. If this happends <crossing_threshold> times
            it decreases dt to dt/2 for the following steps"""
        #crossing detection
        cdef int crossings = 0
        cdef int check_crossing_ = check_crossing
        #initialize output
        output_worlds = [np.asarray(self.worlds).copy()]
        output_time = [self.t]
        output_energy = [self.get_energy()]
        output_pvals = [self.get_output_p()]
        output_Qloc_vals = [self.get_Qloc()]
        #doing iterations
        self.dt = dt
        cdef int i
        cdef double tmp_dt = 0.
        for i in range(NumIterations):
            if flex_factor_dt >0.:
                tmp_dt = self.get_max_dt()
                if tmp_dt > 0.:
                    self.dt = tmp_dt *flex_factor_dt
                    if self.dt > flex_max_dt:
                        self.dt = flex_max_dt
            self._step()
            if check_crossing_:
                while self.exist_crossings(): #crossing detection
                    crossings +=1
                    if crossings > 10: break
                    self.dt = self.dt/2.
                    print("Warning: Crossing Detected at N={} t={}, change dt to dt/2 ={}".format(i,self.t,self.dt))
                    self._step()
                if crossings > 10:
                    print("FAIL: Too many crossings detected")
                    break
            self.flush_worlds_buffer()
            if (NumIterations>=10 and i%(NumIterations/10))==0:
                print("{}% ".format(i*100/NumIterations)),
            if (i%OutputEvery)==0 :
                output_worlds.append(np.asarray(self.worlds).copy())
                output_time.append(self.t)
                output_energy.append(self.get_energy())
                output_pvals.append(self.get_output_p())
                output_Qloc_vals.append(self.get_Qloc())
        print ("100%")
        output_worlds.append(np.asarray(self.worlds).copy())
        output_time.append(self.t)
        output_energy.append(self.get_energy())
        output_pvals.append(self.get_output_p())
        output_Qloc_vals.append(self.get_Qloc())
        
        return np.array(output_worlds), np.array(output_time) , np.array(output_energy), np.array(output_pvals), np.array(output_Qloc_vals)

cdef inline np.float64_t f_bohm_force(np.float64_t[:] pvals):
    return -((2*pvals[0]*pvals[2]-pvals[1]**2)*pvals[1]-pvals[0]**2*pvals[3])/(4*pvals[0]**3)
cdef inline np.float64_t f_bohm_pot(np.float64_t[:] pvals):
    return -((2*pvals[0]*pvals[2]-(pvals[1])**2)/(8*pvals[0]**2))
cdef inline np.float64_t f_energie_pot(np.float64_t[:] pvals):
    return (pvals[1]**2)/(8*pvals[0]**2)


cdef class MIWBase_pval(MIWBase):
    """handles p_values and derivates as extra arrays, and implements Bohmforce + Qmpotential
    in order to use this class you have to overwrite int _update_pvals() function, Bohmforce etc is carried out automatically"""
    #p[i] = pvals[i,0] pdiff[i] = pvals[i,1] ... pdiff3[i]=pvals[i][3]
    cdef np.float64_t[:,:] pvals
    cdef size_t fixed_boundaries
    #cdef int infintiy_boundaries
    property pvals:
        def __get__(self):
            return np.asarray(self.pvals)
    property p:
        def __get__(self):
            return np.asarray(self.pvals[:,0])
    property pdiff:
        def __get__(self):
            return np.asarray(self.pvals[:,1])
    property pdiff2:
        def __get__(self):
            return np.asarray(self.pvals[:,2])
    property pdiff3:
        def __get__(self):
            return np.asarray(self.pvals[:,3])
    property force:
        def __get__(self):
            tmp = np.zeros(self.N)
            cdef size_t i
            for i in range(self.N):
                tmp[i]= F(self.worlds[i])+f_bohm_force(self.pvals[i])
            return tmp 
    def __init__(self,worlds,filter_indices):
        MIWBase.__init__(self,worlds,filter_indices)
        #initialize p arrays
        self.pvals = np.zeros((worlds.shape[0],4),dtype=np.float64)
        #self.infinity_boundary = infinty_boundary
        self._update_pvals()
        
    
    cdef int _update_pvals(self):
        """updates all pvalues befor force calculation should be overloaded for calculation"""
        return 0

    cdef double _step(self):
        """exectues one integration step and writes result to worlds_buffer
            without checking crossings"""
        cdef int i 
        for i in range(self.fixed_boundaries,self.N-self.fixed_boundaries):
            #update worlds with bohmforce and pvals
            self.worlds_buffer[i]=self.worlds[i]+self.dt**2*(F(self.worlds[i]) + f_bohm_force(self.pvals[i]))
    
    cdef flush_worlds_buffer(self):
        MIWBase.flush_worlds_buffer(self)
        self._update_pvals()
    cdef double get_max_force(self):
        """calculates maximal force value"""
        cdef double tmp_maxforce = 0.
        cdef double tmp_force = 0.
        cdef size_t i 
        for i in range(self.fixed_boundaries,self.N-self.fixed_boundaries):
            tmp_force = fabs(f_bohm_force(self.pvals[i])+F(self.worlds[i]))
            if tmp_force > tmp_maxforce:
                tmp_maxforce = tmp_force
        return tmp_maxforce

    cdef double get_energy(self):
        cdef double tmp = 0.
        cdef int i
        for i in range(self.N):
            tmp += V(self.worlds[i])+f_bohm_pot(self.pvals[i])
        return tmp/self.N
    
    cpdef get_output_p(self):
        return np.asarray(self.pvals[:,0])
    cpdef get_Qloc(self):
        tmp = np.zeros(self.N,dtype=np.float64)
        cdef size_t i
        for i in range(self.N):
            tmp[i] = f_bohm_pot(self.pvals[i])
        return tmp

def calc_perror_from_res(refstate, res, method='pot'):
    if method=='diff':
        """ calc sum_i |Pexact(x_i) - Pmodel(x_i)| /N """
        return np.sum(np.abs(refstate.p_(res[0])-res[3]),axis=1)/res[0].shape[1]
    if method=='frac':
        """ calc sum_i |Pmodel(x_i)/Pexact(x_i) - 1|/N """
        return np.sum(np.abs(res[3]/refstate.p_(res[0]) - 1), axis=1)/res[0].shape[1]
    if method=='pot':
        """ calc sum_i |Qpot_exact(x_i)-Q_bohm(x_i)|/N """
        return np.sum(np.abs(res[4]-refstate.qmpot_(res[0])),axis=1)/res[0].shape[1]
