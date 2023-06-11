
# coding: utf-8

# Some basic library stuff to be used in other Notebooks

# In[1]:

import numpy as np
import sympy as s


# In[2]:

class QMstate:
    def __init__(self, wavefunction, do_simplify=False):
        """Basic QMstate class, takes wavefunction as sympy 
        expression and calculates density and its derivatives
        Assumes psi to be real!"""
        self.psi = s.sympify(wavefunction)
        x = s.symbols('x')
        self.p = (self.psi**2).simplify()
        if(do_simplify):
            self.pdiff = s.diff(self.p, x).simplify()
            self.pdiff2 = s.diff(self.pdiff, x).simplify()
            self.pdiff3 = s.diff(self.pdiff2, x).simplify()
        else:
            self.pdiff = s.diff(self.p, x)
            self.pdiff2 = s.diff(self.pdiff, x)
            self.pdiff3 = s.diff(self.pdiff2, x)
        #functions for numerical computation
        self.psi_ = s.lambdify(x, self.psi, "numpy")
        self.p_ = s.lambdify(x, self.p, "numpy")
        self.pdiff_ = s.lambdify(x, self.pdiff, "numpy")
        self.pdiff2_ = s.lambdify(x, self.pdiff2, "numpy")
        self.pdiff3_ = s.lambdify(x, self.pdiff3, "numpy")
    def qmpot_(self,x):
        return -1./4.*(self.pdiff2_(x)*self.p_(x)-self.pdiff_(x)**2/2.)/self.p_(x)**2
    def qmforce_(self,x):
        return 1./4.*(self.pdiff3_(x)/self.p_(x)-self.pdiff2_(x)*self.pdiff_(x)/self.p_(x)**2+self.pdiff_(x)**3/self.p_(x)**3-self.pdiff2_(x)*self.pdiff_(x)/self.p_(x)**2)


# # Harmonic Oscillator

# In[3]:

class HarmonicStationaryState(QMstate):
    def __init__(self,n,a_=1):
        x, a = s.symbols("x a")
        psi = s.sympify("(a/pi)**(1/4)*1/sqrt(2**n*n!)*exp(-a/2 * x**2)")
        psi = psi*s.hermite(n,s.sqrt(a)*x)
        psi = psi.subs('n',n)
        psi = psi.subs('a',a_)
        super().__init__(psi,do_simplify=True)


# Generating initial Data

# In[4]:

import scipy
from scipy.special import erf
from sympy.parsing import mathematica
expr="""{(1 + Erf[x])/2, ((-2*x)/E^x^2 + Sqrt[Pi]*(1 + Erf[x]))/
  (2*Sqrt[Pi]), ((-4*(x + 2*x^3))/E^x^2 + 4*Sqrt[Pi]*(1 + Erf[x]))/
  (8*Sqrt[Pi]), ((16*x*(-3 + x^2 - 2*x^4))/E^x^2 + 
   24*Sqrt[Pi]*(1 + Erf[x]))/(48*Sqrt[Pi]), 
 (-((x*(15 + 34*x^2 - 20*x^4 + 8*x^6))/E^x^2) + 
   12*Sqrt[Pi]*(1 + Erf[x]))/(24*Sqrt[Pi]), 
 (-((x*(60 - 35*x^2 + 106*x^4 - 44*x^6 + 8*x^8))/E^x^2) + 
   30*Sqrt[Pi]*(1 + Erf[x]))/(60*Sqrt[Pi]), 
 (-((x*(495 + 2*x^2*(615 + 8*x^2*(-93 + 72*x^2 - 19*x^4 + 2*x^6))))/
     E^x^2) + 360*Sqrt[Pi]*(1 + Erf[x]))/(720*Sqrt[Pi]), 
 (-((x*(2520 + x^2*(-1995 + 2*x^2*(4011 + 8*x^2*(-408 + 166*x^2 - 
             29*x^4 + 2*x^6)))))/E^x^2) + 
   1260*Sqrt[Pi]*(1 + Erf[x]))/(2520*Sqrt[Pi]), 
 (-((x*(29295 + 2*x^2*(39165 + 2*x^2*(-36267 + 41718*x^2 - 
           20876*x^4 + 5368*x^6 - 656*x^8 + 32*x^10))))/E^x^2) + 
   20160*Sqrt[Pi]*(1 + Erf[x]))/(40320*Sqrt[Pi]), 
 (-((x*(181440 + x^2*(-176715 + 2*x^2*(440937 + 
           2*x^2*(-265869 + 167718*x^2 - 55140*x^4 + 9816*x^6 - 
             880*x^8 + 32*x^10)))))/E^x^2) + 
   90720*Sqrt[Pi]*(1 + Erf[x]))/(181440*Sqrt[Pi]), 
 (-(x*(2735775 + 2*x^2*(3888675 + 4*x^2*(-2439045 + 
          2*x^2*(1862865 + 8*x^2*(-169935 + 69015*x^2 - 15915*x^4 + 
              2078*x^6 - 142*x^8 + 4*x^10)))))) + 
   1814400*E^x^2*Sqrt[Pi]*(1 + Erf[x]))/(3628800*E^x^2*Sqrt[Pi]), 
 (-((x*(19958400 + x^2*(-22713075 + 2*x^2*(67494735 + 
           4*x^2*(-26909685 + 2*x^2*(11493735 + 4*x^2*(-1374645 + 
                 390360*x^2 - 66422*x^4 + 6628*x^6 - 356*x^8 + 
                 8*x^10)))))))/E^x^2) + 9979200*Sqrt[Pi]*
    (1 + Erf[x]))/(19958400*Sqrt[Pi])}"""
#convert to sympy expressions
exprlist = expr[1:-1].split(",")
exprlist = [i.strip() for i in exprlist]
#exprlist = [mathematica.parse(i) for i in exprlist]
exprlist = [i.replace("Erf[x]","erf(x)").replace("E^x^2","exp(x**2)").replace("^","**").replace("Sqrt","sqrt").replace("[","(").replace("]",")").replace("Pi","pi") for i in exprlist]
cumulativeF = [s.sympify(i) for i in exprlist]
cumulativeF_ = [s.lambdify(s.Symbol('x'), i, modules=["numpy",{'erf': scipy.special.erf}]) for i in cumulativeF]


# In[5]:

from scipy.optimize import brentq #root finding method
from scipy.special import erfinv #exact solution for ground state
def get_harmonic_worlds(n, N, useexactifpossible=False,half_sized_tails=False):
    if half_sized_tails:
        #integrate tails to 1/2*(1/N) i.e. combine tails to one interval
        #F(x_i) = (i-1/2)/(N) 
        ns = (np.arange(N)+1 -1./2.)/(N)
    else:
        ns = (np.arange(N)+1)/(N+1.)
    if(n==0 and useexactifpossible):
        return erfinv(2*ns-1) #for excat groundstate solution
    else:
        ffunc = np.vectorize(lambda y : brentq(lambda x: cumulativeF_[n](x)-y,-6,6,xtol=1e-200))
        return ffunc(ns)


# In[6]:

def get_worlds_from_cumulative(N,cumulativeF,half_sized_tails=False):
    if half_sized_tails:
        #integrate tails to 1/2*(1/N) i.e. combine tails to one interval
        #F(x_i) = (i-1/2)/(N) 
        ns = (np.arange(N)+1 -1./2.)/(N)
    else:
        ns = (np.arange(N)+1)/(N+1.)
    ffunc = np.vectorize(lambda y : brentq(lambda x: cumulativeF(x)-y,-6,6,xtol=1e-200))
    return ffunc(ns)


# Use discritezed version of integral:
# $$ \int_{x_n}^{x_{n+1}} p(x) \mathrm{d}x \approx p\left(\frac{x_{n+1}+x_n}{2}\right) (x_{n+1}-x_n) = \frac{1}{N+1}$$
# 
# so starting from $x_0$, solve
# $$x_{n+1} = \frac{1}{p\left(\frac{x_{n+1}+x_n}{2}\right)(N+1)} + x_n$$
# 

# In[49]:

def get_worlds_from_p_iteratively(N,p, startpos, search_interval_size=2.,symmetrize_its=0,verbose=False):
    ## varying starting pos recursively to get more symmetrized distribution
    if symmetrize_its > 0:
        tmp = get_worlds_from_p_iteratively(N,p,startpos,search_interval_size)
        delta = (abs(tmp[0])-abs(tmp[-1]))/2.
        return get_worlds_from_p_iteratively(N,p,startpos+delta,search_interval_size,symmetrize_its-1)
        
    else:        
        worlds = np.zeros(N,dtype=float)
        worlds[0] = startpos
        for i in range(1,N):
            if verbose:
                print("{}: last value {}".format(i,worlds[i-1]))
            worlds[i] = brentq(lambda x: 
                                1./(p((x+worlds[i-1])/2.)*(N+1.))+worlds[i-1]-x,worlds[i-1],worlds[i-1]+search_interval_size,xtol=1e-200)
        return worlds


# In[9]:

def get_double_slid_worlds(N,a,b,half_sized_tails=False):
    """Use gaussian distributions of σ=1 at a and b to calc worlds
    it is asumed 
    that a<b and
    that |b-a| is sufficiently big such that 
    overlapp of gaussians can be neglected"""
    if half_sized_tails:
        #integrate tails to 1/2*(1/N) i.e. combine tails to one interval
        #F(x_i) = (i-1/2)/(N) 
        ns = (np.arange(N)+1 -1./2.)/(N)
    else:
        ns = (np.arange(N)+1)/(N+1.)
    F = lambda x: (cumulativeF_[0](x-a) + cumulativeF_[0](x-b))/2.
    ffunc = np.vectorize(lambda y : brentq(lambda x: F(x)-y,-6+a,6+b,xtol=1e-200))
    return ffunc(ns)
    


# Free evolution gaussian wave packet

# In[2]:

def wavepacket1d(x,t,sigma=1):
    return s.sqrt(sigma)*(2/s.pi)**(s.S(1)/4)/s.sqrt(2*sigma**2+t*s.I)*s.exp(-s.S(1)/(4*sigma**2+t*s.I*2)*x**2)

