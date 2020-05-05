# distutils: language = c++
# distutils: extra_compile_args = -DNOBLAS

import numpy as np
cimport cython
cimport numpy as cnp
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float interp(double[:, ::1] I, int h, int w, float x, float y) nogil:
    x = 0 if x<0 else (w-1.001 if x>w-1.001 else x)
    y = 0 if y<0 else (h-1.001 if y>h-1.001 else y)
    
    cdef int x0=int(x), y0=int(y), x1=x0+1, y1=y0+1
    cdef float dx0=x-x0, dy0=y-y0, dx1=1-dx0, dy1=1-dy0
    
    return I[y0,x0]*dx1*dy1 + I[y0,x1]*dx0*dy1 + I[y1,x0]*dx1*dy0 + I[y1,x1]*dx0*dy0

ctypedef fused my_type:
    int
    double
    long long
    float

ctypedef cnp.float_t FTYPE_t
FTYPE = np.float

@cython.boundscheck(False)
@cython.wraparound(False)
def suppress(my_type[:,::1] E0, my_type[:,::1] O, int r, int s, float m, int nThreads):
    cdef Py_ssize_t h = E0.shape[0]
    cdef Py_ssize_t w = E0.shape[1]
    
    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong
    elif my_type is float:
        dtype = np.float
    
    cdef cnp.ndarray[FTYPE_t, ndim=2, mode='c'] E
    E = np.zeros((h,w),dtype=FTYPE)
    cdef cnp.ndarray[FTYPE_t, ndim=2, mode='c'] E1 
    E1 = np.array(E0,dtype=FTYPE)
    cdef cnp.ndarray[FTYPE_t, ndim=2, mode='c'] O1
    O1 = np.array(O,dtype=FTYPE)
    cdef float e, e0, coso, sino
    
    for x in range(w):
        for y in range(h):
            e=E[y,x]=E1[y,x]
            if not e:
                continue
            e*=m
            coso = float(np.cos(O1[y,x]))
            sino = float(np.sin(O1[y,x]))
            for d in range(-r,r+1):
                e0 = interp(E1,h,w,x+d*coso,y+d*sino)
                if e<e0:
                    E[y,x]=0
                    break
    
    s = int(w//2) if s>w/2 else s
    s = int(h//2) if s>h/2 else s
    
    for x in range(s):
        for y in range(h):
            E[y,x]*=x/float(s)
            E[y,w-1-x]*=x/float(s)
            
    for x in range(w):
        for y in range(s):
            E[y,x]*=y/float(s)
            E[h-1-y,x]*=y/float(s)
    
    return E
    
            
