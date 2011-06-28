cdef extern from "Python.h":
    void *PyMem_Malloc(int)
    void PyMem_Free(void *)

cimport c_numpy

cdef extern from "math.h":
    double log (double x)

version_info = (2, 0)

cdef void *checkArray(c_numpy.ndarray a, char typecode, int itemsize, 
        int nd, int **dims) except NULL:
    cdef int length, size
    cdef char kind
    if a is None:
        raise TypeError("Array required, got None")
    #if a.version != 2:
    #    raise ValueError(
    #        "Unexpected array interface version %s" % str(a.version))
    cdef char typecode2
    typecode2 = a.descr.kind
    if typecode2 != typecode:
        raise TypeError("'%s' type array required, got '%s'" %
                (chr(typecode), chr(typecode2)))
    if a.itemsize != itemsize:
        raise TypeError("'%s%s' type array required, got '%s%s'" %
                (chr(typecode), itemsize, chr(typecode2), a.itemsize))
    if a.nd != nd:
        raise ValueError("%s dimensional array required, got %s" %
                (nd, a.nd))
    if not a.flags & c_numpy.NPY_CONTIGUOUS:
        raise ValueError ('Noncontiguous array')
        
    cdef int dimension, val
    cdef int *var
    for dimension from 0 <= dimension < nd:
        val = a.shape[dimension]
        var = dims[dimension]
        if var[0] == 0:
            # Length unspecified, take it from the provided array
            var[0] = val
        elif var[0] != val:
            # Length already specified, but not the same
            raise ValueError("Dimension %s is %s, expected %s" %
                    (dimension, val, var[0]))
        else:
            # Length matches what was expected
            pass
    return a.data
    
    
cdef void *checkArray1D(c_numpy.ndarray a, char typecode, int size, 
        int *x) except NULL:
    cdef int *dims[1]
    dims[0] = x
    return checkArray(a, typecode, size, 1, dims)
            
cdef void *checkArray2D(c_numpy.ndarray a, char typecode, int size, 
        int *x, int *y) except NULL:
    cdef int *dims[2]
    dims[0] = x
    dims[1] = y
    return checkArray(a, typecode, size, 2, dims)

cdef void *checkArray3D(c_numpy.ndarray a, char typecode, int size, 
        int *x, int *y, int *z) except NULL:
    cdef int *dims[3]
    dims[0] = x
    dims[1] = y
    dims[2] = z
    return checkArray(a, typecode, size, 3, dims)
            
    
cdef double * checkArrayDouble1D(c_numpy.ndarray a, int *x) except NULL:
    return <double *> checkArray1D(a, c'f', sizeof(double), x)
    
cdef double * checkArrayDouble2D(c_numpy.ndarray a, int *x, int *y) except NULL:
    return <double *> checkArray2D(a, c'f', sizeof(double), x, y)



cdef class TSNE(object):
    cdef int _n, _d
    cdef double *_P, *_num, *_Y2
    
    def __init__(self, c_numpy.ndarray P, int d=2):
        """Stores the symmetrical square matrix P[n,n] as compact triangular array
        and allocates working space sufficient to calculate dY[n,d] from Y[n,d]"""
        
        cdef int n=0, w   # array dimension sizes
        cdef int i, k, tri_i   # indicies
        cdef double *P_data, p, p_sum
        P_data = checkArrayDouble2D(P, &n, &n)
        self._n = n
        self._d = d
        w = n*(n-1)//2
        self._P = <double *> PyMem_Malloc(w*sizeof(double))
        self._num = <double *> PyMem_Malloc(w*sizeof(double))
        self._Y2 =  <double *> PyMem_Malloc(n*d*sizeof(double))
        p_sum = 0.0
        for i in range(1, n):
            tri_i = i*(i-1)//2
            for k in range(i):
                self._P[tri_i+k] = p = P_data[i*n+k] + P_data[k*n+i]
                p_sum += 2*p
        self.scaleP(1.0 / p_sum)
        
    def __dealloc__(self):
        if self._P:
            PyMem_Free(self._P)
        if self._num:
            PyMem_Free(self._num)
        if self._Y2:
            PyMem_Free(self._Y2)
    
    def scaleP(self, double scale, double min = 0.0):
        cdef int w
        for w in range(self._n*(self._n-1)//2):
            self._P[w] *= scale
            if self._P[w] < min:
                self._P[w] = min
    
    def calc_KL(self, Y):
        """Calculate KL divergence"""
        return self.calc_dY(Y, None)
    
    def calc_dY(self, c_numpy.ndarray Y, c_numpy.ndarray dY):
        """Calculate dY[n,d] from Y[n,d]"""
        
        cdef int n, d, w   # array dimension sizes
        cdef int i, j, k, tri_i   # indicies
        cdef double num_total, num_scale, p, q, b, y2, dy, cost
        cdef double *P_data, *num_data, *Y_data, *dY_data
    
        n = self._n
        d = self._d
        w = n*(n-1)//2
        P_data = self._P
        num_data = self._num
        Y2_data =  self._Y2
        Y_data = checkArrayDouble2D(Y, &n, &d)
        if dY is None:
            dY_data = NULL
        else:
            dY_data = checkArrayDouble2D(dY, &n, &d)

        for i in range(n):
            Y2_data[i] = 0.0
            for j in range(d):
                Y2_data[i] += Y_data[i*d+j] ** 2
                if dY_data:
                    dY_data[i*d+j] = 0.0

        num_total = 0.0
        for i in range(1, n):
            tri_i = i*(i-1) // 2
            for k in range(i):
                y2 = Y2_data[i] + Y2_data[k] + 1.0
                b = 0.0
                for j in range(d):
                    b += Y_data[i*d+j] * Y_data[k*d+j]
                b = 1.0 / (-2.0 * b + y2)
                num_data[tri_i+k] = b
                num_total += b
        num_scale = 1.0 / (num_total * 2.0)
        
        for i in range(1, n):
            tri_i = i*(i-1) // 2        
            for k in range(i):
                p = P_data[tri_i+k]
                b = num_data[tri_i+k]
                q = b * num_scale
                if q < 1e-12:
                    q = 1e-12
                if dY_data:
                    b *= (p - q)
                    for j in range(d):
                        dy = b * (Y_data[i*d+j] - Y_data[k*d+j])
                        dY_data[i*d+j] += dy
                        dY_data[k*d+j] -= dy         
                else:
                    cost += d * p * log(p / q)

        if dY is None:
            return cost
        else:  
            return dY
            
