cimport c_numpy

cdef extern from "math.h":
    double log (double x)

version_info = (1, 1)

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


def calc_dY(
        c_numpy.ndarray P,
        c_numpy.ndarray Y,
        c_numpy.ndarray num,
        c_numpy.ndarray Y2,    # if this is None calculates cost instead
        c_numpy.ndarray dY,
        ):
    
    """Calculate dY[n,d] from P[n,n] and Y[n,d] using num[n(n-1)/2] and Y2[n] as 
    work space for intermediate values.  P is assumed to be symmetrical."""
    
    cdef int d=0, n=0, w=0    # array dimension sizes
    cdef int i, j, k, tri_i   # indicies
    cdef double num_total, num_scale, p, q, b, cost
    cdef double *P_data, *num_data, *Y_data, *dY_data

    P_data = checkArrayDouble2D(P, &n, &n)
    num_data = checkArrayDouble1D(num, &w)
    Y_data = checkArrayDouble2D(Y, &n, &d)
    if dY is None:
        dY_data = NULL
    else:
        dY_data = checkArrayDouble2D(dY, &n, &d)
    Y2_data = checkArrayDouble1D(Y2, &n)
    assert w >= n*(n-1) // 2
    
    for i in range(n):
        Y2_data[i] = 0.0
        for j in range(d):
            Y2_data[i] += Y_data[i*d+j] ** 2
            if dY_data:
                dY_data[i*d+j] = 0.0
    
    num_total = 0.0
    for i in range(n):
        tri_i = i*(i-1) // 2
        for k in range(i):
            b = 0.0
            for j in range(d):
                b += Y_data[i*d+j] * Y_data[k*d+j]
            b = 1.0 / (-2.0 * b + Y2_data[i] + Y2_data[k] + 1.0)
            num_data[tri_i+k] = b
            num_total += b
    num_scale = 1.0 / (num_total * 2.0)
    
    cost = 0.0
    for i in range(n):
        tri_i = i*(i-1) // 2
        for j in range(d):
            for k in range(i):
                p = P_data[i*n+k]
                b = num_data[tri_i+k]
                q = b * num_scale
                if q < 1e-12:
                    q = 1e-12
                b *= (Y_data[i*d+j] - Y_data[k*d+j])
                b *= (p - q)
                if dY_data:
                    dY_data[i*d+j] += b
                    dY_data[k*d+j] -= b
                else:
                    cost += p * log(p / q)
    if dY is None:
        return cost
    else:  
        return dY    
