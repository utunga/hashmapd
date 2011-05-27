cimport numpy as np

def calc_dY(
        np.ndarray[np.float_t, ndim=2] P not None,
        np.ndarray[np.float_t, ndim=2] Y not None,
        np.ndarray[np.float_t, ndim=1] num not None,
        np.ndarray[np.float_t, ndim=1] Y2 not None,
        np.ndarray[np.float_t, ndim=2] dY not None):

    """For tsne.  Calculates dY[n,d] from P[n,n] and Y[n,d].
    num[n(n-1)/2] and Y2[d] are just intermediate values but they are passed 
    in as parameters so their memory can be efficiently re-used."""
    
    cdef int d, n    # array dimension sizes
    cdef int i, j, k, tri_i   # indicies
    cdef double num_total, num_scale, q, b

    n = Y.shape[0]
    d = Y.shape[1]
    assert P.shape[0] == P.shape[1] == Y.shape[0] == dY.shape[0] == Y2.shape[0] == n
    assert dY.shape[1] == Y.shape[1] == d
    assert num.shape[0] >= n*(n-1)//2
    
    for i in range(n):
        Y2[i] = 0.0
        for j in range(d):
            Y2[i] += Y[i,j] ** 2
            dY[i,j] = 0.0
    
    # num is a symmetrical square array, so stored in compact triangular layout
    num_total = 0.0
    for i in range(n):
        tri_i = i*(i-1)//2
        for k in range(i):
            b = 0.0
            for j in range(d):
                b += Y[i,j] * Y[k,j]
            b = 1.0 / (-2.0 * b + Y2[i] + Y2[k] + 1.0)
            num[tri_i+k] = b
            num_total += b
    num_scale = 1.0 / (num_total * 2.0)
        
    for i in range(n):
        tri_i = i*(i-1)//2        
        for j in range(d):
            for k in range(i):
                b = num[tri_i+k]
                q = b * num_scale
                if q < 1e-12:
                    q = 1e-12
                b *= (Y[i,j] - Y[k,j])
                dY[i,j] += (P[i,k] - q) * b
                dY[k,j] -= (P[k,i] - q) * b
    
    return dY    
    
    