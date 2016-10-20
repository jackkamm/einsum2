cimport numpy as np
import numpy as np
import cython

def parallel_matmul(np.ndarray[np.double_t, ndim=3] A,
                    np.ndarray[np.double_t, ndim=3] B):
    if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
        raise ValueError("Invalid dimensions for matmul")

    cdef np.ndarray[np.double_t, ndim=3] C = np.zeros((A.shape[0],
                                                       A.shape[1],
                                                       B.shape[2]))

    cdef unsigned int i,j,k,l,I,J,K,L
    I,J,K,L = A.shape[0], A.shape[1], A.shape[2], B.shape[2]

    for i in range(I):
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    C[i,j,l] += A[i,j,k] * B[i,k,l]

    return C
