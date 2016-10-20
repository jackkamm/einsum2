import einsum2
import numpy as np

def test_matmul():
    I,J,K,L = np.random.randint(1, 4, size=4)

    A = np.random.normal(size=(I,J,K))
    B = np.random.normal(size=(I,K,L))

    assert np.allclose(einsum2.matmul(A,B), A @ B)
