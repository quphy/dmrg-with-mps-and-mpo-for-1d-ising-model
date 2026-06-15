import numpy as np
from scipy.linalg import svd
import scipy.sparse.linalg as spm

#infinite dmrg
class iDMRG(object):
    def __init__(self, d, h, J, dim_max, tol):
        self.d = d
        self.h = h
        self.J = J
        self.dim_max = dim_max
        self.tol = tol

        self.sx = np.array([[0., 1.], 
                            [1., 0.]], dtype=complex)
        self.sz = np.array([[1., 0.], 
                            [0., -1.]], dtype=complex)
        self.id = np.array([[1., 0.], 
                            [0., 1.]], dtype=complex)
        #MPO tensor
        self.W = np.zeros((3, 3, d, d), dtype=complex)
        self.W[0, 0] = self.id
        self.W[0, 1] = self.sx
        self.W[0, 2] = -self.h * self.sz
        self.W[1, 2] = -self.J * self.sx
        self.W[2, 2] = self.id

        #boundary condition of infinite system
        self.LP = np.zeros((1, 3, 1), dtype=complex)
        self.LP[0, 0, 0] = 1.0
        self.RP = np.zeros((1, 3, 1), dtype=complex)
        self.RP[0, 2, 0] = 1.0
        
        self.E_prev = 0.0
        self.E_per_site = 0.0


    def step(self):
        diml, dimr = self.LP.shape[0], self.RP.shape[0]
        
        #construct two-site Heff
        tmpR = np.tensordot(self.W, self.RP, axes=([1], [1]))
        tmpW = np.tensordot(self.W, tmpR, axes=([1], [0]))
        Heff = np.tensordot(self.LP, tmpW, axes=([1], [0]))
        Heff = np.transpose(Heff, (0, 2, 4, 6, 1, 3, 5, 7))
        
        dim = diml * self.d * self.d *dimr
        Heff_mat = Heff.reshape(dim, dim)
        
        vguess = np.random.rand(dim) + 0j
        vguess /= np.linalg.norm(vguess)
        
        e, v = spm.eigsh(Heff_mat, k=1, which='SA', return_eigenvectors=True, v0=vguess)
        E_current = e[0]
        
        #get new left and right block
        psi = v[:, 0].reshape(diml * self.d, self.d * dimr)
        U, S, Vh = svd(psi, full_matrices=False)
        
        nonzeros = S > self.tol
        new_dim = min(self.dim_max, np.sum(nonzeros))
        
        A = U[:, :new_dim].reshape(diml, self.d, new_dim)
        B = Vh[:new_dim, :].reshape(new_dim, self.d, dimr)
        
        #update new environment blocks
        F_L = np.tensordot(self.LP, A, axes=[2, 0])
        F_L = np.tensordot(self.W, F_L, axes=([2, 0], [2, 1]))
        self.LP = np.tensordot(A.conj(), F_L, axes=([0, 1], [2, 1]))
        
        F_R = np.tensordot(B, self.RP, axes=([2], [2]))
        F_R = np.tensordot(self.W, F_R, axes=([3, 1], [1, 3]))
        self.RP = np.tensordot(B.conj(), F_R, axes=([1, 2], [1, 3]))
        
        return E_current
        
 