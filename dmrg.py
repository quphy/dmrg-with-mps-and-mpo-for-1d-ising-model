import numpy as np
from scipy.linalg import svd
import scipy.sparse.linalg as spm


#M = U * (S * Vh), keep U, abosrb S into next site
def left_norm(M, tol, dim_max):
    diml, d, dimr = M.shape
    Psi = M.reshape(diml * d, dimr)
    U, S, Vh = svd(Psi, full_matrices=False)
    nonzeros = S > tol
    newdim = min(dim_max, np.sum(nonzeros))
    Ut, St, Vht = U[:, :newdim], S[:newdim], Vh[:newdim, :]
    St = St * np.linalg.norm(St)**(-1)
    return Ut.reshape(diml, d, newdim), St, Vht

#M = (U * S) * Vh, keep Vh, absorb S into previous site
def right_norm(M, tol, dim_max):
    diml, d, dimr = M.shape
    Psi = M.reshape(diml, dimr * d)
    U, S, Vh = svd(Psi, full_matrices=False)
    nonzeros = S > tol
    newdim = min(dim_max, np.sum(nonzeros))
    Ut, St, Vht = U[:, :newdim], S[:newdim], Vh[:newdim, :]
    St = St * np.linalg.norm(St)**(-1)
    return Ut, St, Vht.reshape(newdim, d, dimr)

class DMRG(object):
    def __init__(self, mps, mpo, dim_max, tol=1e-10):
        #right_norm for the beginning
        mps.right_norm()
        self.mps = mps
        self.mpo = mpo
        self.L = self.mps.L
        self.tol = tol
        self.dim_max = dim_max
        self.e =[]
        #envirom tensors
        self.LPs = [None] * self.L
        self.RPs = [None] * self.L

        #match the size of boundary of MPO
        self.LPs[0] = np.zeros((self.mps.mps[0].shape[0], self.mpo.W[0].shape[0], self.mps.mps[0].shape[0]), dtype=complex)
        for a in range(self.mps.mps[0].shape[0]): 
            self.LPs[0][a, 0, a] = 1.0
            
        self.RPs[-1] = np.zeros((self.mps.mps[-1].shape[2], self.mpo.W[-1].shape[1], self.mps.mps[-1].shape[2]), dtype=complex)
        for a in range(self.mps.mps[-1].shape[2]): 
            self.RPs[-1][a, self.mpo.W[-1].shape[1] - 1, a] = 1.0
        
        for i in range(self.L - 1, 0, -1):
            self.RPs[i - 1] = self.next_RP(i)

#soleve the local eigenvalue and update site i
    def site_update(self, i):
        LP, RP, W =self.LPs[i], self.RPs[i], self.mpo.W[i]
        heff = HEFF(LP, RP, W)
        heffmat = heff.Heffmat 
        mshape = self.mps.mps[i].shape
        vguess = self.mps.mps[i].reshape(np.prod(mshape))
        e, v = spm.eigsh(heffmat, k=1, which='SA', return_eigenvectors=True, v0=vguess)
        self.e.append(e[0])
        v = v[:, 0]
        M = v.reshape(mshape)
        return M, e[0]
    
#update the left enviroment tensor    
    def next_LP(self,i):
        F = self.LPs[i]
        F = np.tensordot(F, self.mps.mps[i], axes=[2, 0])
        F = np.tensordot(self.mpo.W[i], F, axes=([2, 0], [2, 1]))
        F = np.tensordot(self.mps.mps[i].conj(), F, axes=([0, 1], [2, 1]))
        return F

#part of the sweeping: left to right
    def left_to_right(self):
        tol=self.tol
        dim_max=self.dim_max
        assert self.mps.norm == 'right_norm'
        for i in range(self.L):
            M = self.mps.mps[i]
            M,e =self.site_update(i)
            A, S, V =left_norm(M, tol, dim_max)
            self.mps.mps[i]=A

            if i< self.L -1:
                SV = np.tensordot(np.diag(S),V,1)
                self.mps.mps[i+1]= np.tensordot(SV, self.mps.mps[i + 1], 1)
                self.LPs[i+1] =self.next_LP(i)
        self.mps.norm='left_norm'

#update the right enviroment tensor 
    def next_RP(self,i):
        F=self.RPs[i]
        F = np.tensordot(self.mps.mps[i], F, axes=([2], [2]))
        F = np.tensordot(self.mpo.W[i], F, axes=([3, 1], [1, 3]))
        F = np.tensordot(self.mps.mps[i].conj(), F, axes=([1, 2], [1, 3]))
        return F

#part of the sweeping: right to lefr    
    def right_to_left(self):
        tol=self.tol
        dim_max=self.dim_max
        assert self.mps.norm == 'left_norm'
        for i in range(1,self.L+1):
            M = self.mps.mps[-i]
            M,e =self.site_update(-i)
            U, S, B =right_norm(M, tol, dim_max)
            self.mps.mps[-i]=B

            if i< self.L:
                US = np.tensordot(U, np.diag(S),1)
                self.mps.mps[-i-1]= np.tensordot(self.mps.mps[-i -1], US, 1)
                self.RPs[-i-1] =self.next_RP(-i)
        self.mps.norm='right_norm'

#constrcyt the effective hamiltonian
class HEFF(object):
    def __init__(self, LP, RP, W):
        self.RP, self.LP, self.W = RP, LP, W
        diml1, _, _ = LP.shape
        dimr1, _, _ = RP.shape
        _, _, d1, d2 = W.shape
        self.Heff = self.init_Heff()
        self.Heffmat = self.Heff.reshape(diml1 * dimr1 * d1, d2 * diml1 * dimr1)
    
    def init_Heff(self):
        temp = np.tensordot(self.W, self.RP, axes=([1], [1]))
        temp = np.tensordot(self.LP, temp, axes=([1], [0]))
        return np.transpose(temp, (0, 2, 4, 1, 3, 5))