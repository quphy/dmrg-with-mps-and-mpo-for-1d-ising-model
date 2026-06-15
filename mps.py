import numpy as np
from scipy.linalg import svd

#given the number of site L, dimension of each site d
#the max cut of dim_max, output the dimmension of each site
class MPS(object):
    def __init__(self, L, d, dim_max ):         
        self.L = L
        self.d = d
        self.dim_max = dim_max
        self.bond_dim = self.vbond_dim()
        self.mps = self.random_MPS()
        self.norm = None

    def vbond_dim(self):
        L, d, dim_max =self.L, self.d, self.dim_max
        a = np.ones(L+1, dtype=int)
        for i in range(int(L/2) + 1):
            a[i] = a[L-i] = min(d**i, dim_max)
        return a

#construct a random MPS tensors
    def random_MPS(self):
        L, d = self.L, self.d
        dim = self.bond_dim
        MPS= []
        for i in range(L):
            tensor = np.random.rand(dim[i], d, dim[i+1])
            MPS.append(tensor)
        return MPS

#perform SVD decomposition on the one-dimensional MPS from right to left 
#and rewrite each tensor in right-canonical form.
    def right_norm(self):
        L, d, mps = self.L, self.d, self.mps
        Bs = []
        for i in range(L-1, -1, -1):
            dim1, d, dim2 = mps[i].shape
            m = mps[i].reshape(dim1, d*dim2)
            U, S, Vh =svd(m, full_matrices=False)
            newdim = len(S)
            B = Vh.reshape(newdim,d,dim2)
            Bs.append(B)
            #transfer
            if i >0:
                US = U @ np.diag(S)
                mps[i - 1] = mps[i - 1]@ US
            
        self.mps= Bs[::-1]
        self.norm ='right_norm'

#perform SVD decomposition on the one-dimensional MPS from left to right
#and rewrite each tensor in left-canonical form.
    def left_norm(self):
        L, d, mps = self.L, self.d, self.mps
        As=[]
        for i in range(L):
            dim1, d, dim2 = mps[i].shape
            m = mps[i].reshape(dim1, d*dim2)
            U, S, Vh =svd(m, full_matrices=False)
            newdim = len(S)
            A = Vh.reshape(newdim,d,dim2)
            As.append(A)

            if i < (L-1):
                SV= np.diag(S)@Vh
                mps[i+1]= SV @ mps[i+1]

        self.mps = As
        self.norm = 'left_norm'
             

    



