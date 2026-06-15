import numpy as np

#ising MPO, supported for open and periodic boundary condition
#H = -J \sum_i \sigma_ix \sigma_{i+1}x - h \sum_i \\sigma_iz
class MPO(object):

    def __init__(self, L, d, h, J, periodic=False):
        self.L = L
        self.d = d
        self.h = h
        self.J = J
        self.periodic = periodic

        self.sx = np.array([[0., 1.],
                            [1., 0.]])
        self.sz = np.array([[1., 0.], 
                            [0., -1.]])
        self.id = np.array([[1., 0.], 
                            [0., 1.]])
        self.W = self.construct_W()

    def construct_W(self):
        d, id, sx, sz = self.d, self.id, self.sx, self.sz
        h, J, L = self.h, self.J, self.L
        W = []
        if not self.periodic:
            #open boundary condition
            #the first W matrix, W[0]=(I, sx, -hsz)
            w = np.zeros((1, 3, d, d), dtype=complex)
            w[0, 0] = id
            w[0, 1] = sx
            w[0, 2] = - h * sz
            W.append(w)

            #the middle W matrix, W= (I    sx   -hsz\\
            #                         0     0   -Jsx\\
            #                         0     0     I)
            w = np.zeros((3, 3, d, d), dtype=complex)
            w[0, 0] = id
            w[0, 1] = sx
            w[0, 2] = -h *sz
            w[1, 2] = -J *sx
            w[2, 2] = id
            for _ in range(1,L-1):
                W.append(w.copy())
            
            #the last W matrix W=(-hsz, -Jsx, I)^T
            w = np.zeros((3, 1, d, d), dtype=complex)
            w[0, 0]= -h * sz
            w[1, 0]= -J * sx
            w[2, 0]= id
            W.append(w)

        else:
            #periodic boundary condition
            #pbc W has 4 dimesions, the first W[0]=(I, sx, -hsz, sx)
            w = np.zeros((1, 4, d, d), dtype=complex)
            w[0, 0] = id
            w[0, 1] = sx
            w[0, 2] = - h * sz
            w[0, 3] = sx
            W.append(w)

            #the middle W matrix, W= (I    sx   -hsz    0\\
            #                         0     0   -Jsx    0\\
            #                         0     0     I     0\\
            #                         0     0     0     I)
            w = np.zeros((4, 4, d, d), dtype=complex)
            w[0, 0] = id
            w[0, 1] = sx
            w[0, 2] = -h *sz
            w[1, 2] = -J *sx
            w[2, 2] = id
            w[3, 3] = id
            for _ in range(1,L-1):
                W.append(w.copy())

            ##the last W matrix W=(-hsz, -Jsx, I, -Jsx)^T
            w = np.zeros((4, 1, d, d), dtype=complex)
            w[0, 0]= -h * sz
            w[1, 0]= -J * sx
            w[2, 0]= id
            w[3, 0]= -J *sx
            W.append(w)
        
        
        return W