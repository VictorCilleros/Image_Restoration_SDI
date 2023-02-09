import numpy as np




class ADMMP2:

    def __init__(self, A, R, lambd, mu, nu):

        assert nu>0, "nu doit être strictement positif."
        assert mu>0, "mu doit être strictement positif."

        self.A = A
        self.R = R
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.inv_H = self._inverse(A, R)


    def _mask_operator_generator(self, shape):
        pass
    
    @np.vectorize
    def _shrink(x, lambd): 
        if -lambd<=x<=lambd:
            return 0
        else:
            return x-lambd

    def _inverse(self, A, R):
        pass

    
    def _precompute_matrix(self, T, y, mu):
        return T.T @ y, np.linalg.inv(T.T @ T + mu * np.identity(T.shape[0]))


    def fit_transform(self, y, eps=10e-2):
        T = self._mask_operator_generator(y.shape)
        pre_comput_Ty, inv_Tmu = self._precompute_matrix(T, y, self.mu)
        inv_diag_H = self._inverse(self.A, self.R)
        eta0, eta1 = np.zeros(self.A.shape[0]),  np.zeros(self.R.shape[0])
        x = np.ones(self.A.shape[0])
        u0 = inv_Tmu * (pre_comput_Ty + self.mu * (self.A * x + eta0))
        u1 = self._shrink(self.R * x + eta1, self.lamb/(self.mu*self.nu))
        x_ = np.fft.fft2( inv_diag_H @ np.fft.ifft2(self.A.T @ (u0 - eta0) + self.nu * self.R.T @ (u1 - eta1)) )
        iter = 0
        while np.linalg.norm(x_ - x) / np.linalg.norm(x):
            x = x_
            u0 = inv_Tmu * (pre_comput_Ty + self.mu * (self.A * x + eta0))
            u1 = self._shrink(self.R * x + eta1, self.lamb/(self.mu*self.nu))
            x_ = np.fft.fft2( inv_diag_H @ np.fft.ifft2(self.A.T @ (u0 - eta0) + self.nu * self.R.T @ (u1 - eta1)))
            eta0 = eta0 - u0 + self.A @ x_
            eta1 = eta1 - u1 + self.R @ x_
            iter += 1
        return x_, iter


    def plot_convergence(self):
        pass