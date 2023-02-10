import numpy as np

# TODO : 
#    -  verifier que l'utilisation de H_inv_dot et A_dot fonctionnne bien (peut être prblème de dim, en théorie le produit dans np.fft.ifft2 doit être terme à terme et il faut padder si  besoin)
#    - Vérifier le calcul de R.T@R dans le domaine de Fourier car normalement R.T@R = np.fft.ifft2(D * np.fft.fft2()) avec D diagonale
#    - Si besoin prendre R = Id au début
#    - h opérateur de convolution, par exemple array de taille 9 pour une fenêtre 3*3 (peut être que des 1 ou des coefs d'une gausienne par ex)


class ADMMP2:

    def __init__(self, h:np.ndarray, R:np.ndarray, lambd:float, mu:np.ndarray, nu:float):

        assert nu>0, "nu doit être strictement positif."
        assert mu>0, "mu doit être strictement positif."

        self.h = h   # opérateur de convolution, permet de calculer Ax avec diag(fft2(h))
        self.R = R
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.diag_H_inv = self.diag_H(h=h,R=R)
        self.diag_A = self.diag_A(h=h)
        #self.inv_H = self._inverse(A, R)


    

    #******************************************************************************************************************
    #                                                Pré-compute
    #******************************************************************************************************************

    def _precompute_matrix(self, T:np.ndarray, y:np.ndarray, mu:np.ndarray)-> np.ndarray:
            return T.T @ y, np.linalg.inv(T.T @ T + mu * np.identity(T.shape[0]))

    @np.vectorize
    def _shrink(x:float, lambd:float)->float:
        """
        param x(float): représente une valeur à shrink
        param lambd(float): seuil de shkrink
        return : float, reprénte shrink(x) 
        """
        if -lambd<=x<=lambd:
            return 0
        else:
            return x-lambd

    @np.vectorize
    def _inv(x:float)->float:
        """
        param x(float): paramètre considéré
        return: float, renvoie 1/x si x ne vaut pas 0, 0 sinon.
        """
        return 1/x if x!=0 else 0

    def _mask_operator_generator(self, shape):
            pass
    
    def diag_H(self, h:np.ndarray)-> np.ndarray:
        """
        param h(np.ndarray):opérateur de convolution
        return: np.ndarray, pré-calcule l'inverse de la transformée de Fourier discrète. Utile pour le reste du problème.
        """
        return self._inv(np.fft.fft2(h)**2)

    def diag_R(self, R:np.ndarray)-> np.ndarray:
        """
        param R(np.ndarray):opérateur régularisation
        return: np.ndarray, pré-calcule l'inverse de la transformée de Fourier discrète. Utile pour le reste du problème.
        """
        # TODO
        pass
    def diag_A(self,h:np.ndarray)->np.ndarray:
        return np.fft.fft2(h)


    #******************************************************************************************************************
    #                                                   Utils
    #******************************************************************************************************************

    def H_inv_dot(self,x:np.ndarray)->np.ndarray:
        """
        param x(np.ndarray):paramètre considéré qui doit être multiplié par H_inv
        return: np.ndarray, multiplication en passant par le domaine de Fourier
        """
        return np.fft.ifft2(self.diag_H_inv@np.fft.fft2(x)) + self.nu*np.fft.ifft2(self.R@np.fft.fft2(x))    # TODO faire le produit terme à terme et pas le @
    
    def A_dot(self,x:np.ndarray)->np.ndarray:
        return np.fft.ifft2(self.diag_A@np.fft.fft2(x))      # TODO faire le produit terme à terme et pas le @
    


    #******************************************************************************************************************
    #                                                   ADMM-P2
    #******************************************************************************************************************

    def fit_transform(self, y:np.ndarray, eps:float=10e-2):
        T = self._mask_operator_generator(y.shape)
        pre_comput_Ty, inv_Tmu = self._precompute_matrix(T, y, self.mu)

        #inv_diag_H = self._inverse(self.h, self.R)

        eta0, eta1 = np.zeros(self.y.shape[0]),  np.zeros(self.R.shape[0])
        x = np.ones(self.y.shape[0])
        u0 = inv_Tmu * (pre_comput_Ty + self.mu * (self.A_dot(x) + eta0))
        u1 = self._shrink(self.R * x + eta1, self.lamb/(self.mu*self.nu))
        #x_ = np.fft.fft2( inv_diag_H @ np.fft.ifft2( ))
        x_ = self.H_inv_dot(self.A_dot((u0 - eta0)) + self.nu * self.R.T @ (u1 - eta1))   # transpose sur le A_dot??
        iter = 0

        while (np.linalg.norm(x_ - x) / np.linalg.norm(x))>eps:
            x = x_
            u0 = inv_Tmu * (pre_comput_Ty + self.mu * (self.A_dot(x) + eta0))
            u1 = self._shrink(self.R * x + eta1, self.lamb/(self.mu*self.nu))

            #x_ = np.fft.fft2( inv_diag_H @ np.fft.ifft2())
            x_ = self.H_inv_dot(self.A_dot(u0 - eta0) + self.nu * self.R.T @ (u1 - eta1)) # transpose sur le A_dot??

            eta0 = eta0 - u0 + self.A_dot(x_)
            eta1 = eta1 - u1 + self.R @ x_
            iter += 1
        return x_, iter


    def plot_convergence(self):
        pass