import numpy as np
import plotly.graph_objects as go
import time

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
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.diag_H_inv = self.fdiag_inv(h=h)
        self.A_fft = self.fft(arr=h)
        self.R_fft = self.fft(arr=R)
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
        return : float, renvoie 1/x si x ne vaut pas 0, 0 sinon.
        """
        return 1/x if x!=0 else 0

    def _mask_operator_generator(self, x, y):
            T = np.identity(x.shape[0]*x.shape[1])
            a = (x.shape[0] - y.shape[0]) // 2
            b = (x.shape[1] - y.shape[1]) // 2 
            
            for i in range(a):
                index_list = index_list + [i*x.shape[0] + j for j in range(x.shape[0])]
            for i in range(x.shape[0] -a, x.shape[0]):
                index_list = index_list + [i*x.shape[0] + j for j in range(x.shape[0])]
            for j in range(b):
                index_list = index_list + [x.shape[0]*i + j for i in range(a, x.shape[0] - a)]
            for j in range(x.shape[1] - b, x.shape[1]):
                index_list = index_list + [x.shape[0]*i + j for i in range(a, x.shape[0] - a)]
                
            T = np.delete(T, set(index_list), axis=0)
            
            return T
    
    def fdiag_inv(self, h:np.ndarray)-> np.ndarray:
        """
        param h(np.ndarray):opérateur de convolution
        return : np.ndarray, pré-calcule l'inverse de la transformée de Fourier discrète. Utile pour le reste du problème.
        """
        return self._inv(np.fft.fft2(h)**2)

    def fft(self,arr:np.ndarray)->np.ndarray:
        """
        param h(np.ndarray):opérateur considéré
        return : np.ndarray, pré-calcule la transformée de Fourier discrète. Utile pour le reste du problème.
        """
        return np.fft.fft2(arr)


    #******************************************************************************************************************
    #                                                   Utils
    #******************************************************************************************************************

    def H_inv_dot(self,x:np.ndarray)->np.ndarray:
        """
        param x(np.ndarray):paramètre considéré qui doit être multiplié par H_inv
        return : np.ndarray, multiplication en passant par le domaine de Fourier
        """
        return np.fft.ifft2(self.diag_H_inv@np.fft.fft2(x)) + self.nu*np.fft.ifft2(self.R_fft@np.fft.fft2(x))    # TODO faire le produit terme à terme et pas le @
    
    def A_dot(self,x:np.ndarray)->np.ndarray:
        return np.fft.ifft2(self.A_fft@np.fft.fft2(x))      # TODO faire le produit terme à terme et pas le @

    def A_dot_T(self,x:np.ndarray)->np.ndarray:
        return np.fft.ifft2(self.A_fft.T@np.fft.fft2(x))      # TODO faire le produit terme à terme et pas le @

    def R_dot(self,x:np.ndarray)->np.ndarray:
        return np.fft.ifft2(self.R_fft@np.fft.fft2(x))      # TODO faire le produit terme à terme et pas le @

    def R_dot_T(self,x:np.ndarray)->np.ndarray:
        return np.fft.ifft2(self.R_fft.T@np.fft.fft2(x))      # TODO faire le produit terme à terme et pas le @
    


    #******************************************************************************************************************
    #                                                   ADMM-P2
    #******************************************************************************************************************

    def fit_transform(self, y:np.ndarray, eps:float=10e-2):
        T = self._mask_operator_generator(y.shape)
        pre_comput_Ty, inv_Tmu = self._precompute_matrix(T, y, self.mu)

        eta0, eta1 = np.zeros(self.y.shape[0]),  np.zeros(self.R_fft.shape[0])
        x = np.ones(self.y.shape[0])
        u0 = inv_Tmu * (pre_comput_Ty + self.mu * (self.A_dot(x) + eta0))
        u1 = self._shrink(self.R_dot(x) + eta1, self.lamb/(self.mu*self.nu))
        x_ = self.H_inv_dot(self.A_dot((u0 - eta0)) + self.nu * self.R_dot(u1 - eta1))   # transpose sur le A_dot??
        iter = 0

        #paramètres pour plot la convergence : 
        tabError = []
        tabTime = []
        timeRef= time()
        err = np.linalg.norm(x_ - x) / np.linalg.norm(x)

        while (err)>eps:
            x = x_
            u0 = inv_Tmu * (pre_comput_Ty + self.mu * (self.A_dot(x) + eta0))
            u1 = self._shrink(self.R_dot(x) + eta1, self.lamb/(self.mu*self.nu))

            x_ = self.H_inv_dot(self.A_dot(u0 - eta0) + self.nu * self.R_dot(u1 - eta1)) # transpose sur le A_dot??

            eta0 = eta0 - u0 + self.A_dot(x_)
            eta1 = eta1 - u1 + self.R_dot(x_)
            iter += 1

            tabError.append(err)
            err = np.linalg.norm(x_ - x) / np.linalg.norm(x)
            tabTime.append(time()-timeRef)
            
        return x_, iter,tabError,tabTime

#******************************************************************************************************************
#                                                   Affichages -- PLOTLY
#******************************************************************************************************************

    def plot_convergence_iter(self,tabError : np.ndarray,tabTime : np.ndarray=None)->None:
        
        if tabTime is not None:
            x=tabTime
        else:
            nIter = np.shape(tabError)[0]
            x = np.linspace(0,nIter,nIter+1)

        fig = go.Figure(data=go.Scatter(x = x, y = tabError))
        fig.update_layout(
                    title=go.layout.Title(
                    text="Evolution de l'erreur <br><sup>Algorithme ADMMP2 </sup>",
                    xref="paper",
                    x=0
                        ),
                    xaxis=go.layout.XAxis(
                        title=go.layout.xaxis.Title(
                            text="nombre d'itérations",
                            font=dict(
                                family="Courier New, monospace",
                                size=18,
                                color="#7f7f7f"
                                    )
                                )
                        ),
                    yaxis=go.layout.YAxis(
                            title=go.layout.yaxis.Title(
                            text="Erreur en norme sur l'image",
                            font=dict(family="Courier New, monospace",size=18,color="#7f7f7f")
                                )
                        )
        )
        fig.show()
