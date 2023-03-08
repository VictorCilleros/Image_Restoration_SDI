import numpy as np
import plotly.graph_objects as go
import time
from PIL import Image
import pywt
import pylops

# TODO : 
#    - Vérifier les différentes fonctions pour être sûr de faire les bonnes opérations. (d'après le prof ça à l'air good)


class ADMMP2:

    def __init__(self, x:np.ndarray, h:np.ndarray, lambd:float=0.2, mu:np.ndarray=0.3, nu:float=0.4, sigma:float=5.03e-4):

        assert nu>0, "nu doit être strictement positif."
        assert mu>0, "mu doit être strictement positif."

        self.x = x
        self.h = h   
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.sigma = sigma
        self.A_fft = np.fft.rfft2(np.pad(h, ((0, x.shape[0] - h.shape[0]), (0, x.shape[1]-h.shape[1])), "constant", constant_values=(0)))
        self._A_fft = np.fft.fft2(np.pad(h, ((0, x.shape[0] - h.shape[0]), (0, x.shape[1]-h.shape[1])), "constant", constant_values=(0)))
        self.H_nu_inv = 1/(np.abs(self.A_fft)**2 + nu)

        self.mode = 'zero'
        self.wavelet = 'haar'

    #******************************************************************************************************************
    #                                                Pré-compute
    #******************************************************************************************************************

    def _precompute_matrix(self, y:np.ndarray)-> np.ndarray:
        
        # A vérifier et à faire avec la fonction _inv
        return np.pad(y, pad_width=(self.h.shape[0]-1, 0), mode='constant', constant_values=0)
    
    #******************************************************************************************************************
    #                                                   Utils
    #******************************************************************************************************************
    
    def gaussian_filter_2d(self) -> np.ndarray:
        """Apply gaussian smoothing on a 2D array
        Returns:
            np.ndarray: The 2D array after the gaussian filter
        """
        x = self.x
        h = self.h
        M1 = x.shape[0]
        N1 = x.shape[1]
        M2 = h.shape[0]
        N2 = h.shape[1]

        P2_h = np.pad(h, ((0, M1-M2), (0, N1-N2)), "constant", constant_values=(0))
        dft_P1_x = np.fft.rfft2(x)
        dft_P2_h = np.fft.rfft2(P2_h)
        hadamard = np.multiply(dft_P1_x, dft_P2_h)
        inv = np.fft.irfft2(hadamard)
        return inv[M2-1:, N2-1:]
    
    
    #############################     Sparsifying transform     #############################
    
    def wavelet_transform(self, x:np.ndarray)->np.ndarray:
        """Computes Rx (sparsifying transform with wavelets frames)

        Args:
            x (np.ndarray): image

        Returns:
            np.ndarray: image_reconstructed
            list : list of slices, we need this to reconstruct apply adjoint.
        """
        # Choose the wavelet type and level
        wavelet = self.wavelet
        # Perform 2D wavelet decomposition
        coeffs = pywt.wavedec2(x, wavelet, level=1, mode=self.mode)
        W, slices = pywt.coeffs_to_array(coeffs)

        return W, slices
    
    
    def wavelet_transform_adjoint(self, x:np.ndarray, slices) -> np.ndarray:
        """Computes R*x (sparsifying transform with wavelets frames adjoint)

        Args:
            x (np.ndarray): image

        Returns:
            np.ndarray: Results of Sparsifying adjoint transform on the given image.
        """
        # Choose the wavelet type and level
        wavelet = self.wavelet
        coeffs = pywt.array_to_coeffs(x, slices, output_format='wavedec2')
        # Obtain the adjoint using the inverse wavelet transform
        img_adjoint = pywt.waverec2(coeffs, wavelet, mode=self.mode)

        return img_adjoint

    #Pas utilisé
    def wavelet_transform_2(self, x):
        cA, (cH, cV, cD) = pywt.dwt2(x, 'haar')
        cTOP = np.concatenate((cA, cH), axis=1)
        cBOT = np.concatenate((cV, cD), axis=1)
        c = np.concatenate((cTOP, cBOT), axis=0)
        return c
    
    #Pas utilisé
    def wavelet_transform_adjoint_2(self, c, dim):
        cA_, cH_, cV_, cD_ = c[:dim, :dim], c[:dim, dim:], c[dim:, :dim], c[dim:, dim:]
        coeffs_ = cA_, (cH_, cV_, cD_) 
        xr = pywt.idwt2(coeffs_, 'haar')
        return xr

    
    #############################     FFT Tricks     #############################
    
    def rfft_dot(self, x:np.ndarray, F:np.ndarray)->np.ndarray:
        return np.fft.irfft2(np.multiply(F, np.fft.rfft2(x)))
    
    def fft_dot(self, x:np.ndarray, F:np.ndarray)->np.ndarray:
        return np.fft.ifft2(np.multiply(F, np.fft.fft2(x))).real
    
    def rfft_dot_adj(self, x:np.ndarray, F:np.ndarray)->np.ndarray:
        return np.fft.irfft2(np.multiply(np.conj(F), np.fft.rfft2(x)))
      
 
    #******************************************************************************************************************
    #                                                   ADMM-P2
    #******************************************************************************************************************

    def fit_transform(self, y:np.ndarray, eps:float=10e-2, stop=None):
        
        x = self.x
        xref = self.x
        
        mask = np.full(y.shape, 1/(1+self.mu))
        mask = np.pad(mask, pad_width=(self.h.shape[0]-1, 0), mode='constant', constant_values=1/self.mu)
        pre_comput_Ty = self._precompute_matrix(y)
        
        eta0= np.zeros(x.shape)
        x = self.rfft_dot_adj(pre_comput_Ty, self.A_fft)
        rx, slices = self.wavelet_transform(x)
        eta1 = np.zeros(rx.shape)
        
        u0 = np.multiply(mask, pre_comput_Ty + self.mu * (self.rfft_dot(x, self.A_fft) + eta0))
        #u1 = pywt.threshold(rx + eta1, self.lambd/(self.mu*self.nu), mode='soft')
        u1 = pywt.threshold(rx + eta1, 1e-3, mode='soft')
        x_ = np.fft.irfft2(np.multiply(self.H_nu_inv, np.multiply(np.conj(self.A_fft), np.fft.rfft2(u0-eta0)) + self.nu * np.fft.rfft2(self.wavelet_transform_adjoint(u1 - eta1, slices))))
        
        Ax_ = self.rfft_dot(x_, self.A_fft)
        rx, _ = self.wavelet_transform(x_)
        
        eta0 = eta0 - u0 + Ax_
        eta1 = eta1 - u1 + rx
        
        iter = 0
        
        #paramètres pour plot la convergence : 
        err = np.linalg.norm(x_ - xref,2)**2 / np.linalg.norm(xref,2)**2
        bsnr = 10*np.log10(np.var(Ax_)/self.sigma**2)
        lag = 1/2 * np.linalg.norm(y - self.rfft_dot(u0, self.A_fft)[(x.shape[0]-y.shape[0])//2:(x.shape[0]+y.shape[0])//2, (x.shape[1]-y.shape[1])//2:(x.shape[1]+y.shape[1])//2]) + self.lambd*np.linalg.norm(u1, 1) + self.mu/2 * np.linalg.norm(u0-Ax_-eta0)**2 + self.mu*self.nu/2*np.linalg.norm(u1-rx-eta1)**2
        tabError = [err]
        tabTime = [0]
        tabBSNR = [bsnr]
        tabLagran = [lag]
        timeRef= time.time()
        
        
        #Critère d'arrêt pour les tests
        if stop is None:
            nstop = np.inf
        else:
            nstop=stop
            
        while err>eps and iter<nstop:
            x = x_
            
            u0 = np.multiply(mask, pre_comput_Ty + self.mu * (self.rfft_dot(x, self.A_fft) + eta0))
            rx, slices= self.wavelet_transform(x)
            u1 = pywt.threshold(rx + eta1, self.lambd/(self.mu*self.nu), mode='soft')
            u1 = pywt.threshold(rx + eta1, 1e-3, mode='soft')
            x_ = np.fft.irfft2(np.multiply(self.H_nu_inv, np.multiply(np.conj(self.A_fft), np.fft.rfft2(u0-eta0)) + self.nu * np.fft.rfft2(self.wavelet_transform_adjoint(u1 - eta1, slices))))
            
            Ax_ = self.rfft_dot(x_, self.A_fft)
            rx = self.wavelet_transform(x_)
             
            eta0 = eta0 - u0 + Ax_
            eta1 = eta1 - u1 + rx
            
            
            iter += 1
            
            err = np.linalg.norm(x_ - xref, 2)**2 / np.linalg.norm(xref, 2)**2
            tabError.append(err)
            tabTime.append(time.time()-timeRef)
            bsnr = 10*np.log10(np.var(Ax_)/self.sigma**2)
            tabBSNR.append(bsnr)
            lag = 1/2 * np.linalg.norm(y - self.rfft_dot(u0, self.A_fft)[(x.shape[0]-y.shape[0])//2:(x.shape[0]+y.shape[0])//2, (x.shape[1]-y.shape[1])//2:(x.shape[1]+y.shape[1])//2]) + self.lambd*np.linalg.norm(u1, 1) + self.mu/2 * np.linalg.norm(u0-Ax_-eta0)**2 + self.mu*self.nu/2*np.linalg.norm(u1-rx-eta1)**2
            tabLagran.append(lag)
            
        return x_, iter,tabError,tabTime, tabBSNR, tabLagran

#******************************************************************************************************************
#                                                   Affichages -- PLOTLY
#******************************************************************************************************************

    def plot_convergence_iter(self,tabError : np.ndarray,tabTime : np.ndarray=None, title=None, xlabel=None, ylabel=None)->None:
        
        if tabTime is not None:
            x=tabTime
        else:
            nIter = np.shape(tabError)[0]
            x = np.linspace(0,nIter,nIter+1)

        fig = go.Figure(data=go.Scatter(x = x, y = tabError))
        fig.update_layout(
                    title=go.layout.Title(
                    text=title,
                    xref="paper",
                    x=0
                        ),
                    xaxis=go.layout.XAxis(
                        title=go.layout.xaxis.Title(
                            text=xlabel,
                            font=dict(
                                family="Courier New, monospace",
                                size=18,
                                color="#7f7f7f"
                                    )
                                )
                        ),
                    yaxis=go.layout.YAxis(
                            title=go.layout.yaxis.Title(
                            text=ylabel,
                            font=dict(family="Courier New, monospace",size=18,color="#7f7f7f")
                                )
                        )
        )
        fig.show()
    
class ADMMP2_v2:

    def __init__(self, x:np.ndarray, h:np.ndarray, lambd:float=0.2, mu:np.ndarray=0.3, nu:float=0.4, sigma:float=5.03e-4):

        assert nu>0, "nu doit être strictement positif."
        assert mu>0, "mu doit être strictement positif."

        self.x = x
        self.h = h   
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.sigma = sigma
        self.A_fft = np.fft.rfft2(np.pad(h, ((0, x.shape[0] - h.shape[0]), (0, x.shape[1]-h.shape[1])), "constant", constant_values=(0)))
        self._A_fft = np.fft.fft2(np.pad(h, ((0, x.shape[0] - h.shape[0]), (0, x.shape[1]-h.shape[1])), "constant", constant_values=(0)))
        self.H_nu_inv = 1/(np.abs(self.A_fft)**2 + nu)
        self.R = pylops.signalprocessing.DWT2D(x.shape, wavelet="haar", level=1)
        self.mode = 'zero'
        self.wavelet = 'haar'

    #******************************************************************************************************************
    #                                                Pré-compute
    #******************************************************************************************************************

    def _precompute_matrix(self, y:np.ndarray)-> np.ndarray:
        
        # A vérifier et à faire avec la fonction _inv
        return np.pad(y, pad_width=(self.h.shape[0]-1, 0), mode='constant', constant_values=0)
    
    #******************************************************************************************************************
    #                                                   Utils
    #******************************************************************************************************************
    
    def gaussian_filter_2d(self) -> np.ndarray:
        """Apply gaussian smoothing on a 2D array
        Returns:
            np.ndarray: The 2D array after the gaussian filter
        """
        x = self.x
        h = self.h
        M1 = x.shape[0]
        N1 = x.shape[1]
        M2 = h.shape[0]
        N2 = h.shape[1]

        P2_h = np.pad(h, ((0, M1-M2), (0, N1-N2)), "constant", constant_values=(0))
        dft_P1_x = np.fft.rfft2(x)
        dft_P2_h = np.fft.rfft2(P2_h)
        hadamard = np.multiply(dft_P1_x, dft_P2_h)
        inv = np.fft.irfft2(hadamard)
        return inv[M2-1:, N2-1:]
    
    
    #############################     Sparsifying transform     #############################
    
    def wavelet_transform(self, x:np.ndarray)->np.ndarray:
        """Computes Rx (sparsifying transform with wavelets frames)

        Args:
            x (np.ndarray): image

        Returns:
            np.ndarray: image_reconstructed
            list : list of slices, we need this to reconstruct apply adjoint.
        """
        # Choose the wavelet type and level
      

        return self.R * x
    
    
    def wavelet_transform_adjoint(self, x:np.ndarray) -> np.ndarray:
        """Computes R*x (sparsifying transform with wavelets frames adjoint)

        Args:
            x (np.ndarray): image

        Returns:
            np.ndarray: Results of Sparsifying adjoint transform on the given image.
        """
        return self.R.adjoint() * x

    #Pas utilisé
    def wavelet_transform_2(self, x):
        cA, (cH, cV, cD) = pywt.dwt2(x, 'haar')
        cTOP = np.concatenate((cA, cH), axis=1)
        cBOT = np.concatenate((cV, cD), axis=1)
        c = np.concatenate((cTOP, cBOT), axis=0)
        return c
    
    #Pas utilisé
    def wavelet_transform_adjoint_2(self, c, dim):
        cA_, cH_, cV_, cD_ = c[:dim, :dim], c[:dim, dim:], c[dim:, :dim], c[dim:, dim:]
        coeffs_ = cA_, (cH_, cV_, cD_) 
        xr = pywt.idwt2(coeffs_, 'haar')
        return xr

    
    #############################     FFT Tricks     #############################
    
    def rfft_dot(self, x:np.ndarray, F:np.ndarray)->np.ndarray:
        return np.fft.irfft2(np.multiply(F, np.fft.rfft2(x)))
    
    def fft_dot(self, x:np.ndarray, F:np.ndarray)->np.ndarray:
        return np.fft.ifft2(np.multiply(F, np.fft.fft2(x))).real
    
    def rfft_dot_adj(self, x:np.ndarray, F:np.ndarray)->np.ndarray:
        return np.fft.irfft2(np.multiply(np.conj(F), np.fft.rfft2(x)))
      
 
    #******************************************************************************************************************
    #                                                   ADMM-P2
    #******************************************************************************************************************

    def fit_transform(self, y:np.ndarray, eps:float=10e-2, stop=None):
        
        x = self.x
        xref = self.x
        
        mask = np.full(y.shape, 1/(1+self.mu))
        mask = np.pad(mask, pad_width=(self.h.shape[0]-1, 0), mode='constant', constant_values=1/self.mu)
        pre_comput_Ty = self._precompute_matrix(y)
        
        eta0= np.zeros(x.shape)
        x = self.rfft_dot_adj(pre_comput_Ty, self.A_fft)
        rx = self.wavelet_transform(x)
        eta1 = np.zeros(rx.shape)
        print(pre_comput_Ty.shape, mask.shape, eta0.shape)
        u0 = np.multiply(mask, pre_comput_Ty + self.mu * (self.rfft_dot(x, self.A_fft) + eta0))
        u1 = pywt.threshold(rx + eta1, self.lambd/(self.mu*self.nu), mode='soft')
        #u1 = pywt.threshold(rx + eta1, 1e-3, mode='soft')
        x_ = np.fft.irfft2(np.multiply(self.H_nu_inv, np.multiply(np.conj(self.A_fft), np.fft.rfft2(u0-eta0)) + self.nu * np.fft.rfft2(self.wavelet_transform_adjoint(u1 - eta1))))
        
        Ax_ = self.rfft_dot(x_, self.A_fft)
        rx = self.wavelet_transform(x_)
        
        eta0 = eta0 - u0 + Ax_
        eta1 = eta1 - u1 + rx
        
        iter = 0
        
        #paramètres pour plot la convergence : 
        err = np.linalg.norm(x_ - xref,2)**2 / np.linalg.norm(xref,2)**2
        bsnr = 10*np.log10(np.var(Ax_)/self.sigma**2)
        lag = 1/2 * np.linalg.norm(y - self.rfft_dot(u0, self.A_fft)[(x.shape[0]-y.shape[0])//2:(x.shape[0]+y.shape[0])//2, (x.shape[1]-y.shape[1])//2:(x.shape[1]+y.shape[1])//2]) + self.lambd*np.linalg.norm(u1, 1) + self.mu/2 * np.linalg.norm(u0-Ax_-eta0)**2 + self.mu*self.nu/2*np.linalg.norm(u1-rx-eta1)**2
        tabError = [err]
        tabTime = [0]
        tabBSNR = [bsnr]
        tabLagran = [lag]
        timeRef= time.time()
        
        
        #Critère d'arrêt pour les tests
        if stop is None:
            nstop = np.inf
        else:
            nstop=stop
            
        while err>eps and iter<nstop:
            x = x_
            
            u0 = np.multiply(mask, pre_comput_Ty + self.mu * (self.rfft_dot(x, self.A_fft) + eta0))
            rx = self.wavelet_transform(x)
            u1 = pywt.threshold(rx + eta1, self.lambd/(self.mu*self.nu), mode='soft')
            #u1 = pywt.threshold(rx + eta1, 1e-3, mode='soft')
            x_ = np.fft.irfft2(np.multiply(self.H_nu_inv, np.multiply(np.conj(self.A_fft), np.fft.rfft2(u0-eta0)) + self.nu * np.fft.rfft2(self.wavelet_transform_adjoint(u1 - eta1))))
            
            Ax_ = self.rfft_dot(x_, self.A_fft)
            rx = self.wavelet_transform(x_)
             
            eta0 = eta0 - u0 + Ax_
            eta1 = eta1 - u1 + rx
            
            
            iter += 1
            
            err = np.linalg.norm(x_ - xref, 2)**2 / np.linalg.norm(xref, 2)**2
            tabError.append(err)
            tabTime.append(time.time()-timeRef)
            bsnr = 10*np.log10(np.var(Ax_)/self.sigma**2)
            tabBSNR.append(bsnr)
            lag = 1/2 * np.linalg.norm(y - self.rfft_dot(u0, self.A_fft)[(x.shape[0]-y.shape[0])//2:(x.shape[0]+y.shape[0])//2, (x.shape[1]-y.shape[1])//2:(x.shape[1]+y.shape[1])//2]) + self.lambd*np.linalg.norm(u1, 1) + self.mu/2 * np.linalg.norm(u0-Ax_-eta0)**2 + self.mu*self.nu/2*np.linalg.norm(u1-rx-eta1)**2
            tabLagran.append(lag)
            
        return x_, iter,tabError,tabTime, tabBSNR, tabLagran

#******************************************************************************************************************
#                                                   Affichages -- PLOTLY
#******************************************************************************************************************

    def plot_convergence_iter(self,tabError : np.ndarray,tabTime : np.ndarray=None, title=None, xlabel=None, ylabel=None)->None:
        
        if tabTime is not None:
            x=tabTime
        else:
            nIter = np.shape(tabError)[0]
            x = np.linspace(0,nIter,nIter+1)

        fig = go.Figure(data=go.Scatter(x = x, y = tabError))
        fig.update_layout(
                    title=go.layout.Title(
                    text=title,
                    xref="paper",
                    x=0
                        ),
                    xaxis=go.layout.XAxis(
                        title=go.layout.xaxis.Title(
                            text=xlabel,
                            font=dict(
                                family="Courier New, monospace",
                                size=18,
                                color="#7f7f7f"
                                    )
                                )
                        ),
                    yaxis=go.layout.YAxis(
                            title=go.layout.yaxis.Title(
                            text=ylabel,
                            font=dict(family="Courier New, monospace",size=18,color="#7f7f7f")
                                )
                        )
        )
        fig.show()
 