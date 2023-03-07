import unittest
from src.algorithme.admmp2 import ADMMP2
import numpy as np

class Test_TestADMMP2(unittest.TestCase):
    def test_gaussian_filter_2d(self):
        rng = np.random.default_rng(84548)
        shape = rng.integers(10,100)
        x = rng.integers(0, 255, size=(shape, shape))
        h = h = 1/16 * np.array([[1,2,1],
                     [2,4,2],
                     [1,2,1]])
        
        algo = ADMMP2(x, h)
        y = algo.gaussian_filter_2d()
        self.assertTrue(np.allclose(np.asarray(y.shape), np.asarray(x.shape) - 2))
        
    def test_precompute_matrix_padding(self):
        """ Test if all of the points are in the ball"""
        rng = np.random.default_rng(84548)
        shape= rng.integers(10,100)
        x = rng.integers(0, 255, size=(shape, shape))
        h = 1/16 * np.array([[1,2,1],
                     [2,4,2],
                     [1,2,1]])
        
        algo = ADMMP2(x, h)
        y =  algo.gaussian_filter_2d()
        pre_comput_Ty =  algo._precompute_matrix(x, y, 0.3)
        self.assertTrue(np.allclose(np.asarray(pre_comput_Ty.shape), np.asarray(x.shape)), True)
    
    def test_adjoint_A(self):
        rng = np.random.default_rng(84548)
        shape= rng.integers(10,100) //2 * 2
        x = rng.integers(0, 255, size=(shape, shape)) / 255.
        y = rng.integers(0, 255, size=(shape, shape)) / 255.
        h = 1/16 * np.array([[1,2,1],
                     [2,4,2],
                     [1,2,1]])
        
        algo = ADMMP2(x, h)
        Ax = algo.rfft_dot(x, algo.A_fft)
        A_star_y = algo.rfft_dot_adj(y, algo.A_fft)
        dot_a = np.hstack(Ax) @ np.hstack(y)
        dot_b = np.hstack(x) @ np.hstack(A_star_y)
        self.assertAlmostEqual(dot_a, dot_b)
    
    def test_adjoint_R(self):
        rng = np.random.default_rng(84548)
        shape= rng.integers(10,100) //2 * 2
        x = rng.integers(0, 255, size=(shape, shape)) / 255.
        h = 1/16 * np.array([[1,2,1],
                     [2,4,2],
                     [1,2,1]])
        
        algo = ADMMP2(x, h)
        Rx, slices = algo.wavelet_transform(x)
        y = rng.integers(0, 255, size=Rx.shape) / 255.
        R_star_y = algo.wavelet_transform_adjoint(y, slices)
        dot_a = np.hstack(Rx) @ np.hstack(y)
        dot_b = np.hstack(x) @ np.hstack(R_star_y)
        self.assertAlmostEqual(dot_a, dot_b)