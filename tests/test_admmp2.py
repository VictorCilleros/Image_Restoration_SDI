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
        
    def test_precompute_matrix_invTmu(self):
        #Pas good encore
        """ Test if all of the points are in the ball"""
        rng = np.random.default_rng(84548)
        shape = rng.integers(10,100)
        x = rng.integers(0, 255, size=(shape, shape))
        h = h = 1/16 * np.array([[1,2,1],
                     [2,4,2],
                     [1,2,1]])
        
        algo = ADMMP2(x, h)
        y =  algo.gaussian_filter_2d()
        _, inv_Tmu =  algo._precompute_matrix(x, y, 0.3)
        i, j = inv_Tmu.shape
        test = inv_Tmu.reshape(-1)[:-1].reshape(i-1, j+1)
        self.assertTrue(~np.any(test[:, 1:]), True)
    
    def test_precompute_matrix_padding(self):
        """ Test if all of the points are in the ball"""
        rng = np.random.default_rng(84548)
        shape= rng.integers(10,100)
        x = rng.integers(0, 255, size=(shape, shape))
        h = h = 1/16 * np.array([[1,2,1],
                     [2,4,2],
                     [1,2,1]])
        
        algo = ADMMP2(x, h)
        y =  algo.gaussian_filter_2d()
        pre_comput_Ty, _ =  algo._precompute_matrix(x, y, 0.3)
        self.assertTrue(np.allclose(np.asarray(pre_comput_Ty.shape), np.asarray(x.shape)), True)