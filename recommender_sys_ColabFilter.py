import numpy as np
import pandas as pd

class Collaborative_Filtering(object):

    def __init__(self , Y , k , movie_list , factor=0.01):

        self.Y = np.array(Y , ndmin=2)
        self.Y_init = np.array(Y , ndmin=2)

        self.m , self.n = self.Y.shape # m = users , n = items
        self.k = k #latent factors

        self.U = np.random.randn(self.m , self.k)*factor #togglable 
        self.V = np.random.randn(self.n , self.k)*factor #togglable

        self.movie_list = movie_list

    
