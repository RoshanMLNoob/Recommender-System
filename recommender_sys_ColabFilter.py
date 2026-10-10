import numpy as np
import pandas as pd

class Collaborative_Filtering(object):

    def __init__(self , Y , k , movie_list , factor=0.01 , default=0):

        self.Y = np.array(Y , ndmin=2)
        self.Y_init = np.array(Y , ndmin=2)

        self.m , self.n = self.Y.shape # m = users , n = items
        self.k = k #latent factors

        self.U = np.random.randn(self.m , self.k)*factor #togglable 
        self.V = np.random.randn(self.n , self.k)*factor #togglable

        self.default = default

        self.movie_list = movie_list

    def y_hat(self, U , V , i , j):

        return U[i].T@V[j]

    def rated(self , Y):

        return np.argwhere(Y != self.default)

    def J(self, U,V , lam=0):

        rated = self.rated(Y=self.Y)

        loss=0
        for i,j in rated:
            loss += (self.y_hat(U,V,i,j) - self.Y[i,j])**2 #MSE

        regularization = lam*( np.sum(np.linalg.norm(U , axis=1)**2) + np.sum(np.linalg.norm(V , axis=1)**2) )

        return loss + regularization

    