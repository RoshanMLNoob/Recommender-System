import numpy as np
import pandas as pd
import json

class Collaborative_Filtering(object):

    def __init__(self , Y , k , movie_list , factor=0.01 , default=0 , rating_range=(1,5)):

        self.Y = np.array(Y , ndmin=2)
        self.Y_init = np.array(Y , ndmin=2)

        self.m , self.n = self.Y.shape # m = users , n = items
        self.k = k #latent factors

        self.U = np.random.randn(self.m , self.k)*factor #togglable 
        self.V = np.random.randn(self.n , self.k)*factor #togglable

        self.default = default
        self.movie_list = movie_list
        self.rating_range = rating_range

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

    def optimize_U(self , U,V , lam=0):

        for i in range(self.m):

            rated_ind = np.where(self.Y[i] > 0)[0]

            if len(rated_ind) == 0:
                continue

            Vr = V[rated_ind]
            yr = self.Y[i , rated_ind]

            A = Vr.T@Vr + lam*np.eye(self.k)
            b = Vr.T@yr

            #U[i] = np.linalg.inv(A)@b #Anylatical solution
            U[i] = np.linalg.solve(A,b) #Better as it uses LU Decomposition for speed

    def optimize_V(self , U,V, lam=0):

        for j in range(self.n):

            rated_users = np.where(self.Y[:,j] > 0)[0]

            if len(rated_users) == 0:
                continue

            Ur = U[rated_users]
            yr = self.Y[rated_users,j]

            A = Ur.T@Ur + lam*np.eye(self.k)
            b = Ur.T@yr

            V[j] = np.linalg.solve(A,b)

    def Train(self , lam=0, epoch=10, interval=1):

        self.lam = lam
        print(f"Lambda = {lam}")

        for ep in range(epoch): #Uses ALS (Alternating Least Squares)

            self.optimize_U(self.U , self.V , lam=lam)
            self.optimize_V(self.U , self.V , lam=lam)

            if ep%interval == 0:
                print(f"Epoch {ep+1}: {self.J(self.U,self.V,lam=lam) / np.count_nonzero(self.Y)} ")

        return 0

    def Final_Matrix(self):

        return self.U@self.V.T , self.J(self.U , self.V , lam=self.lam)

    def Show_Result(self , user , top_most=10):

        Y_hat , Cost = self.Final_Matrix()

        print(f"Our model has created a final Matix Y")
        print(f"It's final cost is {Cost / np.count_nonzero(self.Y) }")
        print(f"For user {user} our model recommends these movies along with the rating which the user is likely to give to them: ")

        Y_hat[user][self.Y_init[user] != self.default] = self.default

        recommendation = np.argpartition(Y_hat[user] , -top_most)[-top_most]
        recommendation_score = np.clip(np.partition(Y_hat[user] , -top_most)[-top_most] , self.rating_range[0] , self.rating_range[1])

        show = dict({})

        for r,s in zip(recommendation[::-1],recommendation_score[::-1]):

            show[self.movie_list[r]] = s

        print(f"{json.dumps(show , indent=4)}")

        return 0

    def Show_Result_NewUser(self, new_user_ratings, top_most=10):

        new_user_ratings = np.array(new_user_ratings , ndmin=1)
        rated_ind = np.where(new_user_ratings > 0)[0]

        Vr = self.V[rated_ind]
        yr = new_user_ratings[rated_ind]

        A = Vr.T@Vr + self.lam*np.eye(self.k)
        b = Vr.T@yr
        u_new = np.linalg.solve(A,b)

        y_hat_new = u_new@self.V.T

        print(f"Our model has computed the latent vectors for the new user")
        print(f"For the new user our model recommends the follwing movies along with the ratings it predicts the user might give it to them: ")

        y_hat_new[new_user_ratings != self.default] = 0

        recommendation = np.argpartition(y_hat_new , -top_most)[-top_most]
        recommendation_score = np.clip(np.partition(y_hat_new, -top_most)[-top_most], self.rating_range[0] , self.rating_range[1])

        show = dict({})

        for r,s in zip(recommendation[::-1],recommendation_score[::-1]):

            show[self.movie_list[r]] = s

        print(f"{json.dumps(show , indent=4)}")

        return 0
