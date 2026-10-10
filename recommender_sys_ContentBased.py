import numpy as np
import pandas as pd
import json

class Content_Based_Recommendation(object):

    def __init__(self, X,Y , default=0 , features=None , items=None):

        self.X = np.array(X , ndmin=2)
        self.Y = np.array(Y , ndmin=2)
        self.default = default

        self.items = np.array(items)
        self.features = np.array(features)

    def y_rated(self , user):

        y_rated = self.Y[user, self.Y[user] != self.default]
        return y_rated

    def X_rated(self, user):

        x_rated = self.X[self.Y[user] != self.default]
        return x_rated

    def w(self, user , lam=0):

        XR , yR =self.X_rated(user=user) , self.y_rated(user=user)
        W = np.linalg.inv(XR.T@XR + lam*(np.identity(XR.shape[1])))@(XR.T@yR)
        return W

    def y_hat(self , user , lam=0):

        W = self.w(user=user , lam=lam)
        predicted_ratings = self.X@W
        return predicted_ratings

    def recommended(self , user , lam=0):

        new_movies = self.Y[user]==self.default
        preds = self.y_hat(user=user , lam=lam)[new_movies]
        return preds

    def Cost(self , w , user , lam=0):

        XR , yR = self.X_rated(user=user) , self.y_rated(user=user)
        prediction_results = XR@w
        return np.sqrt((np.sum((prediction_results-yR)**2)+lam*(w.T@w)) / XR.shape[0])

    def run_recommendation(self , user , lam=0 , top_most=None , cost=False):

        items_unrated = self.items[self.Y[user] == self.default]
        ratings_for_unrated_items = self.recommended(user=user , lam=lam)

        _cost_ = self.Cost(self.w(user=user , lam=lam) , user=user , lam=lam)
        recommended_items = dict(zip(items_unrated.tolist() , ratings_for_unrated_items.tolist()))

        if top_most is None:
            if cost:
                return json.dumps(recommended_items , indent=4) , _cost_
            return json.dumps(recommended_items , indent=4)
        else:
            if cost:
                return json.dumps(sorted(recommended_items.items() , key=lambda item: item[1], reverse=True)[:top_most], indent=4) , _cost_
            return json.dumps(sorted(recommended_items.items() , key=lambda item: item[1], reverse=True)[:top_most] , indent=4)
