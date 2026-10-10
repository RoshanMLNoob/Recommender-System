import numpy as np
import pandas as pd
import os
import sys
from recommender_sys_ColabFilter import Collaborative_Filtering
from recommender_sys_ContentBased import Content_Based_Recommendation as Content_Based
import json

file_anime = r"C:\Users\rosha\OneDrive\Desktop\Python_ML\Machine_Learning_Practice_Projects\Recommender_System\data\anime.csv"
file_rating = r"C:\Users\rosha\OneDrive\Desktop\Python_ML\Machine_Learning_Practice_Projects\Recommender_System\data\rating.csv"

users = (0,1000)

a_df = pd.read_csv(file_anime)
r_df = pd.read_csv(file_rating)

#Creating a empty Y matrix of zeros


users = list(set(r_df["user_id"]))[users[0]:users[1]]
movies = list(a_df["anime_id"])
movies_name = list(a_df["name"])


id2mn = {i:j for i,j in zip(movies,movies_name)}
m2i = {j:i for i,j in enumerate(movies)}


Y = np.zeros((len(users),len(movies)))

alpha = 0.5

for u in users:

    user_movies , user_ratings = list(r_df.loc[r_df["user_id"]==u , 'anime_id']) , list(r_df.loc[r_df["user_id"]==u , "rating"])
    user_movies_indeces = [m2i[m] for m in user_movies]

    for i,m in enumerate(user_movies_indeces):
        if user_ratings[i] != 0:
            Y[u-1 , m] = user_ratings[i]
        else:
            Y[u-1 , m] = alpha*np.mean(np.where(np.array(user_ratings)==-1 , 0 , np.array(user_ratings)))


user = 512 #Togglable for user to user

user_movies , user_ratings = [id2mn[m] for m in list(r_df.loc[r_df["user_id"]==user , 'anime_id'])] , list(r_df.loc[r_df["user_id"]==user , "rating"])
print('user ratings: ')
print(json.dumps({mov:rat for mov,rat in zip(user_movies , user_ratings) if rat != -1.0} , indent=4))

Recommender = Collaborative_Filtering(Y=Y , k=50 , movie_list=movies_name , factor=0.01 , default=0 , rating_range=(1,10))
Recommender.Train(lam=0.05 , epoch=100 , interval=10)
Recommender.Run(user=user-1 , top_most=10)