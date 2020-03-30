
# coding: utf-8

# In[2]:

import sys
import time
import numpy as np
from helpers import load_data


# 1. do not recommend the moive that has been finished watching and rated, 
#    becasuse we assume that user will not see the same moive twice
# 2. using the new rating to mupdate the old matirx, we assume new rating is reliable 
#    and is helpful to recommend new moive to user
#    help later training
# 3. new variable flag: control whether or not user finishing watching one moive

def application_topmoives():
    
    try:
        predicted_all = np.load('data/final_x.npy')
    except:
        sys.exit('No prediction data, please train the recommendation system first!')
        
    ratings = load_data('data/data_train.csv')
    real_time_ratings = ratings.copy()
    recomd_matrix = predicted_all.copy()

    N=5       # recommend top N movies to user
    movie = np.zeros(N)

    while 1:
        user = int(input('Please input your user number (0~999) '))
        if user >= 0 and user < 1000:
            print('Welcome, Mr/Mrs. user no.',user)
            break
        else:
            print('Out of range! Please input again')

    for i in range(N):
        fav = max(recomd_matrix[np.where(real_time_ratings.toarray()[:,user] == 0)[0],user])
        movie[i] = np.where(recomd_matrix[:,user] == fav)[0][0]
        print('\nWe recommend the moive No.%s to you \n(%s out of 5 in recommended list)'%(int(movie[i]),i+1))
        time.sleep(3)
        flag = input('\nHave you finished watching the moive? T or F  ')
        rating = input('Please write your rating to this moive (1~5star)\n')
        if flag == 'T':
            real_time_ratings[int(movie[i]),user]=rating
        else:
            recomd_matrix[int(movie[i]),user]=rating
        print('\n')
        print("This round of recommended movies has finished, \nThank you for your time! Hope to see you again next time")

