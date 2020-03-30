# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import numpy  as np
import pandas as pd 

from surprise import Reader
from surprise import Dataset
from surprise import KNNBaseline
from surprise import SVD
from surprise import CoClustering
from surprise.model_selection import train_test_split

from helpers import load_data, calculate_rmse

def load_data_forSP(path):
    # Import data
    path_dataset = path
    
    print('Loading the data...')
    ratings = load_data(path_dataset)
    
    coo = ratings.tocoo(copy = False)
    ratings_sp_df = pd.DataFrame({'item': coo.row, 'user': coo.col, 'rating': coo.data})[['item', 'user', 'rating']].reset_index(drop=True)
    
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))
    
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(ratings_sp_df[['user', 'item', 'rating']], reader)
    
    return data


def svd(data, kwargs):
    # Set algorithm
    n_factors     = kwargs.get('k_features')
    n_epochs      = kwargs.get('maxiter')
    lr_pu         = kwargs.get('lr_pu')
    lr_qi         = kwargs.get('lr_qi')
    reg_bu        = kwargs.get('reg_bu')
    reg_qi        = kwargs.get('reg_qi')
    
    
    algo = SVD(n_factors[0], n_epochs, 
               lr_pu[0]    , lr_qi[0]   , 
               reg_bu[0]   , reg_qi[0]  , 
               random_state = kwargs['random_seed'] )
    
    # Train the algorithm on the data, and predict ratings for the testset
    algo.fit(data)
     
    # Predict the full matrix
    prediction = np.zeros([10000,1000])
    for row in range(10000):
        for col in range(1000):
            prediction[row,col] = algo.predict(str(row+1),str(col+1)).est
            
    return prediction
    

def KNN(data, kwargs):
    # Set algorithm
    k_neigbor     = kwargs.get('n_neigbor')
    min_neighb    = kwargs.get('min_neigbor')
    similarity    = kwargs.get('similarity')
    
    options = {'name': similarity}
    algo = KNNBaseline(k = k_neigbor, 
                       min_k = min_neighb, 
                       sim_options = options)
    
    # Train the algorithm on the data, and predict ratings for the testset
    algo.fit(data)
    
    prediction = np.zeros([10000,1000])
    for row in range(10000):
        for col in range(1000):
            prediction[row,col] = algo.predict(str(row+1),str(col+1)).est
            
    return prediction
    
    
def cluster(data, kwargs):            
    # Set algorithm
    cluster_u     = kwargs.get('user_cluster')
    cluster_i     = kwargs.get('item_cluster')
    n_epochs      = kwargs.get('maxiter')
    
    # Set algorithm
    algo = CoClustering(n_cltr_u  = cluster_u[0], n_cltr_i  = cluster_i[0],
                        n_epochs  = n_epochs , random_state = kwargs['random_seed'] )
    
    # Train the algorithm on the data, and predict ratings for the testset
    algo.fit(data)
    
    prediction = np.zeros([10000,1000])
    for row in range(10000):
        for col in range(1000):
            prediction[row,col] = algo.predict(str(row+1),str(col+1)).est
            
    return prediction


def get_splib_predictions(name, training_data, testing_data, kwargs):
    '''
    name           :  name of the algorihm to be used. Now the KNN, Cosluster and SVD are supported.
    
    training_data  :  training data. If "load data" is set and the "path" is given in the kwargs, this data will be ignored
    
    testing_data   :  testing data use for calculating rmse of prediction.
    
    random_seed    :  random seed for SVD to train the recommendation system, default is 
    
    kwargs         :  kwargs for surpriseLib, different algorithms need different args.
                          SVD: { 'k_features',  'maxiter',  'lr_pu',  'lr_qi',  'reg_bu',  'reg_qi', 'random_seed' }
                          KNN: { 'n_neigbor' ,  'min_neigbor', 'similarity'}
                          cluster: {'user_cluster', 'item_cluster', 'maxiter', 'random_seed'}
    '''
    # 1. if "load data" is in the kwargs and is True, and the path is also given, the program will load data from the path
    # 2. else if data is given, use the given data
    # 3. else there is no avaliable data, so an error will be reported
    # The data is treated as a training set    
    if (kwargs.get('loaddata') is not None) and (kwargs.get('path') is not None):
        if kwargs['loaddata']:
            trainset = load_data_forSP(kwargs['path'])
    elif training_data is not None:
        coo = training_data.tocoo(copy = False)
        ratings_sp_df = pd.DataFrame({'item': coo.row, 'user': coo.col, 'rating': coo.data})[['item', 'user', 'rating']].reset_index(drop=True)   
        # A reader is still needed but only the rating_scale param is requiered.
        reader = Reader(rating_scale=(1, 5))
        # The columns must correspond to user id, item id and ratings (in that order).
        data = Dataset.load_from_df(ratings_sp_df[['user', 'item', 'rating']], reader)  
        trainset, testset = train_test_split(data, test_size=0.01, random_state = 0)
        
    else:
        sys.exit('No input data for surpries!')
        
    # train the model
    if name.lower() == 'svd':
        prediction =  svd(trainset, kwargs)
        
    elif name.lower() == 'knn':
        prediction =  KNN(trainset, kwargs)
        
    elif name.lower() == 'cluster':
        prediction =  cluster(trainset, kwargs)     
        
    else:
        sys.exit('The algorothm', name, 'is not supported yet.')
    
    
    testrmse  = calculate_rmse( prediction[testing_data.nonzero()], testing_data[testing_data.nonzero()].toarray()[0] )
    trainrmse = calculate_rmse( prediction[training_data.nonzero()], training_data[training_data.nonzero()].toarray()[0] )
    
    return prediction, trainrmse, testrmse
    