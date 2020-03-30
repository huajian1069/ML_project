# -*- coding: utf-8 -*-
import numpy as np
from bodyfunctions import parameter_search, algorithm_test
from recommendation_system import application_topmoives

func = 'recommendation_system'
path_dataset  =  './data/data_train.csv'

# In[1] grid search and cross validation -- only support our own models
if func == 'grid_search':
    cross_fold           =   5
    
    kwargs = { 'algorithms'  :  ['SGD'],
               'k_range'     :  np.array([5, 8]) ,
               'lambda_u'    :  np.array([0.2, 0.3]) ,
               'lambda_i'    :  np.array([0.2, 0.3]),
               'k_features'  :  np.array([5, 6]),  
               'maxiter'     :  50,  
               'lr_pu'       :  np.array([0.02,0.05]),  
               'lr_qi'       :  np.array([0.02,0.05]),  
               'reg_bu'      :  np.array([0.2, 0.5]),  
               'reg_qi'      :  np.array([0.2, 0.5]), 
               'random_seed' :  0,
               'n_neigbor'   :  np.array([20, 40]),  
               'min_neigbor' :  np.array([1, 5]),
               'similarity'  :  'cosine',
               'user_cluster'  :  np.array([3, 5]),
               'item_cluster'  :  np.array([10, 20]),  
               'mode'        : 'cross_validation'
               }
    
    RMSE_train_overall, RMSE_test_overall = parameter_search(path_dataset, cross_fold, kwargs)
    
    print('the lowerest testing rmse achieved is:', np.min(RMSE_test_overall.reshape(-1)) )

# In[2] model testing 
# choose a model and test it
elif func == 'test':
    kwargs = { 'algorithms'  :  ['knn'],     # choose one algorithm
               'k_range'     :  np.array([8]) ,
               'lambda_u'    :  np.array([0.3]) ,
               'lambda_i'    :  np.array([0.02]),
               'k_features'  :  np.array([6]),  
               'maxiter'     :  50,  
               'lr_pu'       :  np.array([0.02]),  
               'lr_qi'       :  np.array([0.02]),  
               'reg_bu'      :  np.array([0.2]),  
               'reg_qi'      :  np.array([0.2]), 
               'random_seed' :  0,
               'n_neigbor'   :  np.array([20]),  
               'min_neigbor' :  np.array([1]),
               'similarity'  :  'cosine',
               'user_cluster'  :  np.array([3]),
               'item_cluster'  :  np.array([10]),  
               'mode'        :  'test'
               }
    
    X, RMSE_test, RMSE_train = algorithm_test(path_dataset, kwargs = kwargs)
    
    print('The test rmse of', kwargs['algorithms'], 'is', RMSE_test, '\n' 
          'The training rmse of', kwargs['algorithms'], 'is', RMSE_train, '\n' )

# In[3]
elif func == 'recommendation_system':
    application_topmoives()

# In[4]
else: 
    print('Function', func, 'is not supported!')