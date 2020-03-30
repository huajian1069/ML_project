import os
import sys
import numpy as np
import scipy.sparse as sp
from helpers import split_data
from SurpriseLib import get_splib_predictions

def cross_validation(ratings, fold, solver, lambda_usr = 0.02, lambda_item = 0.02, k_features = 5 , name = None, kwargs = None):
    
    A = ratings.copy();
    train_array = [];
    test_array  = [];
    # Check if the data have been generated and whether the data is correct
    # if file_flag is False, then new data must been generated
    file_flag   = filecheck(fold)
    
    for i in range(fold):
        
        trainset_file_path = 'trainset_%s_th.npz'             % (i)
        testset_file_path  = 'testset_%s_th.npz'              % (i)
        
        if (os.path.exists(trainset_file_path) and os.path.exists(testset_file_path) and file_flag):
        # if data have been generated before, just load them 
            train = sp.load_npz(trainset_file_path)
            test  = sp.load_npz(testset_file_path)
            
            train_array.append(train)
            test_array.append(test)
        else:
        # else generate k-fold dataset for training and testing
            p_test   = 1/(fold-i);
            _, test  = split_data(A,p_test,seed=998);
            train    = sp.lil_matrix(ratings.shape)
            train    = ratings - test;
            test     = test.tocsc()
            
            train_array.append(train);
            test_array.append(test);
            A = A - test;
            
            print('\n Crossvalidation dataset has been created')
            sp.save_npz(trainset_file_path, train.tocsr())
            sp.save_npz(testset_file_path, test.tocsr())
     
    '''
    11/Dec./2018, Zhantao Deng
        1. delete calculation of rmse since sgd provides rmse. 
        2. choose model showing the best err, rather than returning the last model 
        3. 
    '''
    # train period
    # Generate predictions by particular models specified by parameters
    
    # define rmse variables, intialize them with a large num
    rmse_test  = [100]
    rmse_train = [100]
    for ind, item  in enumerate(train_array):
        train = item
        test  = test_array[ind]
        
        if name is None:
            X_whole, trainERR, testERR = solver(ratings,    train,      test, 
                                                lambda_usr, lambda_item, k_features, ind, kwargs = kwargs)
        else: 
            sys.exit('Only SGD, ALS and ALS_ours support our cross validation')
            
            
        # store rmse
        rmse_test.append(testERR)
        rmse_train.append(trainERR)
        
        # if train err is smaller than former train err, store the model as the best model  
        if trainERR <= rmse_train[ind]:
            Best_X   = X_whole
            
    # transform from float to integer
    Best_X  =  transform(Best_X)
    
    return Best_X, np.mean(rmse_test[1:]), np.mean(rmse_train[1:])


def transform(x):
    integer_X_test_test=[min(max(int(item+0.5),1),5) for item in x.T]
    return np.array(integer_X_test_test)


def filecheck(fold):
    '''
        to make sure the the data have been generated in the same fold number, we must check whether fold-th set is exist. 
    '''
    training_foldfile   = 'trainset_%s_th.npz'             % (fold-1)
    testset_foldfile    = 'testset_%s_th.npz'              % (fold-1)
    
    training_foldfile_  = 'trainset_%s_th.npz'             % (fold)
    testset_foldfile_   = 'testset_%s_th.npz'              % (fold)
    
    return (os.path.exists(training_foldfile) and os.path.exists(testset_foldfile) and \
            (not os.path.exists(training_foldfile_)) and (not os.path.exists(testset_foldfile_)) )