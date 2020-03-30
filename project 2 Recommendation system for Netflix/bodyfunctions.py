import sys
import numpy as np
from helpers import load_data, split_data
#from ALS import get_ALS_predictions
from ALS_ours import get_ALS_predictions
from SGD import get_SGD_predictions
from CrosVali import cross_validation
from SurpriseLib import get_splib_predictions

def parameter_search(path_dataset, fold, kwargs = None):
    # remind user to copy data
    input("\nIf your working directory contains following file, please copy them to another folder first: \n\
        --> prediction_overall.npy \n\
        --> RMSE_train_overall.npy \n\
        --> RMSE_test_overall.npy  \n\
    They are final results of last execution and they will be replaced by the results of current execution. \n\
    If you have saved them, Press Enter to continue...\n")
    

    # 1. Load the data    
    print('Loading the data...')
    ratings = load_data(path_dataset)
    
    # 2. Set optimization parameters
    algorithms = kwargs['algorithms']
    k_range    = kwargs['k_range']
    lambda_user_range = kwargs['lambda_u'] 
    lambda_item_range = kwargs['lambda_i']
    

    print('\nInitializing parameters for training...')
    prediction_overall   =   np.zeros( ( len(algorithms), len(k_range), len(lambda_user_range), len(lambda_item_range), ratings.shape[0] * ratings.shape[1]) )
    RMSE_train_overall   =   np.zeros( ( len(algorithms), len(k_range), len(lambda_user_range), len(lambda_item_range) ) )
    RMSE_test_overall    =   np.zeros( ( len(algorithms), len(k_range), len(lambda_user_range), len(lambda_item_range) ) )
    
    # 3. Search over given ranges
    print('\nConducting cross validation...')
    index = 0;
    for ind_alg, alg in enumerate(algorithms):
        
        if alg.lower() ==  'als' or alg.lower() ==  'als_ours' or alg.lower() ==  'sgd':
        
            for ind_k, k in enumerate(k_range):
                
                for ind_lambda_usr, lambda_user in enumerate(lambda_user_range):
                    
                    for ind_lambda_item, lambda_item in enumerate(lambda_item_range):
                        
                        if    alg.lower() ==  'als':
                            X, RMSE_test, RMSE_train = cross_validation(ratings, fold, get_ALS_predictions, lambda_user, lambda_item, k, kwargs = kwargs)
                        
                        elif  alg.lower() ==  'als_ours':
                            X, RMSE_test, RMSE_train = cross_validation(ratings, fold, get_ALS_predictions, lambda_user, lambda_item, k, kwargs = kwargs)
                                
                        elif  alg.lower() ==  'sgd':
                            X, RMSE_test, RMSE_train = cross_validation(ratings, fold, get_SGD_predictions, lambda_user, lambda_item, k, kwargs = kwargs)
                            
                        prediction_overall [ ind_alg, ind_k, ind_lambda_usr, ind_lambda_item, : ]   =   X
                        
                        RMSE_train_overall [ ind_alg, ind_k, ind_lambda_usr, ind_lambda_item    ]   =   RMSE_train
                        RMSE_test_overall  [ ind_alg, ind_k, ind_lambda_usr, ind_lambda_item    ]   =   RMSE_test
                    
                        index += 1
                        
                        print('\nNow we have accomplished:', 
                              100 * index/ ( len(algorithms) * len(k_range) * len(lambda_user_range) * len(lambda_item_range) ),
                              '% task.\n')      
                        
        elif alg.lower() ==  'svd' or alg.lower() ==  'knn' or alg.lower() ==  'coclustering':
            
            X, RMSE_test, RMSE_train = cross_validation(ratings, fold, get_splib_predictions, name = alg.lower(), kwargs = kwargs)
            
        else:
            print('Algorithm', alg, 'is not supported in this project!')
            sys.exit(1)
    
    # 4. Save results
    np.save("prediction_overall", prediction_overall)
    np.save("RMSE_train_overall", RMSE_train_overall)
    np.save("RMSE_test_overall" , RMSE_test_overall )
    
    input("\nSearch has finished, press Enter to continue...\n")
    
    return RMSE_train_overall, RMSE_test_overall



def algorithm_test(path_dataset, kwargs):
    '''test algorithms given '''
    ratings = load_data(path_dataset)
    
    train, test  = split_data(ratings);
    
    alg         = kwargs['algorithms']
    n_features  = kwargs['k_range']
    lambda_user = kwargs['lambda_u']
    lambda_item = kwargs['lambda_i']
    
    if    alg[0].lower() ==  'als':
        X, RMSE_test, RMSE_train = get_ALS_predictions(ratings, train, test, n_features, lambda_user, lambda_item, kwargs = kwargs)
    
    elif  alg[0].lower() ==  'als_ours':
        X, RMSE_test, RMSE_train = get_ALS_predictions(ratings, train, test, n_features, lambda_user, lambda_item, kwargs = kwargs)
            
    elif  alg[0].lower() ==  'sgd':
        X, RMSE_test, RMSE_train = get_SGD_predictions(ratings, train, test, n_features, lambda_user, lambda_item, kwargs = kwargs)
    
    elif alg[0].lower() ==  'svd' or alg[0].lower() ==  'knn' or alg[0].lower() ==  'cluster':
        
        X, RMSE_test, RMSE_train = get_splib_predictions(alg[0].lower(), train, test, kwargs = kwargs)
        
    else:
        print('Algorithm', alg, 'is not supported in this project!')
        sys.exit(1)
    
    return X, RMSE_test, RMSE_train
    
#def reproduce():
    
    