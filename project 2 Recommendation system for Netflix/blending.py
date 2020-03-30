from __future__ import print_function
import numpy as np
import os
from helpers import load_data, split_data, calculate_rmse, get_linear_blend_clf, transform
from helpers import generate_submission
import scipy.sparse as sp




# TO RUN this function , you need to create two folders in the current path:
#  ./data/   ./split/
#  and one csv file store in path
#  ./data/data_train.csv

def auto_load_data():
    '''
    This function is used to load raw data from .csv file 
    and reload the splited training and testing set
    '''
    path_dataset = "data/data_train.csv"

    print('Loading the data...\n')
    ratings = load_data(path_dataset)

    # Split in training and testing sets
    train_file_path = 'split/sparse_trainset.npz'
    test_file_path = 'split/sparse_testset.npz'

    if os.path.exists(train_file_path) and os.path.exists(test_file_path):
        train = sp.lil_matrix(sp.load_npz(train_file_path))
        test = sp.lil_matrix(sp.load_npz(test_file_path))
    else:
        print('\nSplitting the data in train and test sets...')
        train, test = split_data(ratings, p_test=0.1)
        sp.save_npz(train_file_path, sp.csr_matrix(train))
        sp.save_npz(test_file_path, sp.csr_matrix(test))
    return ratings,train,test




# TO RUN this function , you need to create one folder in the current path:
#  ./dump

def blending(test):
    '''
    This function is used to blending the predicted results of all models
    data is loaded from the ./dump folder
    
    The rule of naming the predicted value on full matrix and testset is as following:
    full matrix:  number of the model + _full + else.npy
    test matrix:  number of the model + _test + else.npy
    for example,  0_full_ALS.npy
                  0_test_ALS.npy
    all of these files should be stored in the ./dump directory
    '''
    X = np.zeros((1,10000000))
    X_test = test[test.nonzero()].toarray()
    path = 'dump'
    files = os.listdir(path)
    for name in files:
        for i in range(100):
            if name.endswith('.npy') and name.startswith('%s'%(i)):
                if name.startswith('%s_full'%(i)):
                    temp1 = np.load(os.path.join(path, name)).reshape(1,-1)
                    X = np.concatenate((X,temp1))
                elif name.startswith('%s_test'%(i)):
                    temp2 = np.load(os.path.join(path, name)).reshape(1,-1)
                    X_test = np.concatenate((X_test,temp2))
                else:
                    print('Wrong naming rule of the file!                           Please look the </dump> folder and check')
    X=np.delete(X,0,0)
    X_test=np.delete(X_test,0,0)
    y_test=test[test.nonzero()].toarray()[0]
    Num_model=X.shape[0]
    print('\nThe number of model is ',Num_model)

    # Linear blend of the previous models computed on the test set.
    clf = get_linear_blend_clf(X_test, y_test)
    print('RMSE Test: %f' % calculate_rmse(clf.predict(X_test.T), y_test))
    print('Weights of the different models:', clf.coef_)

    # Final predicted labels matrix
    predicted_labels = clf.predict(X.T).reshape(10000,1000)
    # Generate the CSV submission file
    generate_submission(predicted_labels)
    np.save('data/final_x.npy',predicted_labels)

