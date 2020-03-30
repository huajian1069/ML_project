# -*- coding: utf-8 -*-
"""Some functions used by project 1"""

import csv
import numpy as np
import math
import matplotlib.pyplot as plt

#from Lib_for_project1 import *

#import matplotlib.pyplot as plt





# Evaluating algorithms / training models
def training_model(data_train, yb_train, opt_model, cros_vali, degree, lambda_, gamma, outlier, method):
    
    seed = np.random.randint(cros_vali*100)
    
    print('step 1:Started main function!')
    print('       Key parameters: cross validation slices      :',cros_vali)
    print('                       highest polinomial degree is :',degree   )
    print('                       the trained model number is  :',opt_model   )
    
    data_train_copy = data_train.copy() 
    
    print('step 2:Feature augmentaion...')
    #data_train = add_feature(data_train)
    data_train = build_model_data(data_train, degree)
    
    print('step 3:Cross validation begins...')
    indices = build_k_indices(yb_train, cros_vali, seed) 
    corrections, best_w, loss = cross_validation(data_train, yb_train, indices, opt_model, lambda_, gamma, outlier, method)
    
    print('step 4:End of Trainging\n')

    return np.mean(corrections), best_w, loss



## BASIC FUNCTIONS ## 
# Data loading
def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


# Data processing
def data_process(train_x, train_y, test_x, method, outlier):
    ''' Dealing with outlies in given dataset.
       
       train_x : a matrix of training data, each row is a data point; columns are features to be processed;
       train_y : label vector, corresponing to train_x;
       test_x  : a matrix of testing data, each row is a data point; columns are features to be processed;
       method  : str, the operation to implement on training and testing data;
              'no'         : no operation will be conducted;
              'mean'        : replace nan by mean of every column;
              'normalization'  : normalize training and testing data with (x-x.min)/(x.max() - x.min());
              'drop'        : drop columns containing nan;
       outlier : a variable, such as -999, 0 and np.nan; This given outlier value will be cleaned. '''
   
    print('  -> Data processing:', method)
    
    if method == 'no':
        print('No data processing is conducted')
    else:
        # replace outliers by nan
        if ~np.isnan(outlier):
            train_x[train_x == outlier] = np.nan
            test_x [ test_x == outlier] = np.nan
        else:
            print('No need to replace outlies by nan!\n')

        if method == 'mean':
            # nan is replaced by mean value here
            train_x  = fill_mean(train_x, train_y, 'train_data')
            test_x   = fill_mean(test_x , []     , 'test_data' )
            
        elif method == 'normalization':
            # To normalize data, firstly, nan is replaced by mean value here
            train_x = fill_mean(train_x, train_y, 'train_data')
            test_x  = fill_mean(test_x , []     , 'test_data' )
            # Normalization
            train_x = normalization(train_x, train_y, 'train_data')      
            test_x  = normalization(test_x , []     , 'test_data' )
    
        elif method == 'drop':
            # Drop columns containing nan
            to_drop_train  = np.isnan(train_x.min(axis = 0))
            to_drop_test   = np.isnan( test_x.min(axis = 0))
            to_drop      = to_drop_train | to_drop_test
            
            train_x = train_x[:, ~to_drop]
            test_x  = test_x [:, ~to_drop]      
        else:
            print('No such processing method! Please check your input!')
            return -1  
    return train_x, test_x  


def fill_mean(data, label, category):
    ''' Replacing nan with mean value of each column.
      
       data    : matrix, data to be processed; Each row is a point and each column is a feature;
       label   : vector, labels corresponing to every row of data;
       category : str, 'train_data' -- the data is training set and the label is not empty;
                  'test_data'  -- the data is testing set and the label is empty;
       '''
    # for training data, the mean of each classes can be obtained, 
    # so the nan of a point is replace by the mean of the point's corresponding class
    if len(data) != 0:
        if category == 'train_data':
            class_1 = data[label == 1 ]
            class_2 = data[label == -1]
            mean_1 = np.nanmean(class_1,axis = 0)
            mean_2 = np.nanmean(class_2,axis = 0)

            for ct_i in range(class_1.shape[1]):
                class_1[ np.isnan(class_1[:,ct_i]), ct_i] = mean_1[ct_i]
                class_2[ np.isnan(class_2[:,ct_i]), ct_i] = mean_2[ct_i] 
            data[label == 1 ] = class_1
            data[label == -1] = class_2

        # Since we would not know the class of testing data, we replace nan by the mean of all rows 
        elif category == 'test_data':
            mean = np.nanmean(data,axis = 0)     
            for ct_i in range(data.shape[1]):
                data[ np.isnan(data[:,ct_i]), ct_i] = mean[ct_i]
        else:
            print('No such data category! Please check your input!')    
    else:
        print('Data is empty, no operation is implemented.')     
    
    return data


def normalization(data, label, category):
    ''' Normalizing the given data.
      
       data    : ndarray   , data to be processed; Each row is a point and each column is a feature;
       label   : vector    , labels corresponing to every row of data;
       category : str      , 'train_data' -- the data is training set and the label is not empty;
                       'test_data'  -- the data is testing set and the label is empty;
       '''
    eps = 1e-2   # to prevent division by zero
    if len(data) != 0:
        if category == 'train_data':
            # for training data, the maximum and minimum values of two classes can be obtained, 
            # so points are normalized with maximum and minimum values of their corresponding class
            class_1 = data[label == 1, 1:data.shape[1] ]
            class_2 = data[label == -1, 1:data.shape[1]]  
            class_1 = (class_1 - np.nanmin(class_1,axis=0))/(np.nanmax(class_1,axis=0) - np.nanmin(class_1,axis=0) + eps)
            class_2 = (class_2 - np.nanmin(class_2,axis=0))/(np.nanmax(class_2,axis=0) - np.nanmin(class_2,axis=0) + eps)   

            data[label == 1, 1:data.shape[1] ] = class_1
            data[label == -1, 1:data.shape[1]] = class_2

        elif category == 'test_data':
            # Since the class of testing points are unknown, these points are normalized with maximum and minimum values among all points
            data_temp = data[:, 1:data.shape[1]].copy()
            data_temp = (data_temp - np.nanmin(data_temp,axis=0))/(np.nanmax(data_temp,axis=0) - np.nanmin(data_temp,axis=0) + eps)

            data[:, 1:data.shape[1]] = data_temp.copy()
        else:
            print('No such data category! Please check your input!')
    else:
        print('Data is empty, no operation is implemented.')
        
    return data


# Feature augmentation
def add_feature(data):
    ''' Expanding features to improve performance. Exponential items, cosine item and logrithmic items are appended. '''
    r_data=data.copy()
    
    # Appending expenantial columns
    r_data=np.c_[r_data,np.exp(data[:,4])]
    r_data=np.c_[r_data,np.exp(data[:,6])]
    r_data=np.c_[r_data,np.exp(data[:,7])]
    r_data=np.c_[r_data,np.exp(data[:,12])]
    r_data=np.c_[r_data,np.exp(data[:,14])]
    r_data=np.c_[r_data,np.exp(data[:,15])]
    r_data=np.c_[r_data,np.exp(data[:,18])]
    r_data=np.c_[r_data,np.exp(data[:,24])]
    
    # Appending cosine columns
    r_data=np.c_[r_data,np.cos(data[:,10])]
    r_data=np.c_[r_data,np.cos(data[:,11])]
    r_data=np.c_[r_data,np.cos(data[:,24])]
    r_data=np.c_[r_data,np.cos(data[:,27])]
    
    # Appending logrithmic columns
    r_data=np.c_[r_data,np.log(data[:,9])]
    r_data=np.c_[r_data,np.log(data[:,10])]
    r_data=np.c_[r_data,np.log(data[:,13])]
    r_data=np.c_[r_data,np.log(data[:,19])]

    r_data=np.c_[r_data,data[:,range(7,10)]**5]

    return r_data


# Cross validation
def cross_validation(feature, lable, indices, opt_model, lambda_, gamma, outlier, method):
    '''cross validation to evaluate performance of a given mode.
    
       feature      : ndarray  , feature vectors
       lable       : array   , lable of every row of feature
       indices      : ndarray  , grouped indices of the dataset
       opt_model     : int    , chosen classification model
                  1 -- Least Squrare
                  2 -- Ridge Regression
                 ...
                  7 -- Fisher Linear Discriminant
       lambda_      : float   , ridge regression coefficiency
       outlier      : float   , outlier to be processed
       method       :  str   , method to be used in data processing
       
       '''
    iter_times  = indices.shape[0]
    correction  = []
    weight     = []
    
    for ct_iter in range(iter_times):
        # create training data
        test_x = feature[indices[ct_iter]]
        test_y = lable[indices[ct_iter]].reshape(indices[ct_iter].shape[0],1)
        
        # create testing data
        train_indices = indices[ np.arange(indices.shape[0]) != ct_iter ]
        train_indices = train_indices.ravel()
        train_x = feature[ train_indices ]
        train_y = lable  [ train_indices ].reshape(train_indices.shape[0],1)

        # data processing
        train_x, test_x = data_process(train_x, train_y, test_x, method, outlier)
        
        w, loss = train_model(train_x, train_y, opt_model, lambda_, gamma)
        prediction = predict_labels(w, test_x)
        prediction=prediction.reshape(prediction.shape[0],1)
        #print('prediction',prediction.shape)
       # print(test_y.astype(float).shape)
       # print(prediction == test_y.astype(float))
        #print( np.sum(prediction == test_y.astype(float)))
        #while(1):
         #   a=1;    
        correction.append( np.sum(prediction == test_y.astype(float))/test_y.shape[0] )
        weight.append(w)    
        
        print('          --** Accuracy of cross validation', ct_iter, 'is:', correction[ct_iter])
        
    best_w = weight [ correction.index(max(correction))]
    return correction, best_w, loss


# Training models


def train_model(train_x, train_y, chosen_model, lambda_, gamma):
    ''' Training the model designated by chose_model
    
       train_x     : ndarray, features of training data
       train_y     : vector , labels corresponding to train_x
       chosen_model  : int   , to specify which model will be trained
       lambda_     : float  , ridge regression coeffocoency
       gamma      _: float  , step size of iterative algorithms
       ''' 
    if chosen_model  == 1:     
        w,loss=train_normal(train_y,train_x)
    
    elif chosen_model == 2:
        w,loss=train_ridge(train_y, train_x, lambda_)
    
    elif chosen_model == 3:
        w,loss=train_gradient(train_y, train_x, gamma)
    
    elif chosen_model == 4:
        w,loss=train_logistic(train_y, train_x, gamma)
        
    elif chosen_model == 5:
        w,loss=train_gradient_SGD(train_y, train_x, gamma)
                                   
    elif chosen_model == 6: 
        w,loss=train_reg_logistic(train_y, train_x, lambda_, gamma)
        #reserved for another algorithm
        
    elif chosen_model == 7:
        train_y=train_y.ravel();
       # print(train_y.shape)
        class_1 = train_x[train_y == -1][:,1:train_x.shape[1]]
        class_2 = train_x[train_y == 1][:,1:train_x.shape[1]]
        
        w, loss = Fisher_classifier(class_1, class_2)
        
    return w, loss

def train_normal(yb_train, data_train):
    w,loss=least_squares(yb_train,data_train)
    return w,loss

def train_ridge(yb_train, data_train, lambda_):
    w,loss=ridge_regression(yb_train, data_train, lambda_)
    return w,loss

def train_gradient(yb_train, data_train, step_size):
    max_iters  = 200
    gamma     = step_size
    initial_w  = np.zeros(len(data_train.T));
    w,loss    = least_squares_GD(yb_train, data_train, initial_w, max_iters, gamma)
    return w,loss

def train_logistic(yb_train, data_train, step_size):
    initial_w=np.ones(len(data_train.T));
    max_iters=300;
    gamma=step_size;
    w,loss=logistic_regression(yb_train, data_train, initial_w, max_iters, gamma)
    return w,loss

def train_gradient_SGD(yb_train, data_train, step_size):
    batch_size=1;
    max_iters = 300
    gamma = step_size
    initial_w=np.zeros(len(data_train.T));
    w,loss=least_squares_SGD(yb_train, data_train, initial_w, max_iters, gamma,batch_size)
    return w,loss 

def train_reg_logistic(yb_train, data_train,lambda_,gamma):
    
    initial_w=np.ones(len(data_train.T));
    max_iters=100;
    w,loss=reg_logistic_regression(yb_train, data_train, lambda_,initial_w, max_iters, gamma)
    return w,loss

def Fisher_classifier(class_1, class_2):
    # means
    m_1 = np.mean(class_1, axis = 0)
    m_2 = np.mean(class_2, axis = 0)
    
    # inner class1 dispersion
    S_1 = (class_1 - m_1).T.dot(class_1 - m_1)
    # inner class2 dispersion
    S_2 = (class_2 - m_2).T.dot(class_2 - m_2)
    # seperating level
    S_W = S_1 + S_2
    
    # perpendicular vector of the separating hyperplane
    w_star = np.linalg.inv(S_W.astype(float)).dot(m_2 - m_1)
    
    mapped_b = class_1.dot(w_star)
    m_t_b = mapped_b.mean()
    mapped_s = class_2.dot(w_star)
    m_t_s = mapped_s.mean()

    # intersection point of the separating hyperplane and the perpendicular vector 
    w0 = -0.5*(m_t_b + m_t_s)
    
    return np.insert(w_star, 0, w0), np.nan



## __CODES PROVIDED BY TA AND FUNCTIONS FROM PREVIOUS HOMEWORKS__ ## 

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                for k in range(k_fold)]
    return np.array(k_indices)

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w', newline='\n') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    w=w.reshape(w.shape[0],1)
    y=y.reshape(y.shape[0],1)
    for n_iter in range(max_iters):
        gradient,loss=compute_gradient(y,tx,w)
        w=w-gamma*gradient
    return w,loss

    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size):
    w=initial_w
    w=w.reshape(w.shape[0],1)
    for n_iter in range(max_iters):
        gradient,loss=compute_stoch_gradient(y,tx,w,batch_size)
        w=w-gamma*gradient
   # w = initial_w
  #  w=w.reshape(w.shape[0],1)
  #  for n_iter in range(max_iters):
       # for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            #print('y_batch',y_batch.shape)
            #print('tx_batch',tx_batch.shape)
      #      grad, _ = compute_stoch_gradients(y_batch, tx_batch, w)
            #print('grad',grad)
            # update w through the stochastic gradient update
     #       w = w - gamma * grad
            # calculate loss
    #loss = compute_loss(y, tx, w)
    #loss=1
    return w,loss

def compute_stoch_gradients(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares(y, tx):
    w=np.linalg.inv(tx.T@tx)@tx.T@y
    w=w.reshape(w.shape[0],1)
    e=y-tx@w
    loss=e.T@e/2/len(y) 
    return w,loss

def ridge_regression(y, tx, lambda_):
    w=np.linalg.inv(tx.T@tx+2*len(y)*lambda_*np.eye(tx.shape[1]))@tx.T@y;
    w=w.reshape(w.shape[0],1)
    e=y-tx@w
    loss=e.T@e/2/len(y)
    return w,loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    w=w.reshape(w.shape[0],1)
    #print("max_iter",max_iters)
    for n_iter in range(max_iters):
        gradient,loss=compute_logis_gradient(y,tx,w)
        w=w-gamma*gradient
    return w,loss

def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    w = initial_w
    w=w.reshape(w.shape[0],1)
    for n_iter in range(max_iters):
        gradient,loss=compute_reg_logis_gradient(y,tx,w,lambda_)
        w=w-gamma*gradient
    return w,loss

def build_model_data(data,degree):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(data)
    r_data=data.copy()

    t_data=data
    t_data = np.c_[np.ones(num_samples), r_data]
    for i in range(len(r_data.T)):
        for j in range(2,degree+1):
            t_data=np.c_[t_data,r_data[:,i]**(j)]
    return t_data

def compute_logis_gradient(y,tx,w):
    e=y-sigma(tx@w)
    loss=np.sum(np.log(1+np.exp(tx@w)))-y.T@(tx@w)
    #s=tx@w
    gradient=-tx.T@e/len(y)
    return gradient,loss

def compute_reg_logis_gradient(y,tx,w,lambda_):
    e=y-sigma(tx@w);
    loss=np.sum(np.log(1+np.exp(tx@w)))-y.T@(tx@w)+lambda_*w.T@w/2
    gradient=-tx.T@e/len(y)+lambda_*w;
    #print("loss",loss)
    return gradient,loss

def sigma(a):
    if a.shape[0]<100:
        return 1/(1+np.exp(-a))
    else:
        a1=sigma(a[:int(a.shape[0]/2)+1])
        a2=sigma(a[int(a.shape[0]/2+1):])
        res=np.concatenate((a1,a2))
        return res
    
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e=y-tx@w;
    loss=e.T@e/(2*len(y));
    gradient=-tx.T@e/len(y);
    return gradient,loss   
    
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w,batch_size):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    for shuffled_y,shuffled_tx in batch_iter(y,tx,batch_size=1, num_batches=1):
       # shuffled_tx=shuffled_tx[0].reshape(shuffled_tx[0].shape[0],1)
       # shuffled_tx=shuffled_tx[0]
        e=shuffled_y-shuffled_tx@w
        loss=e.T@e/(2*len(shuffled_tx))    
        gradient=-shuffled_tx.T@e/len(shuffled_tx)    
    return gradient,loss



train_path = "train.csv"
test_path  = "test.csv"

yb_train,data_train,ids_train = load_csv_data(train_path, sub_sample=False)
test_label, test_data, ids_test = load_csv_data(test_path, sub_sample=False)

# all parameters are the same as those are used to produce the uploaded results,
# but the accuracy mighted be slightly different from uploaded one since there are several random parameters

algrithm      =  1         # algorithm to be used
                           # 1 -- Least Squrare
                           # 2 -- Ridge Regression
                           # 3 -- gradient descent
                           # 4 -- Logistic Regression
                           # 5 -- stochastic gradient descent
                           # 6 -- Reg_Logistic Regression
                           # ... --  8 is reserved for an other algo
                           # 7 -- Fisher Linear Discriminant
cros_slice    =  5         # slice numbers to conduct cross validation
degree        =  1         # the highest degree when polynomial items are append during feature augmentation
gamma         =  1e-7      # step_size used in iterative algorithms
lambda_       =  0.1       # coefficiency used for ridge regression 
method        =  'no'      # operation to be implemented on raw data to deal with outliers
                           # all methods: 'mean', 'normalization', 'drop', 'no'
outlier       =  -999

mean_accuracy, best_w, loss = training_model(data_train     , yb_train   , 
                                       algrithm       , cros_slice , 
                                       degree         , lambda_    , 
                                       gamma          , outlier    , 
                                       method)
print('w=',best_w)
print('loss=',loss)