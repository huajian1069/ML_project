#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import helpers


# In[ ]:


def init_MF(train, n_features):
    """Initialize the parameters for matrix factorization."""
    n_items, n_users = train.shape

    user_features = 2.5 * np.random.rand(int(n_features), n_users)
    item_features = 2.5 * np.random.rand(n_items, int(n_features))

    return user_features, item_features


# In[ ]:


def matrix_factorization_SGD(train, test, n_features, lambda_user, lambda_item, fold_num):
    """matrix factorization by SGD."""
    
    print(
        '--->Starting SGD with n_features = %d, lambda_user = %f, lambda_item = %f, the %d-th fold'
        % (n_features, lambda_user, lambda_item, fold_num)
    )
    
    # define parameters
    gamma = 0.01
    n_epochs = 50
    prev_train_rmse = 100
    prev_train_rmse = 100
    
    user_features_file_path = 'Learningdata/user_features_%s_%s_%s_%s_%s_%s.npy'  \
                                % ('sgd', fold_num, n_epochs, n_features, lambda_user, lambda_item)

    item_features_file_path = 'Learningdata/item_features_%s_%s_%s_%s_%s_%s.npy'  \
                                % ('sgd', fold_num, n_epochs, n_features, lambda_user, lambda_item)

    if (os.path.exists(user_features_file_path) and
            os.path.exists(item_features_file_path)):
        
        user_features = np.load(user_features_file_path)
        item_features = np.load(item_features_file_path)

        train_rmse = helpers.calculate_rmse(
            np.dot(item_features, user_features)[train.nonzero()],
            train[train.nonzero()][0]
        )

        test_rmse = helpers.calculate_rmse(
            np.dot(item_features, user_features)[test.nonzero()],
            test[test.nonzero()][0]
        )

        print("      Train error: %f, test error: %f" % (train_rmse, test_rmse))

        return user_features, item_features, train_rmse, test_rmse
    
    else:
        try:
            os.mkdir('Learningdata/')
            print('\n The folder, "Learningdata", is created to store learning results.\n')
        except:
            print('\n New data will be saved to the "Learningdata" folder. \n')
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, n_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()

    for it in range(n_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[d, :]
            user_info = user_features[:, n]
            err = train[d, n] - user_info@item_info
    
            # calculate the gradient and update
            item_features[d, :] += gamma * (err * user_info - lambda_item *item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user *user_info)
        
        # evaluate the train error
        train_rmse = helpers.calculate_rmse(
            np.dot(item_features, user_features)[train.nonzero()],
            train[train.nonzero()][0]
        )
        
        # evaluate the test error
        test_rmse = helpers.calculate_rmse(
            np.dot(item_features, user_features)[test.nonzero()],
            test[test.nonzero()][0]
        )
        print("      [Epoch %d / %d] train error: %f, test error: %f" % (it + 1, n_epochs, train_rmse, test_rmse))
        if (train_rmse > prev_train_rmse or abs(train_rmse - prev_train_rmse) < 1e-5):
            print('SGD Algorithm has converged!')
            break
            
        prev_train_rmse = train_rmse
        prev_test_rmse  = test_rmse
        
    np.save(user_features_file_path, user_features)
    np.save(item_features_file_path, item_features)
    
    return user_features, item_features, prev_train_rmse, prev_test_rmse  # output the final rmse


# In[ ]:


def get_SGD_predictions(ratings, train, test, lambda_user, lambda_item, features_num, fold_num = 0, kwargs = None):
    """Return differents predictions corresponding to the given parameters

    Args:
        ratings (n_users x n_itens): The global dataset.
        train (n_users x n_items): The train dataset.
        test (n_users x n_items): The test dataset.
        features_num (N): Array representing the n_features parameter for the
                          different models to compute.
        lambda_user: This value is for all the models.
        lambda_item: This value is for all the models.

    Returns:
        X (n_users x n_items): Returns the global predictions for all the models.
        X_train: Returns the predictions for the non zero values of the train dataset.
        y_train: Returns the true labels for the train dataset.
        X_test: Returns the predictions for the non zero values of the test dataset.
        y_test: Returns the true labels for the test dataset.
    """
    n_models = features_num.size

    X = np.zeros((n_models, train.shape[0] * train.shape[1]))
    X_train = np.zeros((n_models, train.nnz))
    X_test = np.zeros((n_models, test.nnz))

#    y_train = ratings[train.nonzero()].toarray()[0]
#    y_test = ratings[test.nonzero()].toarray()[0]

    for idx, n_features in enumerate([features_num]):
        user_features, item_features, train_rmse, test_rmse = matrix_factorization_SGD(
            train, test, n_features, lambda_user, lambda_item, fold_num
        )

        predicted_labels = np.dot(item_features, user_features)
        predicted_labels[predicted_labels > 5] = 5
        predicted_labels[predicted_labels < 1] = 1

        X[idx] = np.asarray(predicted_labels).reshape(-1)
        X_train[idx] = predicted_labels[train.nonzero()]
        X_test[idx] = predicted_labels[test.nonzero()]

    return X, train_rmse, test_rmse

