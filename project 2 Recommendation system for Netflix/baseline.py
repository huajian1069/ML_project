import numpy as np



def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    num_items, num_users = train.shape
    prediction = np.zeros((10000, 1000))
    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = train[:, user_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]
        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_mean = nonzeros_train_ratings.mean()
        else:
            user_train_mean = 3
        prediction[:,user_index] = user_train_mean
    np.save('dump/0_full_usermean.npy',prediction.reshape(-1))
    np.save('dump/0_test_usermean.npy',prediction[test.nonzero()])
    return prediction


def baseline_item_mean(train, test):
    """baseline method: use the user means as the prediction."""
    num_items, num_users = train.shape
    prediction = np.zeros((10000, 1000))
    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[item_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]
        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean = nonzeros_train_ratings.mean()
        else:
            item_train_mean = 3
            print('haha')
        prediction[item_index, :] = item_train_mean
    np.save('dump/41_full_itemmean.npy',prediction.reshape(-1))
    np.save('dump/41_test_itemmean.npy',prediction[test.nonzero()])
    return prediction
