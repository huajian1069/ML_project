


from blending import blending,auto_load_data
from ALS import *
from baseline import *
ratings,train,test=auto_load_data()



X, X_train, y_train, X_test, y_test = get_ALS_predictions(
    ratings,
    train,
    test,
    n_features_array=range(1,31),
    lambda_user=0.2,
    lambda_item=0.02
)

baseline_item_mean(train, test)

blending(test)

