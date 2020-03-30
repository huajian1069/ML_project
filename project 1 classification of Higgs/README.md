# Repo for Machine Learning 2018 Homework 01 

## Architecture of the directory

The directory contains 4 files:
	
	run.py -- script to reproduce our results.

	MLproject1_TeamAvatar.ipynb -- calling functions from following two files to accomplish a classification task.

	Lib_Models.py -- functions related to different models, for example: 'Fisher_classifier', 'train_ridge' and 'predict_labels'.

	Lib_Data_Processing.py -- functions related to data processing, for example: 'add_feature', 'fill_mean' and 'normalization'.


## Architecture of the code

Our code have 4 main steps: data loading, feature augmentation, cross validation, testing.

In data loading, we import data from corresponding csv files and rephrase labels to be -1 ('b') and +1 ('s').

In feature augmentation, there are two steps. Firstly ( in function 'add_feature()' ), nonlinear features are appended to original data by apply cosine function, logrithmic function and exponential function on original features. Then (in function 'build_model_data()'), polinomial terms are added to the data. In fact, the greatest degree appended to the data can be set through the input parameter 'degree'. In the run.py,  the greatest degree is set to 2.  

After feature augmentation, cross validation is conducted. With the help of 'build_k_indices()', several groups of indices are generated and are sent to cross validation to generate taining data and testing data. Then, a for loop is created. During each loop, outliers in training and testing data are processed. At last, the processed traing data are use to train a classifier and the corresponding testing data are used to test the classifier.

The final step is testing. Same as steps above, logrithmic, exponential, cosine and polinomial features are appanded to testing data. Then outliers in testing data are processed in similar ways as those are conducted on training data. Finally, testing data are sent to 'predict_labels()' and its prediction are written to csv by create_csv_submission().


## To reproduce our results

Executing the run.py can reproduce our results. In order to make it more understandable for others to reproduce our results, in the third and fourth cell in MLproject1_TeamAvatar.ipynb, we have set all necessary parameters, such as cross-validation size, data processing methods and algorithm that we used. So, executing the first 4 cells is enough to reproduce our results.

To make it convenient for people to test different parameters and different models, from the 5th cell to the end in MLproject1_TeamAvatar.ipynb, we provide a templet. Moreover, we provide details of our functions as their docstring. Below is a brief introduction of input parameters:

algrithm  	   # algorithm to be used
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


cros_slice     # Slice numbers to conduct cross validation
degree         # The highest degree when polynomial items are append during feature augmentation
gamma          # Step_size used in iterative algorithms
lambda_        # Coefficiency used for ridge regression 
method         # Operation to be implemented on raw data to deal with outliers
                   # All methods: 'mean', 'normalization', 'drop', 'no'
outlier        # Outliers to be replaced
