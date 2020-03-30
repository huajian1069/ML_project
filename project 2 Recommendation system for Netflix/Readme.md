# Recommendation system 
This project attempts to predict missing ratings given a sparse rating matrix. We try to solve this problem with various algorithms and build our final model based on Matrix Factorization ALS.

## 1. Prerequisites
### Anaconda
This project is based on anaconda and jupiter notebook. Download and install instructions can be found at: https://www.anaconda.com/download/. After installing Anaconda, `pip` and `conda` can be used to install Python packages.

### Scipy
We use the scientific computing and visualization functionalities of [scipy](https://www.scipy.org/install.html), especially the numpy, pandas and matplotlib package. These packages can be installed by typing the following command in your Anaconda Prompt.
```
python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```

### Seaborn
We use [Seaborn](https://seaborn.pydata.org/) to visualize data. The package can be installed by typing the following command in your Anaconda Prompt.
```
pip install seaborn
```

### Surprise
We use [Surprise](http://surpriselib.com/) to build KNN, SVD and Coclustering algorithm. The scikit can be installed by typing the following command in your Anaconda Prompt.
```
pip install scikit-surprise
```

### SKLearn(scikit-learn)
We use [SKLearn](https://scikit-learn.org/stable/) to blending different models. The package can be installed by typing the following command in your Anaconda Prompt.
```
conda install scikit-learn
```

## 2.Architecture of the directory
The main modules are listed below:
|   Modules   |    Function   |
|-------------|:-------------|
| run         |Script to reproduce our results.|
| main        | The main function, all parameters are set in this file.|
| bodyfunctions | Providing functions to search parameters and test models |
| CrosVali    | Serving for cross validation. |
| SGD         | Using SGD as the solver to solve recommendation system problem. |
| ALS         | Inclding functions using ALS as the solver to solve recommendation system problem.|
| ALS_ours    | Modified ALS, but it does not outperform the ALS. |
| SurpriseLib | To generate different models for blending, [surpriseLib](http://surpriselib.com/) is used. This file contains the functions we use and the output data are properly orgnized.|
|recommendation_system| A simple Top-N (default 5) movies recommendation system|
| Helpers     | This is the same Helper of exercise 10. |

## Architecture of the code
All parameters are initiatialized in the main.py, and two modes are provided to test the code:
- Grid search
    - Allocating each parameter a range of number to be searched by setting the "kwargs";
    - Chooseing grid search function through setting "func" to be "grid_search";
    - run the main. py file;
- Model testing
    - Setting the parapmeters of the corrresponding solver in "kwargs";
    - Setting the "func" to be "test";
    - run the main. py file
- Top N movies recommendation system 
    - Loading the final prediction from "data" folder;
    - Given a user index, the program will recommend the top N (default 5) movies;

At first, grid search and cross validation is implemented to find the best parameters for our models. Then, we test several 'good' parameters and save their predictions in a folder. Finally, we blend these models and generate the submission file. 

## To reproduce the results
To reproduce the result, firstly, please unzip the run.zip. It is enough to execute the run.py in the unziped folder, and the 'submission.csv' will be save in "Data" folder.
