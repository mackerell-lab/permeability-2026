from stat import *
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, GridSearchCV, KFold, cross_val_score,RepeatedKFold,train_test_split
import os
import argparse
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# n_jobs_model = -1
# n_jobs_cv = -1
n_jobs_model = 4
n_jobs_cv = 4

# Define custom metrics
def pearson_r(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def spearman_rho(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]

# Define scoring dictionary
scoring = {
    'MAE': 'neg_mean_absolute_error',
    'MSE': 'neg_mean_squared_error',
    'R2': 'r2',
    'PearsonR': make_scorer(pearson_r, greater_is_better=True),
    'Rho': make_scorer(spearman_rho, greater_is_better=True)
}

def model_validation(X, y, model, n_splits=5, n_repeats=5, random_state=128):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    results = cross_validate(model, X, y, scoring=scoring, cv=rkf, return_train_score=False)

    # Convert to DataFrame and correct sign on neg MAE/MSE
    results_df = pd.DataFrame(results)
    for metric in ['test_MAE', 'test_MSE']:
        if metric in results_df.columns:
            results_df[metric] = -results_df[metric]
    
    return results_df

def plot_learning_curve(model, X_train, Y_train, prefix="learning_curve"):
    # cross-validation
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=128) # 5-fold cross validation for 5 times
    train_sizes, train_scores, test_scores = learning_curve(model,
                                                            X_train,
                                                            Y_train,
                                                            cv=rkf,
                                                            scoring='r2',
                                                            n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            shuffle=True,
                                                            random_state=42)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_scores_mean, label="Training Score", marker='o')
    plt.plot(train_sizes, test_scores_mean, label="Validation Score", marker='s')
    plt.xlabel("Training Set Size")
    plt.ylabel("r2")
    # plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(prefix + ".png")

def validation(raw_data, property, ml_model, model_prefix=None):
    basename = os.path.splitext(os.path.basename(raw_data))[0]
    data_df = pd.read_csv(raw_data)

    # Drop column "Name" from the dataset for optimization
    # if there are more labels in the dataset that need to be 
    # droped, here is the place
    data_df.drop(columns=["Name"], inplace=True) 
    if "Parent" in data_df.columns:
        data_df.drop(columns=["Parent"], inplace=True) 
    # if "SMILES" in data_df.columns:
    #    data_df.drop(columns=["SMILES"], inplace=True) 

    data_df.dropna(axis=0, how='any', inplace=True) 
    Y_data = data_df[property]
    X_data = data_df
    X_data.drop(property, axis=1, inplace=True) 
    X_data, Y_data = shuffle(X_data, Y_data, random_state=42)     
    # split to training set and hold-out test set
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=84)  

    if model_prefix == None:
        model_file = f'{basename}_{ml_model}.rds'
    else:
        model_file = f'{model_prefix}_{ml_model}.rds'
        
    ## loaded pre-trained ADME models
    try:
        loaded_model = joblib.load(model_file)
    except:
        print(f'There are no pre-trained {model_file}')

    # Hypterparamter tuning for ML models
    if ml_model == 'RF':

        plot_learning_curve(model, X_train, Y_train, prefix=f'learning_curve_{ml_model}')

        # validation of the optimized model
        cv_pearson_r, r_test = model_validation(X_train, Y_train, X_test, Y_test, model)
                
    elif ml_model == 'XGBoost':

        plot_learning_curve(model, X_train, Y_train, prefix=f'learning_curve_{ml_model}')

        # validation of the optimized model
        cv_pearson_r, r_test = model_validation(X_train, Y_train, X_test, Y_test, model)
        
    elif ml_model == 'LightGBM':

        plot_learning_curve(model, X_train, Y_train, prefix=f'learning_curve_{ml_model}')

        # validation of the optimized model    
        cv_pearson_r, r_test = model_validation(X_train, Y_train, X_test, Y_test, model)

    elif ml_model == 'SVM_linear':
    
        plot_learning_curve(model, X_train_scaled, Y_train, prefix=f'learning_curve_{ml_model}')
            
        # validation of the optimized model
        cv_pearson_r, r_test = model_validation(X_train_scaled, Y_train, X_test_scaled, Y_test, model)

    elif ml_model == 'SVM_rbf':
    
        plot_learning_curve(model, X_train_scaled, Y_train, prefix=f'learning_curve_{ml_model}')
            
        # validation of the optimized model
        cv_pearson_r, r_test = model_validation(X_train_scaled, Y_train, X_test_scaled, Y_test, model)
        
    elif ml_model == 'Lasso':

        plot_learning_curve(model, X_train_scaled, Y_train, prefix=f'learning_curve_{ml_model}')

        # validation of the optimized model    
        cv_pearson_r, r_test = model_validation(X_train_scaled, Y_train, X_test_scaled, Y_test, model)    

    else:
        print("The ML algorithms should be one of RF, SVM_linear, SVM_rbf, XGBoost, LightGBM, and Lasso")  


if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Optimize different ML models based on the input CSV file")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("property", default="LOG10 MDR1-MDCK Papp_A2B (cm/s)", help="Property Name")
    parser.add_argument("model_prefix", default=None, help="If you want to test models trained on other databases, use the prefix for those models")

    # Parse arguments
    args = parser.parse_args()

    validation_result = pd.DataFrame(columns=['ml_model', 'Pearson_r_CV', 'Pearson_r_test'])

    for model in ["RF", "SVM_linear", "SVM_rbf", "XGBoost", "LightGBM", "Lasso"]:  # "RF" 
        print ("Optimizing %s model..."%model)
        validation(args.input_csv, args.property, ml_model=model, model_prefix=args.model_prefix)

        validation_result['ml_model'] = [ml_model]
        validation_result['Pearson_r_CV'] = [cv_pearson_r]
        validation_result['Pearson_r_test'] = [r_test]
        validation_result.to_csv('%s_validation_results.csv'%(basename),index=True, sep=',')  
