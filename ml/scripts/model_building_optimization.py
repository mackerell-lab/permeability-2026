import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold, KFold
import os
import argparse
import optuna
import optuna.visualization as vis
from functools import partial

# n_jobs_model = -1
# n_jobs_cv = -1
n_jobs_model = 4
n_jobs_cv = 4
n_jobs_study = 1

#  Define Optuna objective for ML models 

def objective_rf(trial, X_train, Y_train):
    # First, we randomly choose between continuous or categorical options
    max_depth_choice = trial.suggest_categorical('max_depth_choice', [None, 'continuous'])

    # Handle max_depth based on choice
    if max_depth_choice == 'continuous':
        max_depth = trial.suggest_int('max_depth', 1, 100)  # Allow continuous range of values for max_depth
    else:
        max_depth = None  # If None is chosen, no limit on tree depth

    max_features_choice = trial.suggest_categorical('max_features_choice', ['sqrt', 'None', 'continuous'])

    # Handle max_features based on choice
    if max_features_choice == 'continuous':
        max_features = trial.suggest_float('max_features', 0.1, 1.0)
    elif max_features_choice == 'sqrt':
        max_features = 'sqrt'
    else:
        max_features = None

    # Store the max_features manually
    trial.set_user_attr("max_features", max_features)

    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=50)

    # Create the RandomForest model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        random_state=42,
        n_jobs=n_jobs_model
    )

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=128) # 5-fold cross validation for 3 times
    score = cross_val_score(model, X_train, Y_train, scoring='r2', cv=rkf, n_jobs=n_jobs_cv)
    return score.mean()

def objective_xgb(trial, X_train, Y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 3.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.4),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 1.4)
    }

    model = xgb.XGBRegressor(
        **params,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0,
        n_jobs=n_jobs_model
    )

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=128) # 5-fold cross validation for 3 times
    score = cross_val_score(model, X_train, Y_train, scoring='r2', cv=rkf, n_jobs=n_jobs_cv)
    return score.mean()

def objective_lgb(trial, X_train, Y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=50),
        'num_leaves': trial.suggest_int('num_leaves', 5, 75, step=5),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 40, step=5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 5, step=1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.8),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.8)
    }

    model = lgb.LGBMRegressor(
        **params,
        objective='regression',
        random_state=42,
        n_jobs=n_jobs_model
    )

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=128) # 5-fold cross validation for 3 times
    score = cross_val_score(model, X_train, Y_train, scoring='r2', cv=rkf, n_jobs=n_jobs_cv)
    return score.mean() 

def objective_svm_linear(trial, X_train, Y_train):
    C = trial.suggest_float('C', 0.1, 50.0, log=True)
    epsilon = trial.suggest_float('epsilon', 0.01, 0.5, log=True)

    model = SVR(kernel="linear", C=C, epsilon=epsilon)

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=128) # 5-fold cross validation for 3 times
    score = cross_val_score(model, X_train, Y_train, scoring='r2', cv=rkf, n_jobs=n_jobs_cv)
    return score.mean()

def objective_svm_rbf(trial, X_train, Y_train):
    C = trial.suggest_float('C', 0.1, 50.0, log=True)
    epsilon = trial.suggest_float('epsilon', 0.01, 0.5, log=True)

    # RBF Kernal
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    model = SVR(C=C, epsilon=epsilon, gamma=gamma)

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=128) # 5-fold cross validation for 3 times
    score = cross_val_score(model, X_train, Y_train, scoring='r2', cv=rkf, n_jobs=n_jobs_cv)
    return score.mean()

def objective_lasso(trial, X_train, Y_train):
    alpha = trial.suggest_float('alpha', 0.001, 5.0, log=True)  # log-scale search from 0.001 to 5
    model = linear_model.Lasso(alpha=alpha, max_iter=10000, random_state=42)

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=128) # 5-fold cross validation for 3 times
    score = cross_val_score(model, X_train, Y_train, scoring='r2', cv=rkf, n_jobs=n_jobs_cv)
    return score.mean()

#################################################################################

def build_and_optimize(raw_data, property, ml_model):
    basename = os.path.splitext(os.path.basename(raw_data))[0]
    data_df = pd.read_csv(raw_data)

    # Drop unnecessary columns
    drop_cols = ["Name", "Parent", "SMILES", "Mol", "Standard_SMILES", "Cluster_ID", "Scaffold_ID"]
    data_df.drop(columns=[col for col in drop_cols if col in data_df.columns], inplace=True)
    data_df.dropna(axis=0, how='any', inplace=True)

    Y = data_df[property]
    data_df.drop(property, axis=1, inplace=True) 
    X = data_df 

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold_idx + 1}/5 ===")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        Y_train, Y_val = Y.iloc[train_index], Y.iloc[val_index]

        study = optuna.create_study(direction='maximize')

        # Hypterparamter tuning for ML models
        if ml_model == 'RF':

            study.optimize(partial(objective_rf, X_train=X_train, Y_train=Y_train), n_trials=100, n_jobs=n_jobs_study)

            # Save the best results ---
            best_params = study.best_params
            best_score = study.best_value

            feature_choice = best_params['max_features_choice']
            depth_choice = best_params['max_depth_choice']

            if feature_choice == 'continuous':
                max_features = best_params['max_features']
            elif feature_choice == 'sqrt':
                max_features = 'sqrt'
            else:  # 'none'
                max_features = None

            if depth_choice == 'continuous':
                max_depth = best_params['max_depth']
            else:  # 'none'
                max_depth = None

            # Apply best parameters
            model = RandomForestRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=max_depth,
                max_features=max_features,
                random_state=42,
                n_jobs=n_jobs_model
            )

            model.fit(X_train, Y_train)

        elif ml_model == 'XGBoost':

            study.optimize(partial(objective_xgb, X_train=X_train, Y_train=Y_train), n_trials=100, n_jobs=n_jobs_study)  # We can increase n_trials if needed

            # Save the best results ---
            best_params = study.best_params
            best_score = study.best_value

            # Apply best parameters
            model = xgb.XGBRegressor(
                **best_params,
                objective='reg:squarederror',
                random_state=42,
                verbosity=0,
                n_jobs=n_jobs_model
            )

            model.fit(X_train, Y_train)

        elif ml_model == 'LightGBM':

            study.optimize(partial(objective_lgb, X_train=X_train, Y_train=Y_train), n_trials=100, n_jobs=n_jobs_study)

            # Save results ---
            best_params = study.best_params
            best_score = study.best_value

            # Apply best parameters
            model = lgb.LGBMRegressor(
                **best_params,
                objective='regression',
                random_state=42,
                n_jobs=n_jobs_model
            )

            model.fit(X_train, Y_train)

        elif ml_model == 'SVM_linear':
        
            scaler = RobustScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train) 

            study.optimize(partial(objective_svm_linear, X_train=X_train_scaled, Y_train=Y_train), n_trials=100, n_jobs=n_jobs_study)

            # Save the best results ---
            best_params = study.best_params
            best_score = study.best_value 

            # Apply best parameters
            model = SVR(kernel="linear", **study.best_params)
            model.fit(X_train_scaled, Y_train)

        elif ml_model == 'SVM_rbf':
        
            scaler = RobustScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train) 

            study.optimize(partial(objective_svm_rbf, X_train=X_train_scaled, Y_train=Y_train), n_trials=100, n_jobs=n_jobs_study)

            # Save the best results ---
            best_params = study.best_params
            best_score = study.best_value 

            # Apply best parameters
            model = SVR(**study.best_params)
            model.fit(X_train_scaled, Y_train)
            
        elif ml_model == 'Lasso':

            scaler = RobustScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            
            study.optimize(partial(objective_lasso, X_train=X_train_scaled, Y_train=Y_train), n_trials=100, n_jobs=n_jobs_study)

            # Save results ---
            best_params = study.best_params
            best_score = study.best_value

            # Apply best parameters
            model = linear_model.Lasso(alpha=best_params['alpha'], max_iter=10000, random_state=42)
            model.fit(X_train_scaled, Y_train)

        else:
            print("The ML algorithms should be one of RF, SVM_linear, SVM_rbf, XGBoost, LightGBM, and Lasso")  

        output_dir = os.path.join(os.getcwd(), ml_model)

        # Save the model
        joblib.dump(model, os.path.join(output_dir, f'{basename}_fold{fold_idx+1}_{ml_model}.rds'))

        # Write out best parameter details
        with open(os.path.join(output_dir,f'hyperparameter_for_{basename}_fold{fold_idx+1}_{ml_model}_Bayesian_Tuned.txt'), 'w') as f:
                f.write(f"The best_params are {best_params} \n")
                f.write(f"The best_score of R2 is {best_score:.4f} \n\n")

        # Plot the optuna study details
        fig1 = vis.plot_optimization_history(study)
        fig2 = vis.plot_param_importances(study)

        fig1.write_html(os.path.join(output_dir,f"optuna_history_{basename}_fold{fold_idx+1}_{ml_model}.html"))
        fig2.write_html(os.path.join(output_dir,f"optuna_param_importance_{basename}_fold{fold_idx+1}_{ml_model}.html"))

if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Optimize different ML models based on the input CSV file")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("property", default="LOG10 MDR1-MDCK Papp_A2B (cm/s)", help="Property Name")

    # Parse arguments
    args = parser.parse_args()

    for model in ["XGBoost", "LightGBM"]:  # "SVM_linear", "RF", "SVM_rbf", "Lasso"
        # Create output directory named after the ML model
        output_dir = os.path.join(os.getcwd(), model)
        os.makedirs(output_dir, exist_ok=True)
        print ("Building and optimizing %s model..."%model)
        build_and_optimize(args.input_csv, args.property, ml_model=model)
