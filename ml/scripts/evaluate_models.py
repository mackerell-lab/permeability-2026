import os
import re
import shap
import joblib
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from yellowbrick.model_selection import LearningCurve
from yellowbrick.regressor import PredictionError, ResidualsPlot

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'PearsonR': pearsonr(y_test, y_pred)[0],
        'SpearmanRho': spearmanr(y_test, y_pred)[0]
    }, y_pred

def parse_filename(filename):
    # This regex assumes format: <prefix>_<outer>_train_<split>_fold<inner>_<model>.rds
    match = re.search(r"_(\d+)_train_(\w+)_fold(\d+)_", filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    outer_fold = int(match.group(1))
    split = match.group(2)
    inner_fold = int(match.group(3))
    return outer_fold, split, inner_fold

def generate_yellowbrick_plots(model, X_test, Y_test, X_train, Y_train, out_prefix):
    try:
        # Prediction Error
        pe = PredictionError(model)
        pe.fit(X_train, Y_train)
        pe.score(X_test, Y_test)
        pe.show(outpath=f"{out_prefix}_prediction_error.png", clear_figure=True)

        # Residuals Plot
        rp = ResidualsPlot(model)
        rp.fit(X_train, Y_train)
        rp.score(X_test, Y_test)
        rp.show(outpath=f"{out_prefix}_residuals.png", clear_figure=True)

        # Learning Curve
        lc = LearningCurve(model, scoring='r2', cv=5)
        lc.fit(X_train, Y_train)
        lc.show(outpath=f"{out_prefix}_learning_curve.png", clear_figure=True)
    except Exception as e:
        print(f"Yellowbrick plot error for {out_prefix}: {e}")

def generate_shap_summary_plot(model, X_train, Y_train, out_prefix):
    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        # SHAP dot summary plot
        shap.summary_plot(shap_values, X_train, show=False)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_shap_summary_dot.png")
        plt.close()

        # SHAP bar plot with clustering - extremly slow
        # clust = shap.utils.hclust(X_train, Y_train, linkage="single")
        # shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=1, show=False)
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_shap_summary_bar.png")
        plt.close()

    except Exception as e:
        print(f"SHAP plot error for {out_prefix}: {e}")

def main(model_type, model_dir, data_prefix, data_column, test_data_dir, output_prefix, shap_on=False):
    records = []
    for fname in sorted(os.listdir(model_dir)):
        if not (fname.endswith(f"_{model_type}.rds") and fname.startswith(data_prefix)):
            continue

        model_path = os.path.join(model_dir, fname)
        outer_fold, split, inner_fold = parse_filename(fname)

        test_file = f"{data_prefix}_{outer_fold}_test_{split}.csv"
        train_file = f"{data_prefix}_{outer_fold}_train_{split}.csv"
        test_path = os.path.join(test_data_dir, test_file)
        train_path = os.path.join(test_data_dir, train_file)

        if not os.path.exists(test_path):
            print(f"Missing test file: {test_path}, skipping.")
            continue

        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            continue

        df = pd.read_csv(test_path)
        df.dropna(axis=0, how='any', inplace=True)
        y = df[str(data_column)]

        drop_cols = ["Name", "Parent", "SMILES", "Mol", "Standard_SMILES", "Cluster_ID", "Scaffold_ID", str(data_column)]

        # Retain Name and SMILES separately for output
        name_smiles_df = df[[col for col in ["Name", "SMILES"] if col in df.columns]].copy()
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        if "Name" in X.columns: X = X.drop(columns=["Name"])
        if "SMILES" in X.columns: X = X.drop(columns=["SMILES"])

        # Train data
        data_df = pd.read_csv(train_path)

        # Drop unnecessary columns
        drop_cols = ["Name", "Parent", "SMILES", "Mol", "Standard_SMILES", "Cluster_ID", "Scaffold_ID"]
        # drop_cols = ["Parent", "SMILES", "Mol", "Standard_SMILES", "Cluster_ID", "Scaffold_ID", str(data_column)]
        data_df.drop(columns=[col for col in drop_cols if col in data_df.columns], inplace=True)
        data_df.dropna(axis=0, how='any', inplace=True)

        Y_train = data_df[str(data_column)]
        data_df.drop(str(data_column), axis=1, inplace=True) 
        X_train = data_df 

        # test alignment
        X = X.reindex(columns=X_train.columns, fill_value=0.0)
        assert list(X.columns) == list(X_train.columns)

        try:
            metrics, y_pred = evaluate_model(model, X, y)
        except Exception as e:
            print(f"Error evaluating {fname}: {e}")
            continue

        metrics.update({
            'outer_fold': outer_fold,
            'inner_fold': inner_fold,
            'split': split,
            'model_file': fname
        })
        records.append(metrics)

        # Save prediction vs true data
        result_df = name_smiles_df.copy()
        result_df['True'] = y.values
        result_df['Predicted'] = y_pred
        result_df['Absolute_Error'] = np.abs(y - y_pred)

        output_dir_er = os.path.join(os.getcwd(), "error")
        os.makedirs(output_dir_er, exist_ok=True)
        output_error_csv = os.path.join(output_dir_er, f"{data_prefix}_{outer_fold}_test_{split}_fold{inner_fold}_predicted.csv")
        # output_error_csv = f"{data_prefix}_{outer_fold}_test_{split}_fold{inner_fold}_predicted.csv"
        result_df.to_csv(output_error_csv, index=False)

        # Save Yellowbrick and SHAP plots
        output_dir_yb = os.path.join(os.getcwd(), "yb")
        os.makedirs(output_dir_yb, exist_ok=True)

        generate_yellowbrick_plots(model, X, y, X_train, Y_train, os.path.join(output_dir_yb, f"{data_prefix}_{outer_fold}_test_{split}_fold{inner_fold}"))
        if shap_on:
            output_dir_shap = os.path.join(os.getcwd(), "shap")
            os.makedirs(output_dir_shap, exist_ok=True)
            generate_shap_summary_plot(model, X_train, Y_train, os.path.join(output_dir_shap, f"{data_prefix}_{outer_fold}_test_{split}_fold{inner_fold}"))

    df_results = pd.DataFrame(records)
    df_results.to_csv(f"{output_prefix}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation of trained models")
    parser.add_argument("--model_type", required=True, help="Model name, e.g. RF, SVM_rbf")
    parser.add_argument("--model_dir", required=True, help="Directory containing trained .rds models")
    parser.add_argument("--data_prefix", required=True, help="Prefix of train/test data - Both have to be same")
    parser.add_argument("--data_column", required=True, help="Column name of label (Y)")
    parser.add_argument("--test_data_dir", required=True, help="Directory containing test CSV files")
    parser.add_argument("--output_prefix", required=True, help="Where to save output")

    # Mutually exclusive SHAP flag
    shap_group = parser.add_mutually_exclusive_group()
    shap_group.add_argument("--shap_on", dest="shap", action="store_true", help="Enable SHAP plots")
    shap_group.add_argument("--no_shap", dest="shap", action="store_false", help="Disable SHAP plots")
    parser.set_defaults(shap=False)  # Default: SHAP off

    args = parser.parse_args()

    main(args.model_type,
        args.model_dir,
        args.data_prefix,
        args.data_column,
        args.test_data_dir,
        args.output_prefix,
        args.shap
    )
