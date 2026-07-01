import os
import re
import shap
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn

# from torch.utils.data import TensorDataset, DataLoader

# Set threading env
os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

warnings.filterwarnings("ignore")

# Global constants
BATCH_SIZE = 128
EPOCHS = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architectures
ARCHITECTURES = {
        "arch1": {"layers": [512, 256, 64], "dropouts": [0.25, 0.25, 0.10]},
        "arch2": {"layers": [1000, 500], "dropouts": [0.25, 0.10]},
        "arch3": {"layers": [2000, 1000], "dropouts": [0.25, 0.10]},
        "arch4": {"layers": [200, 100, 50], "dropouts": [0.25, 0.25, 0.10]},
        "arch5": {"layers": [4000, 2000, 1000, 1000], "dropouts": [0.25, 0.25, 0.25, 0.10]},
        }

class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim, layer_sizes, dropouts):
        super(FCNN, self).__init__()
        layers = []
        in_dim = input_dim
        for out_dim, p in zip(layer_sizes, dropouts):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(p),
                nn.BatchNorm1d(out_dim)
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

##############################################################################

def load_model(model_path, X_train):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract architecture name from filename
    arch_name_match = re.search(r'_fold\d+_(arch\d+)_FCNN\.pt$', os.path.basename(model_path))
    if not arch_name_match:
        raise ValueError(f"Cannot parse architecture name from {model_path}")
    arch_name = arch_name_match.group(1)
    config = ARCHITECTURES[arch_name]

    model = FCNN(input_dim=X_train.shape[1], output_dim=1,
            layer_sizes=config["layers"], dropouts=config["dropouts"]).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, arch_name

def evaluate_model(model, X, y_test, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).cpu().numpy().flatten()

    return {
            'R2': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'PearsonR': pearsonr(y_test, y_pred)[0],
            'SpearmanRho': spearmanr(y_test, y_pred)[0]
            }, y_pred

def parse_filename(filename):
    match = re.search(r"_(\d+)_train_(\w+)_fold(\d+)_", filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    outer_fold = int(match.group(1))
    split = match.group(2)
    inner_fold = int(match.group(3))
    return outer_fold, split, inner_fold
 
def plot_results(y_true, y_pred, out_prefix):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    # plt.title(f"{model_name} Predictions")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_scatter.png", dpi=300)
    plt.close()

def generate_shap_summary_plot(model, X, out_prefix):
    try:
        explainer = shap.DeepExplainer(model, torch.tensor(X, dtype=torch.float32))
        shap_values = explainer.shap_values(torch.tensor(X, dtype=torch.float32))
        # SHAP dot summary plot
        shap.summary_plot(shap_values, X.iloc, show=False)
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
        if not (fname.endswith(f"_{model_type}.pt") and fname.startswith(data_prefix)):
        # if not fname.endswith(f"_{model_type}.pt"):
            continue

        model_path = os.path.join(model_dir, fname)
        outer_fold, split, inner_fold = parse_filename(fname)
        scaler_path = os.path.join(model_dir, f"{data_prefix}_{outer_fold}_train_{split}_fold{inner_fold}_scaler.save")

        test_file = f"{data_prefix}_{outer_fold}_test_{split}.csv"
        train_file = f"{data_prefix}_{outer_fold}_train_{split}.csv"
        test_path = os.path.join(test_data_dir, test_file)
        train_path = os.path.join(test_data_dir, train_file)

        if not os.path.exists(test_path):
            print(f"Missing test file: {test_path}, skipping.")
            continue

        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Error loading {scalar_path}: {e}")
            continue

        # Train data
        data_df = pd.read_csv(train_path)
        data_df.dropna(axis=0, how="any", inplace=True)   # drop rows only

        drop_cols = ["Name", "Parent", "SMILES", "Mol", "Standard_SMILES", "Cluster_ID", "Scaffold_ID"]
        data_df.drop(columns=[c for c in drop_cols if c in data_df.columns], inplace=True)

        Y_train = data_df[str(data_column)]
        X_train_df = data_df.drop(columns=[str(data_column)])

        # Save feature schema
        feature_cols = X_train_df.columns.tolist()

        # Scale AFTER schema is fixed
        X_train = scaler.transform(X_train_df)


        try:
            model, arch_name = load_model(model_path, X_train)
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            continue


        df = pd.read_csv(test_path)
        df.dropna(axis=0, how="any", inplace=True)   # SAME rule as train

        y = df[str(data_column)]

        drop_cols = ["Parent", "SMILES", "Mol", "Standard_SMILES", "Cluster_ID", "Scaffold_ID", str(data_column)]
        name_smiles_df = df[[c for c in ["Name", "SMILES"] if c in df.columns]].copy()

        X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X_df = X_df.drop(columns=[c for c in ["Name", "SMILES"] if c in X_df.columns])

        # Enforce the training schema
        X_df = X_df.reindex(columns=feature_cols, fill_value=0.0)

        # Scale
        X = scaler.transform(X_df)

        try:
            metrics, y_pred = evaluate_model(model, X, y, device)
        except Exception as e:
            print(f"Error evaluating {fname}: {e}")
            continue

        metrics.update({
            'architecture': arch_name,
            'outer_fold': outer_fold,
            'inner_fold': inner_fold,
            'split': split,
            'basename': fname
        })
        records.append(metrics)

        # Save prediction vs true data
        result_df = name_smiles_df.copy()
        result_df['True'] = y
        result_df['Predicted'] = y_pred
        result_df['Absolute_Error'] = np.abs(y - y_pred)

        output_dir_er = os.path.join(os.getcwd(), "error")
        os.makedirs(output_dir_er, exist_ok=True)
        output_error_csv = os.path.join(output_dir_er, f"{data_prefix}_{outer_fold}_test_{split}_fold{inner_fold}_{arch_name}_predicted.csv")
        result_df.to_csv(output_error_csv, index=False)

        # Save Yellowbrick and SHAP plots
        output_dir_yb = os.path.join(os.getcwd(), "yb")
        os.makedirs(output_dir_yb, exist_ok=True)

        plot_results(y, y_pred, os.path.join(output_dir_yb, f"{data_prefix}_{outer_fold}_test_{split}_fold{inner_fold}_{arch_name}"))
        if shap_on:
            output_dir_shap = os.path.join(os.getcwd(), "shap")
            os.makedirs(output_dir_shap, exist_ok=True)
            generate_shap_summary_plot(model, X_train, os.path.join(output_dir_shap, f"{data_prefix}_{outer_fold}_test_{split}_fold{inner_fold}_{arch_name}"))

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
