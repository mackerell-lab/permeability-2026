import os
import re
import argparse
import joblib
import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# warnings.filterwarnings("ignore")

# Set threading env
os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

# Global constants
BATCH_SIZE = 128
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"DEVICE : {DEVICE}")

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

def evaluate_model(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'PearsonR': pearsonr(y_true, y_pred)[0],
        'SpearmanRho': spearmanr(y_true, y_pred)[0]
    }

def parse_filename(filename):
    match = re.search(r"_(\d+)_train_(\w+)", filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    return int(match.group(1)), match.group(2)

def train_model(model, train_loader, val_loader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = []
        targets = []
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy().flatten()
            preds.extend(out)
            targets.extend(yb.numpy().flatten())
    return np.array(targets), np.array(preds)

def build_and_optimize(raw_data, property, ml_model):
    basename = os.path.splitext(os.path.basename(raw_data))[0]
    output_dir = os.path.join(os.getcwd(), ml_model)
    os.makedirs(output_dir, exist_ok=True)
    outer_fold, split = parse_filename(basename)

    df = pd.read_csv(raw_data)
    ids = df["Name"]
    drop_cols = ["Name", "Parent", "SMILES", "Mol", "Standard_SMILES", "Cluster_ID", "Scaffold_ID"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    # df.dropna(inplace=True)
    df.dropna(axis=0, how='any', inplace=True)

    Y = df[property].values
    X = df.drop(columns=[property]).values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    records = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold_idx + 1}/5 ===")
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        scaler = RobustScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        joblib.dump(scaler, os.path.join(output_dir, f"{basename}_fold{fold_idx+1}_scaler.save"))

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        for name, config in ARCHITECTURES.items():
            model = FCNN(input_dim=X_train.shape[1], output_dim=1,
                         layer_sizes=config["layers"], dropouts=config["dropouts"]).to(DEVICE)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0004)

            y_true, y_pred = train_model(model, train_loader, val_loader, criterion, optimizer)
            metrics = evaluate_model(y_true, y_pred)
            metrics.update({
                'architecture': name,
                'outer_fold': outer_fold,
                'inner_fold': fold_idx + 1,
                'split': split,
                'basename': basename
            })
            records.append(metrics)

            torch.save(model.state_dict(), os.path.join(output_dir, f'{basename}_fold{fold_idx+1}_{name}_{ml_model}.pt'))

    pd.DataFrame(records).to_csv(f"validation_{ml_model}.csv", index=False, mode='a', header=not os.path.exists(f"validation_{ml_model}.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to input CSV")
    parser.add_argument("property", help="Target property column name")
    args = parser.parse_args()

    model = "FCNN"
    print(f"Building {model} model using PyTorch...")
    build_and_optimize(args.input_csv, args.property, ml_model=model)
