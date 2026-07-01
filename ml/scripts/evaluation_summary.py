import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def extract_system(filename):
    if 'pampa' in filename.lower():
        return 'pampa'
    elif 'popc' in filename.lower():
        return 'popc'
    else:
        return 'unknown'

def plot_metrics_histograms(df, system, split, output_dir):
    metrics = ["R2", "RMSE", "MAE", "PearsonR", "SpearmanRho"]
    sub_df = df[(df["system"] == system) & (df["split"] == split)]

    if sub_df.empty:
        print(f"Skipping empty group: {system}-{split}")
        return

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for idx, metric in enumerate(metrics):
        sns.histplot(sub_df[metric], bins=20, kde=False, ax=axes[idx], color='steelblue')
        axes[idx].set_title(metric, fontsize=14)
        axes[idx].set_xlabel(metric)
        axes[idx].set_ylabel("Count")
        axes[idx].set_xlim([-1,1])

    fig.suptitle(f"{system.upper()} - {split.capitalize()}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{system}_{split}_histograms.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot histograms for nested CV metrics.")
    parser.add_argument("--input_csv", required=True, help="Input CSV file with metrics")
    parser.add_argument("--output_dir", default="figures", help="Directory to save plots")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df['system'] = df['model_file'].apply(extract_system)

    splits = ["random", "butina", "scaffold"]
    systems = ["pampa", "popc"]

    for system in systems:
        for split in splits:
            plot_metrics_histograms(df, system, split, args.output_dir)

if __name__ == "__main__":
    main()

