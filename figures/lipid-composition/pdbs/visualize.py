import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Plot centered bilayer data")

    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of centered.out files (mass_XXXX_YYYY_dummy_centered.out)"
    )

    parser.add_argument(
        "--set",
        type=int,
        nargs="+",
        required=True,
        help="Set numbers to plot (1, 2, 3...)"
    )

    parser.add_argument(
        "--set_name",
        type=str,
        nargs="+",
        required=True,
        help="Label names corresponding to each set number"
    )


    parser.add_argument(
        "--outfile",
        type=str,
        default="plot.png",
        help="Output plot filename"
    )

    return parser.parse_args()


def extract_prefix(filename):
    """Extract XXXX from mass_XXXX_YYYY_dummy_centered.out"""
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) >= 3:
        return parts[1]   # XXXX
    return base


def load_centered_data(file, set_number):
    """
    file: path to centered CSV
    set_number: 1-based index (1, 2, 3)
    """
    df = pd.read_csv(file, sep="\t", header=0)

    if "Z_centered" not in df.columns:
        raise ValueError("File does not contain Z_centered column")

    z = df["Z_centered"].values

    set_col_name = f"Set{set_number}"
    if set_col_name not in df.columns:
        raise ValueError(f"File does not contain column {set_col_name}")

    set_values = df[set_col_name].values

    dist = np.abs(z)
    # unique_d, avg_values = average_by_distance(dist, set_values)
    unique_d, avg_values = bin_by_distance(dist, set_values, bin_width=3.0)

    return unique_d, avg_values


def average_by_distance(distances, values):
    """Average values for symmetric ±Z entries."""
    unique = sorted(list(set(distances)))
    avg_vals = []

    for d in unique:
        idx = np.where(distances == d)[0]
        avg_vals.append(np.mean(values[idx]))

    return np.array(unique), np.array(avg_vals)

def bin_by_distance(distances, values, bin_width=3.0):
    """
    Bin |Z| values into windows of `bin_width` Å and average the values in each bin.

    Returns:
        bin_centers: array of bin center positions
        bin_avgs: array of averaged values per bin
    """
    max_dist = distances.max()
    bins = np.arange(0, max_dist + bin_width, bin_width)

    bin_indices = np.digitize(distances, bins)
    bin_avgs = []
    bin_centers = []

    for i in range(1, len(bins)):
        idx = np.where(bin_indices == i)[0]
        if len(idx) > 0:
            bin_avgs.append(np.mean(values[idx]))
            bin_centers.append((bins[i-1] + bins[i]) / 2)

    return np.array(bin_centers), np.array(bin_avgs)


def main():
    args = parse_args()

    plt.figure(figsize=(4, 3))

    for file in args.files:
        prefix = extract_prefix(file)

        for set_idx, set_label in zip(args.set, args.set_name):
            # Load data
            try:
                x, y = load_centered_data(file, set_idx)
            except Exception as e:
                print(f"Skipping {file}: {e}")
                continue

            label = f"{prefix} - {set_label}"
            plt.plot(x, y, label=label)

    plt.xlabel("Distance from Bilayer Center (Å)", fontsize=12)
    plt.ylabel("Normalized Mass Density $(AMU/Å^3)$", fontsize=10)
    # plt.title("Mass Distribution of CHARMM36 Atom Types", fontsize=14)
    plt.ylim(0,0.002)
    plt.xlim(0,100)
    plt.legend()
    # plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=300)
    print(f"Saved: {args.outfile}")


if __name__ == "__main__":
    main()

