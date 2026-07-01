import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to plot scatter plots with histograms on top and right
def plot_scatter_with_histograms(
    df, x_column, y_column, xlabel, ylabel, filename):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1, 6, 0], width_ratios=[6, 1, 0], hspace=0.05, wspace=0.05
    )

    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1])

    x_data = df[x_column]
    y_data = df[y_column]

    # Histogram on top
    # ax_histx.hist(x_data.dropna(), bins=20, color="gray", alpha=0.7)
    ax_histx.hist(x_data, bins=50, color="gray", alpha=0.7)
    ax_histx.set_ylabel("Frequency", fontsize=18)
    ax_histx.set_xticks([])
    ax_histx.set_xlim(x_data.min(), x_data.max())

    # Scatter plot
    ax_scatter.scatter(
        x_data,
        y_data,
        alpha=0.4,
    )
    ax_scatter.set_ylabel(ylabel, fontsize=18)
    ax_histy.hist(
        # np.log10(y_data.dropna()),
        y_data,
        bins=50,
        color="gray",
        alpha=0.7,
        orientation="horizontal",
    )
    ax_histy.set_ylim(y_data.min(), y_data.max())
    ax_scatter.set_ylim(y_data.min(), y_data.max())

    ax_scatter.set_xlim(x_data.min(), x_data.max())
    ax_scatter.set_xlabel(xlabel, fontsize=18)
    ax_histy.set_xlabel("Frequency", fontsize=18)
    ax_histy.set_yticks([])

    ax_scatter.tick_params(axis="both", which="major", labelsize=14)
    ax_histx.tick_params(axis="y", which="major", labelsize=14)
    ax_histy.tick_params(axis="x", which="major", labelsize=14)

    # plt.tight_layout(pad=0.5)
    plt.savefig(filename)
    plt.close()

# Load datasets
df = pd.read_csv("ADME_MDR1_ER_PAAP-cm_s.csv")

# Generate scatter plots with histograms
plot_scatter_with_histograms(
    df,
    "LOG10 MDR1-MDCK Papp_A2B (cm/s)",
    "LOG 10 ER Calc",
    "log10(P_app)",
    "Efflux Ratio",
    "stats.png",
)
