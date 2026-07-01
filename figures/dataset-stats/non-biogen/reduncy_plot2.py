import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
input_csv = "membrane-permeability-revised-dataset-in-vitro.csv" 
molecule_col = "Molecule"
papp_col = "P_app"
reference_col = "reference"
assay_col = "Assay"

# Assays to exclude
exclude_assays = ["bovine BMEC", "rat BCEC", "human BMEC", "Caco-2", "VB-Caco-2"]

font_size_title = 30
font_size_labels = 28
font_size_ticks = 26
font_size_legend = 26

# === LOAD DATA ===
df = pd.read_csv(input_csv)

# Exclude specified assays
df = df[~df[assay_col].isin(exclude_assays)]

# Find molecules with duplicates
duplicate_mols = df["Molecule"].value_counts()
duplicate_mols = duplicate_mols[duplicate_mols > 1].index
df = df[df["Molecule"].isin(duplicate_mols)]

if df.empty:
    print("No duplicate molecules found after filtering.")
else:
    # === Prepare color and marker maps ===
    assays = sorted(df["Assay"].unique())
    refs = sorted(df["reference"].unique())

    color_map = {assay: plt.cm.tab10(i % 10) for i, assay in enumerate(assays)}
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
    marker_map = {ref: markers[i % len(markers)] for i, ref in enumerate(refs)}

    # Assign x positions evenly spaced per molecule
    unique_mols = sorted(df["Molecule"].unique())
    x_positions = {mol: i for i, mol in enumerate(unique_mols)}

    # === Plot setup ===
    plt.figure(figsize=(max(16, len(unique_mols) * 0.5), 10))

    for _, row in df.iterrows():
        mol = row["Molecule"]
        assay = row["Assay"]
        ref = row["reference"]
        y = np.log10(row["P_app"])

        plt.scatter(
            x_positions[mol],
            y,
            color=color_map[assay],
            marker=marker_map[ref],
            s=300,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
            label=f"{assay}_{ref}"
        )

    # === Format axes and labels ===
    plt.xticks(
        ticks=list(x_positions.values()),
        labels=list(x_positions.keys()),
        rotation=90,
        ha="center",
        va="top",
        fontsize=font_size_ticks
    )
    plt.tick_params(axis='x', pad=8)

    plt.yticks(fontsize=24)
    # plt.xlabel("Molecule", fontsize=font_size_labels)
    plt.ylabel("log10(P_app)", fontsize=font_size_labels)
    # plt.title("P_app Comparison Across Assays")

    plt.tight_layout()

    # === SAVE MAIN PLOT ===
    base_name = os.path.splitext(os.path.basename(input_csv))[0]

    png_name = f"{base_name}_duplicate_logPapp_filtered_color_shape.png"

    plt.savefig(png_name, dpi=300, bbox_inches="tight")

    # ==========================================================
    # === CREATE SEPARATE LEGEND FIGURE ===
    # ==========================================================

    handles, labels = plt.gca().get_legend_handles_labels()

    unique = dict(zip(labels, handles))

    fig_legend = plt.figure(figsize=(8, max(4, len(unique) * 0.3)))

    fig_legend.legend(
        unique.values(),
        unique.keys(),
        loc="center",
        fontsize=font_size_legend,
        frameon=False,
        ncol=3
    )

    plt.axis("off")

    legend_name = f"{base_name}_legend.png"

    fig_legend.savefig(
        legend_name,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig_legend)

    print(f"Saved main plot: {png_name}")
    print(f"Saved legend plot: {legend_name}")