import os
import argparse
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from collections import defaultdict
from rdkit import RDLogger

# Optional: silence RDKit warnings
RDLogger.DisableLog("rdApp.*")

# === Standardization Functions ===
uncharger = rdMolStandardize.Uncharger()
tautomer_enum = rdMolStandardize.TautomerEnumerator()

def standardize_molecule(mol):
    try:
        mol = rdMolStandardize.Cleanup(mol)          
        mol = rdMolStandardize.FragmentParent(mol)
        mol = uncharger.uncharge(mol)  
        mol = tautomer_enum.Canonicalize(mol)    # Tautomer canonicalization
        Chem.SanitizeMol(mol)                    # Ensure chemical validity
        return mol
    except Exception as e:
        return None  # Skip mol if any step fails

def smiles_to_standardized_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return standardize_molecule(mol)
    return None

# === Load and Standardize from CSV ===
def load_standardized_molecules(csv_path, smiles_column="SMILES"):
    df = pd.read_csv(csv_path)
    df["Mol"] = df[smiles_column].apply(smiles_to_standardized_mol)
    df = df[df["Mol"].notnull()].reset_index(drop=True)
    df["Standard_SMILES"] = df["Mol"].apply(Chem.MolToSmiles)
    return df

# === Random split ===
def random_split(df, train_frac=0.8, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(df)) # Shuffling
    cutoff = int(train_frac * len(df))
    train_idx, test_idx = indices[:cutoff], indices[cutoff:]
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)

# === Butina clustering split ===
def butina_split(df, cutoff=0.35, train_frac=0.8, seed=42):

    # Generate Morgan fingerprints
    mols = list(df["Mol"])
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    
    # Computing pairwise distances for Butina
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    
    #  Clustering
    clusters = list(Butina.ClusterData(dists, len(fps), cutoff, True))
    
    # Assign cluster ID
    cluster_id_map = {}
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_id_map[idx] = i
    df["Cluster_ID"] = df.index.map(cluster_id_map)
    
    np.random.seed(seed) 
    np.random.shuffle(clusters)  # shuffle clusters instead of sorting by size
    # clusters = sorted(clusters, key=lambda x: -len(x))  # optional: sort by size

    # Assign clusters to train/test
    train_rows, test_rows = [], []
    train_count = 0
    total_count = len(df)

    for cluster in clusters:
        cluster_df = df.iloc[list(cluster)]
        if (train_count + len(cluster_df)) / total_count <= train_frac:
            train_rows.append(cluster_df)
            train_count += len(cluster_df)
        else:
            test_rows.append(cluster_df)

    # Concatenate and reset indices
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    return train_df, test_df

# === Scaffold clustering split ===
def scaffold_split(df, train_frac=0.8, seed=42):

    # Group molecules by scaffold
    scaff_dict = defaultdict(list)
    for idx, mol in enumerate(df["Mol"]):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaff_dict[scaffold].append(idx)
    
    scaffold_groups = list(scaff_dict.values())
    # Tag each molecule with scaffold ID
    scaffold_id_map = {}
    for i, group in enumerate(scaffold_groups):
        for idx in group:
            scaffold_id_map[idx] = i
    df["Scaffold_ID"] = df.index.map(scaffold_id_map)

    np.random.seed(seed)
    np.random.shuffle(scaffold_groups)
    # Sort scaffold groups by size descending
    # scaffold_groups = sorted(scaff_dict.values(), key=lambda x: -len(x))

    train_idx, test_idx = [], []

    # Assign clusters to train/test
    for group in scaffold_groups:
        if len(train_idx) / len(df) <= train_frac:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    
    return train_df, test_df

def plot_group_sizes(train_df, test_df, group_col, title_prefix, file_prefix):
    # Count group sizes
    train_sizes = train_df[group_col].value_counts()
    test_sizes = test_df[group_col].value_counts()

    # Select top 100 groups (by size)
    top_n = 100
    train_top = train_sizes.sort_values(ascending=False).head(top_n).values
    test_top = test_sizes.sort_values(ascending=False).head(top_n).values

    # Pad shorter array if needed
    min_len = min(len(train_top), len(test_top))
    train_top = train_top[:min_len]
    test_top = test_top[:min_len]

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    # --- Left: Scatter + line plot for top 100 clusters ---
    axs[0].plot(range(min_len), train_top, marker='o', label='Train', linestyle='-', color='blue')
    axs[0].plot(range(min_len), test_top, marker='o', label='Test', linestyle='-', color='orange')
    axs[0].set_title(f"{title_prefix} Top {top_n} Group Sizes")
    axs[0].set_xlabel("Group Rank")
    axs[0].set_ylabel("Number of Molecules")
    axs[0].legend()
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # --- Right: Histogram of group size distribution ---
    bins = range(1, max(train_sizes.max(), test_sizes.max()) + 2)
    axs[1].hist(train_sizes.values, bins=bins, color='blue', alpha=0.6, label='Train')
    axs[1].hist(test_sizes.values, bins=bins, color='orange', alpha=0.6, label='Test')
    axs[1].set_title(f"{title_prefix} Group Size Distribution")
    axs[1].set_xlabel("Group Size (Number of Molecules)")
    axs[1].set_ylabel("Frequency")
    axs[1].legend()
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(f"{file_prefix}_{title_prefix}_group_sizes.png")
    plt.close()


def calculate_train_test_similarity(train_df, test_df, mol_col="Mol", fp_radius=2, fp_nbits=2048):
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, fp_radius, nBits=fp_nbits) for m in train_df[mol_col]]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(m, fp_radius, nBits=fp_nbits) for m in test_df[mol_col]]

    similarities = []
    for tfp in train_fps:
        sims = DataStructs.BulkTanimotoSimilarity(tfp, test_fps)
        similarities.extend(sims)

    return similarities


def plot_similarity_distribution_replicates(similarity_replicates, file_prefix, title="", hist=True):
    """
    Plot multiple overlaid histograms of similarity values for each replicate
    in a single figure for the given split type.
    
    Parameters:
        similarity_replicates (list of lists): Outer list is per replicate, inner list is similarity values.
        file_prefix (str): Prefix for output file.
        title (str): Title or split type (e.g., 'Random').
    """
    num_replicates = len(similarity_replicates)
    cmap = colormaps['tab10']

    plt.figure(figsize=(10, 6))

    if hist:

        # Fixed bin range for Tanimoto similarities
        bins = np.linspace(0, 1, 50)

        for i, sims in enumerate(similarity_replicates):
            plt.hist(
                sims,
                bins=bins,
                alpha=0.5,
                label=f"Replicate {i+1}",
                color=cmap(i % 10),
                edgecolor='black',
                linewidth=0.5,
                density=False
            )

        plt.title(f"{title} Split: Tanimoto Similarity Distribution Across Replicates")
        plt.xlabel("Tanimoto Similarity")
        plt.ylabel("Frequency")

    else:
        for i, sims in enumerate(similarity_replicates):
            color = cmap(i % 10)
            sns.kdeplot(
                sims,
                bw_adjust=1.2,
                label=f"Replicate {i+1}",
                color=color,
                linewidth=2
            )

        plt.title(f"{title} Split: Tanimoto Similarity Distribution (KDE) Across Replicates")
        plt.xlabel("Tanimoto Similarity")
        plt.ylabel("Density")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_{title}_replicate_similarity_hist.png")
    plt.close()


def plot_similarity_violinplot(similarity_dict, file_prefix):
    # Convert similarity_dict to a tidy DataFrame
    data = [(split, val) for split, sims in similarity_dict.items() for val in sims]
    df = pd.DataFrame(data, columns=["Split", "Similarity"])

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="Split", y="Similarity", hue="Split", palette="pastel", inner="quartile", legend=False)
    plt.legend().remove()
    plt.title("Tanimoto Similarity Distributions Across Splits")
    plt.xlabel("Split Type")
    plt.ylabel("Tanimoto Similarity")
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_similarity_violinplot.png")
    plt.close()

def generate_random_list(n, seed=42):
    random.seed(seed)
    return [random.randint(0, 2**31 - 1) for _ in range(n)]

# === Example usage ===
if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Split data randomly, butina clustering and by scaffold grouping")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("replicates_num", help="Number of test-train replicates (Outer CV fold)")
    parser.add_argument("--data_stats", default=True, help="Plot data statistics")

    # Parse arguments
    args = parser.parse_args()

    basename = os.path.splitext(os.path.basename(args.input_csv))[0]

    df = load_standardized_molecules(args.input_csv, smiles_column="SMILES")
    replicates_seed_list = generate_random_list(int(args.replicates_num))

    # Dict to store stats
    if args.data_stats:
        similarity_dict = {
            "Random": [],
            "Butina": [],
            "Scaffold": []
        }

    # Creating N replicates
    for num in range(len(replicates_seed_list)):

        replicate_seed = replicates_seed_list[num]
        file_prefix_replicate = basename + "_" + str(num+1)
        # Split the dataset
        train_rand, test_rand = random_split(df, seed=replicate_seed)
        train_butina, test_butina = butina_split(df, cutoff=0.35, seed=replicate_seed)
        train_scaff, test_scaff = scaffold_split(df, seed=replicate_seed)

        # Save the datasets
        train_rand.to_csv(f"{basename}_{num+1}_train_random.csv", index=False)
        test_rand.to_csv(f"{basename}_{num+1}_test_random.csv", index=False)

        train_butina.to_csv(f"{basename}_{num+1}_train_butina.csv", index=False)
        test_butina.to_csv(f"{basename}_{num+1}_test_butina.csv", index=False)

        train_scaff.to_csv(f"{basename}_{num+1}_train_scaffold.csv", index=False)
        test_scaff.to_csv(f"{basename}_{num+1}_test_scaffold.csv", index=False)


        if args.data_stats:

            sims_rand = calculate_train_test_similarity(train_rand, test_rand)
            sims_butina = calculate_train_test_similarity(train_butina, test_butina)
            sims_scaff = calculate_train_test_similarity(train_scaff, test_scaff)

            # Accumulate similarity scores
            similarity_dict["Random"].append(sims_rand)
            similarity_dict["Butina"].append(sims_butina) 
            similarity_dict["Scaffold"].append(sims_scaff)

            # Plotting
            plot_group_sizes(train_butina, test_butina, group_col="Cluster_ID", title_prefix="Butina", file_prefix=file_prefix_replicate)
            plot_group_sizes(train_scaff, test_scaff, group_col="Scaffold_ID", title_prefix="Scaffold", file_prefix=file_prefix_replicate)

            sims_all = {
                "Random": sims_rand,
                "Butina": sims_butina,
                "Scaffold": sims_scaff
                }

            plot_similarity_violinplot(sims_all, file_prefix=file_prefix_replicate)

    if args.data_stats:

        for split_type, replicate_list in similarity_dict.items():
            plot_similarity_distribution_replicates(replicate_list, file_prefix=basename, title=split_type, hist=False)
