import pandas as pd
from rdkit import Chem
from pathlib import Path

# -----------------------------
# Helpers
# -----------------------------

def normalize_parent(name: str) -> str:
    """Normalize Parent names by stripping whitespace and lowercasing."""
    if pd.isna(name):
        return None
    return name.strip().lower()


def smiles_to_inchikey(smiles: str) -> str:
    """Convert SMILES to InChIKey using RDKit."""
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.InchiToInchiKey(inchi)
        return inchikey
    except:
        return None


# -----------------------------
# Main processing function
# -----------------------------

def process_csv_files(csv_paths):
    """
    Load N CSV files, normalize Parent, convert SMILES → InChIKey,
    and report unique/duplicate Parents and SMILES.
    """
    all_dfs = []

    for path in csv_paths:
        df = pd.read_csv(path)
        df["Parent_norm"] = df["Parent"].apply(normalize_parent)
        df["Molecule_norm"] = df["Molecule"].apply(normalize_parent)
        df["InChIKey"] = df["SMILES"].apply(smiles_to_inchikey)
        df["source_file"] = Path(path).name
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # -----------------------------
    # Unique and duplicate Parents
    # -----------------------------
    unique_parents = df_all["Parent_norm"].dropna().unique()
    duplicate_parents = (
        df_all[df_all.duplicated("Parent_norm", keep=False)]
        .sort_values("Parent_norm")
    )

    unique_ligands = df_all["Molecule_norm"].dropna().unique()
    duplicate_ligands = (
        df_all[df_all.duplicated("Molecule_norm", keep=False)]
        .sort_values("Molecule_norm")
    )

    # -----------------------------
    # Unique and duplicate molecules (by InChIKey)
    # -----------------------------
    unique_molecules = df_all.drop_duplicates("InChIKey")
    duplicate_molecules = (
        df_all[df_all.duplicated("InChIKey", keep=False)]
        .sort_values("InChIKey")
    )

    return {
        "all_data": df_all,
        "unique_parents": unique_parents,
        "duplicate_parents": duplicate_parents,
        "unique_ligands": unique_ligands,
        "duplicate_ligands": duplicate_ligands,
        "unique_molecules": unique_molecules,
        "duplicate_molecules": duplicate_molecules,
    }


if __name__ == "__main__":
    # list of CSV files
    csv_files = [
        "../data-processed/popc_chol-non-biogen/P_app/explicit_protomers/popc-chol_P_app_including_protomers_lgfe-cMdr1_KO_MDCK1-beran.csv",
        "../data-processed/popc_chol-non-biogen/P_app/explicit_protomers/popc-chol_P_app_including_protomers_lgfe-MDCK-Hellinger.csv",
        "../data-processed/popc_chol-non-biogen/P_app/explicit_protomers/popc-chol_P_app_including_protomers_lgfe-MDCK-Irvine.csv",
        "../data-processed/popc_chol-non-biogen/P_app/explicit_protomers/popc-chol_P_app_including_protomers_lgfe-MDCK-MDR1-Hellinger.csv",
        "../data-processed/popc_chol-non-biogen/P_app/explicit_protomers/popc-chol_P_app_including_protomers_lgfe-MDCK-MDR1-QWang.csv",
        "../data-processed/popc_chol-non-biogen/P_app/explicit_protomers/popc-chol_P_app_including_protomers_lgfe-MDCK-MDR1-Summerfield.csv"
    ]

    results = process_csv_files(csv_files)

    print("\n=== UNIQUE PARENTS ===")
    print(f"Number of unique parents: {len(results['unique_parents'])}")
    print(results["unique_parents"])

    # print("\n=== DUPLICATE PARENTS ===")
    # print(results["duplicate_parents"][["Parent", "Parent_norm", "source_file"]])

    print("\n=== UNIQUE Molecules (By Molecule Name) ===")
    print(f"Number of unique ligands: {len(results['unique_ligands'])}")
    print(results["unique_ligands"])

    
    print("\n=== UNIQUE MOLECULES (by InChIKey) ===")
    print(f"Number of unique molecules: {len(results['unique_molecules'])}")
    print(results["unique_molecules"][["Parent", "SMILES", "InChIKey"]])

    # print("\n=== DUPLICATE MOLECULES (by InChIKey) ===")
    # print(results["duplicate_molecules"][["Parent", "SMILES", "InChIKey", "source_file"]])