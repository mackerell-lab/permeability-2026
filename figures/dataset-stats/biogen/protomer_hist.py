import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import glob

# Read the file
df = pd.read_csv("../data/POPC-CHOL/biogen-ligands/lgfe/full_data/ligands_including_protomers.csv")

# Extract base name (everything before last "_")
df["BaseName"] = df["Ligand"].str.rsplit("_", n=1).str[0]

# Count protomers per molecule
protomer_counts = df.groupby("BaseName")["Ligand"].nunique()

count_freq = Counter(protomer_counts)

x = sorted(count_freq.keys())
y = [count_freq[k] for k in x]

plt.figure(figsize=(8,6))
bars = plt.bar(x, y, color="skyblue", edgecolor="black")

plt.yscale("log")
plt.ylim(1, None)

for bar, count in zip(bars, y):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        count * 1.05,   # position slightly above bar
        str(count),
        ha='center',
        va='bottom',
        fontsize=14
    )

plt.xlabel("Number of Protomers per Molecule", fontsize=18)
plt.ylabel("Count of Molecules (Log Scale)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.title("Protomer Count Distribution")
plt.xticks(range(1, protomer_counts.max() + 1))
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.savefig("protomer_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
