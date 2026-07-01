#!/usr/bin/env python3
import pandas as pd
import argparse

def load_density_file(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Skip lines that don't start with a float
            try:
                float(parts[0])
            except ValueError:
                continue

            # Try to convert as many numeric columns as possible
            numeric = []
            for p in parts:
                try:
                    numeric.append(float(p))
                except ValueError:
                    break  # stop at first non-numeric column

            # Need at least Z + AllAtoms
            if len(numeric) < 2:
                continue

            # Ensure at least 5 columns by padding missing ones
            while len(numeric) < 5:
                numeric.append(0.0)

            # Use only the first 5
            z, a, s1, s2, s3 = numeric[:5]
            rows.append([z, a, s1, s2, s3])

    return pd.DataFrame(rows, columns=["Z", "AllAtoms", "Set1", "Set2", "Set3"])

def extract_centered_window(df, z_center, N):
    # Find index whose Z is closest to z_center
    idx = (df["Z"] - z_center).abs().idxmin()

    # Window limits
    start = max(idx - N, 0)
    end = min(idx + N, len(df) - 1)

    # Extract window and center Z
    window = df.iloc[start:end+1].copy()
    window["Z_centered"] = window["Z"] - z_center

    return window, start, end, idx


def main():
    parser = argparse.ArgumentParser(description="Center Z-axis and extract ±N rows around z_center.")
    parser.add_argument("--input", required=True, help="Input density file")
    parser.add_argument("--z_center", type=float, required=True, help="Z value to center on")
    parser.add_argument("--N", type=int, required=True, help="Number of rows above and below center")
    parser.add_argument("--output", default="centered_output.dat", help="Output file name")

    args = parser.parse_args()

    df = load_density_file(args.input)
    window, start, end, idx = extract_centered_window(df, args.z_center, args.N)

    # Save output
    window.to_csv(args.output, sep="\t", index=False)

    # Print useful info
    print("\n=== Selected Window Information ===")
    print(f"Closest Z to z_center ({args.z_center}) is at index {idx}: Z = {df.loc[idx, 'Z']:.4f}")
    print(f"Extracting rows from index {start} to {end}")
    print(f"Total rows extracted: {len(window)}")
    print(f"Output written to: {args.output}\n")


if __name__ == "__main__":
    main()

