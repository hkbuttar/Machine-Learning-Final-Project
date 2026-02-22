"""
Download all datasets for the Clothing Fit Prediction project.

Run from project root:
    python download_data.py

Datasets:
  1. Clothing Fit Dataset (ModCloth + RentTheRunway)
  2. Clothes-Size-Prediction (baseline benchmark)
"""

import kagglehub
import shutil
import os

os.makedirs("Data/Raw", exist_ok=True)

# ── 1. Clothing Fit Dataset (ModCloth + RentTheRunway) ──────────────────────
print("Downloading: Clothing Fit Dataset...")
path1 = kagglehub.dataset_download("rmisra/clothing-fit-dataset-for-size-recommendation")
print(f"  Source: {path1}")
for file in os.listdir(path1):
    shutil.copy(os.path.join(path1, file), os.path.join("Data/Raw", file))
    print(f"  Saved: Data/Raw/{file}")

# ── 2. Clothes-Size-Prediction ──────────────────────────────────────────────
print("\nDownloading: Clothes-Size-Prediction...")
path2 = kagglehub.dataset_download("tourist55/clothessizeprediction")
print(f"  Source: {path2}")
for file in os.listdir(path2):
    shutil.copy(os.path.join(path2, file), os.path.join("Data/Raw", file))
    print(f"  Saved: Data/Raw/{file}")

print("\nDone. All files saved to Data/Raw/")
print("Contents:")
for f in sorted(os.listdir("Data/Raw")):
    size = os.path.getsize(f"Data/Raw/{f}") / (1024 * 1024)
    print(f"  {f:<45} {size:.1f} MB")
