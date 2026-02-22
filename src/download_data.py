"""
Download the Clothing Fit Dataset for Size Recommendation from Kaggle
and save raw files to Data/Raw/

Run from the project root:
    python download_data.py
"""

import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("rmisra/clothing-fit-dataset-for-size-recommendation")
print("Path to dataset files:", path)

# Create Data/Raw directory
os.makedirs("Data/Raw", exist_ok=True)

# Copy all files to Data/Raw
for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join("Data/Raw", file)
    shutil.copy(src, dst)
    print(f"Saved: {dst}")

print("\nDone. Files saved to Data/Raw/")