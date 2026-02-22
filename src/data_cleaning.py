"""
Data Cleaning Pipeline for Clothing Fit Prediction Project
===========================================================

Three datasets:
  1. RentTheRunway (PRIMARY)   — 192K transactions, fit feedback + body measurements + reviews
  2. ModCloth (SECONDARY)      — 83K transactions, cross-retailer generalization testing
  3. Clothes-Size-Prediction   — 119K rows, clean baseline for classifier benchmarking

Run from project root:
    python src/data_cleaning.py

Inputs:
    Data/Raw/renttherunway_final_data.json
    Data/Raw/modcloth_final_data.json
    Data/Raw/final_test.csv              (download from Kaggle if missing)

Outputs:
    Data/Processed/renttherunway_clean.csv
    Data/Processed/modcloth_clean.csv
    Data/Processed/clothes_size_clean.csv
"""

import pandas as pd
import numpy as np
import re
import os
import sys


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_height_rtw(h):
    """Convert 5' 8" → 68.0 inches."""
    if pd.isna(h):
        return np.nan
    m = re.match(r"(\d+)'\s*(\d+)", str(h).strip())
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    return np.nan


def parse_height_modcloth(h):
    """Convert '5ft 6in' → 66.0 inches."""
    if pd.isna(h):
        return np.nan
    m = re.match(r"(\d+)ft\s*(\d+)in", str(h).strip())
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    return np.nan


def parse_weight(w):
    """Convert '137lbs' → 137.0."""
    if pd.isna(w):
        return np.nan
    m = re.search(r"(\d+)", str(w).strip())
    if m:
        return float(m.group(1))
    return np.nan


def parse_bust_size(b):
    """Split '34d' → (34.0, 'd')."""
    if pd.isna(b):
        return np.nan, np.nan
    m = re.match(r"(\d+)\s*([a-zA-Z]+)", str(b).strip().lower())
    if m:
        return float(m.group(1)), m.group(2)
    return np.nan, np.nan


CUP_MAP = {
    'aa': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4,
    'dd': 5, 'ddd': 6, 'e': 5, 'f': 6, 'g': 7,
    'h': 8, 'i': 9, 'j': 10
}

def cup_to_numeric(cup):
    """Map cup letter to ordinal: a=1, b=2, d=4, dd=5, etc."""
    if pd.isna(cup):
        return np.nan
    return CUP_MAP.get(str(cup).strip().lower(), np.nan)


FIT_MAP = {'small': 0, 'fit': 1, 'large': 2}


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_summary(df, name):
    """Print cleaning summary stats."""
    print(f"\n  Cleaned shape: {df.shape}")

    # Find the fit column
    fit_col = 'fit' if 'fit' in df.columns else 'fit_equiv' if 'fit_equiv' in df.columns else 'size'
    print(f"\n  {fit_col} distribution:")
    for val, count in df[fit_col].value_counts().items():
        print(f"    {str(val):<8} {count:>7,} ({count/len(df)*100:.1f}%)")

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        print(f"\n  Remaining nulls:")
        for col, n in nulls.items():
            print(f"    {col:<25} {n:>7,} ({n/len(df)*100:.1f}%)")
    else:
        print(f"\n  No remaining nulls.")

    print(f"\n  Columns ({len(df.columns)}):")
    for i in range(0, len(df.columns), 4):
        cols = df.columns[i:i+4]
        print(f"    {', '.join(cols)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RENTTHERUNWAY (PRIMARY)
# ═══════════════════════════════════════════════════════════════════════════════

def clean_renttherunway(path):
    """
    Clean RentTheRunway dataset.

    Raw: 192,544 rows × 15 columns
    Key nulls: weight (15.6%), bust size (9.6%), body type (7.6%), age (0.5%)
    """
    print_section("1. RentTheRunway (PRIMARY)")
    df = pd.read_json(path, lines=True)
    print(f"  Raw shape: {df.shape}")

    # ── Parse measurements ──────────────────────────────────────────────
    # Height: "5' 8" → 68.0 inches
    df['height_inches'] = df['height'].apply(parse_height_rtw)

    # Weight: "137lbs" → 137.0
    df['weight_lbs'] = df['weight'].apply(parse_weight)

    # Bust size: "34d" → bust_band=34, cup_size_num=4
    bust_parsed = df['bust size'].apply(lambda x: pd.Series(parse_bust_size(x)))
    df['bust_band'] = bust_parsed[0]
    df['cup_size_letter'] = bust_parsed[1]
    df['cup_size_num'] = df['cup_size_letter'].apply(cup_to_numeric)

    # ── Parse review date ───────────────────────────────────────────────
    df['review_date'] = pd.to_datetime(df['review_date'], format='%B %d, %Y', errors='coerce')
    df['review_year'] = df['review_date'].dt.year
    df['review_month'] = df['review_date'].dt.month

    # ── Encode target ───────────────────────────────────────────────────
    df['fit_label'] = df['fit'].map(FIT_MAP)

    # ── Encode categoricals ─────────────────────────────────────────────
    # Body type → numeric
    body_types = sorted(df['body type'].dropna().unique())
    body_map = {bt: i for i, bt in enumerate(body_types)}
    df['body_type_num'] = df['body type'].map(body_map)
    print(f"  Body types found: {body_map}")

    # Rented for → numeric
    rental_types = sorted(df['rented for'].dropna().unique())
    rental_map = {rt: i for i, rt in enumerate(rental_types)}
    df['rented_for_num'] = df['rented for'].map(rental_map)
    print(f"  Rental occasions: {rental_map}")

    # ── BMI (where weight + height available) ───────────────────────────
    df['bmi'] = (df['weight_lbs'] * 703) / (df['height_inches'] ** 2)

    # ── Drop original unparsed columns ──────────────────────────────────
    drop_cols = ['height', 'weight', 'bust size']
    df.drop(columns=drop_cols, inplace=True)

    # ── Clean column names ──────────────────────────────────────────────
    df.columns = df.columns.str.replace(' ', '_').str.lower()

    # ── Drop rows with missing target ───────────────────────────────────
    before = len(df)
    df.dropna(subset=['fit_label'], inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing fit label")

    # ── Reorder columns for readability ─────────────────────────────────
    id_cols = ['user_id', 'item_id']
    target_cols = ['fit', 'fit_label']
    measurement_cols = ['height_inches', 'weight_lbs', 'bmi', 'bust_band',
                        'cup_size_letter', 'cup_size_num', 'size', 'age']
    cat_cols = ['category', 'body_type', 'body_type_num', 'rented_for', 'rented_for_num']
    review_cols = ['rating', 'review_summary', 'review_text', 'review_date',
                   'review_year', 'review_month']
    remaining = [c for c in df.columns if c not in
                 id_cols + target_cols + measurement_cols + cat_cols + review_cols]
    col_order = id_cols + target_cols + measurement_cols + cat_cols + review_cols + remaining
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    print_summary(df, 'RentTheRunway')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODCLOTH (SECONDARY)
# ═══════════════════════════════════════════════════════════════════════════════

def clean_modcloth(path):
    """
    Clean ModCloth dataset.

    Raw: 82,790 rows × 18 columns
    Key nulls: waist (96.5%), bust (85.7%), shoe_size (66.3%), shoe_width (77.5%)
    Strategy: Drop columns >80% null. Keep what's usable.
    """
    print_section("2. ModCloth (SECONDARY)")
    df = pd.read_json(path, lines=True)
    print(f"  Raw shape: {df.shape}")

    # ── Drop columns >80% null ──────────────────────────────────────────
    null_pcts = df.isnull().mean()
    high_null = null_pcts[null_pcts > 0.80].index.tolist()
    print(f"  Dropping >80% null columns: {high_null}")
    df.drop(columns=high_null, inplace=True)

    # ── Parse height ────────────────────────────────────────────────────
    df['height_inches'] = df['height'].apply(parse_height_modcloth)
    df.drop(columns=['height'], inplace=True)

    # ── Cup size → numeric ──────────────────────────────────────────────
    df['cup_size_num'] = df['cup size'].apply(cup_to_numeric)
    df.drop(columns=['cup size'], inplace=True)

    # ── Encode target ───────────────────────────────────────────────────
    df['fit_label'] = df['fit'].map(FIT_MAP)

    # ── Encode length ───────────────────────────────────────────────────
    length_map = {'slightly short': -1, 'just right': 0, 'slightly long': 1}
    df['length_num'] = df['length'].map(length_map)

    # ── Clean column names ──────────────────────────────────────────────
    df.columns = df.columns.str.replace(' ', '_').str.lower()

    # ── Drop rows with missing target ───────────────────────────────────
    before = len(df)
    df.dropna(subset=['fit_label'], inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing fit label")

    # ── Reorder columns ────────────────────────────────────────────────
    id_cols = ['user_id', 'item_id', 'user_name']
    target_cols = ['fit', 'fit_label']
    measurement_cols = ['height_inches', 'hips', 'bra_size', 'cup_size_num', 'size']
    cat_cols = ['category', 'length', 'length_num']
    review_cols = ['quality', 'review_summary', 'review_text']
    remaining = [c for c in df.columns if c not in
                 id_cols + target_cols + measurement_cols + cat_cols + review_cols]
    col_order = id_cols + target_cols + measurement_cols + cat_cols + review_cols + remaining
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    print_summary(df, 'ModCloth')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLOTHES-SIZE-PREDICTION (SUPPLEMENT)
# ═══════════════════════════════════════════════════════════════════════════════

def clean_clothes_size(path):
    """
    Clean Clothes-Size-Prediction dataset.

    Raw: 119,218 rows × 4 columns (weight, age, height, size)
    Very clean — minimal processing needed.
    Used as baseline benchmark for Phase 2 classifiers.
    """
    print_section("3. Clothes-Size-Prediction (SUPPLEMENT)")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # ── Drop nulls (< 1%) ──────────────────────────────────────────────
    before = len(df)
    df.dropna(inplace=True)
    print(f"  Dropped {before - len(df)} null rows ({(before-len(df))/before*100:.2f}%)")

    # ── Filter unrealistic values ───────────────────────────────────────
    before = len(df)
    df = df[(df['age'] >= 18) & (df['age'] <= 80)]
    df = df[(df['weight'] >= 35) & (df['weight'] <= 150)]  # kg
    df = df[(df['height'] >= 140) & (df['height'] <= 210)]  # cm
    print(f"  Filtered {before - len(df)} unrealistic rows (age/weight/height outliers)")

    # ── Encode size target ──────────────────────────────────────────────
    size_order = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']
    size_map = {s: i for i, s in enumerate(size_order)}
    df['size_label'] = df['size'].map(size_map)

    # ── Map to fit-equivalent labels for comparability ──────────────────
    # XXS, XS, S → "small", M → "fit", L, XL, XXL, XXXL → "large"
    fit_equiv = {
        'XXS': 'small', 'XS': 'small', 'S': 'small',
        'M': 'fit',
        'L': 'large', 'XL': 'large', 'XXL': 'large', 'XXXL': 'large'
    }
    df['fit_equiv'] = df['size'].map(fit_equiv)
    df['fit_equiv_label'] = df['fit_equiv'].map(FIT_MAP)

    # ── Convert height to inches for cross-dataset comparability ────────
    df['height_inches'] = df['height'] / 2.54
    # Convert weight to lbs
    df['weight_lbs'] = df['weight'] * 2.205
    # BMI
    df['bmi'] = (df['weight_lbs'] * 703) / (df['height_inches'] ** 2)

    # ── Rename for clarity ──────────────────────────────────────────────
    df.rename(columns={
        'weight': 'weight_kg',
        'height': 'height_cm'
    }, inplace=True)

    print_summary(df, 'Clothes-Size-Prediction')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("Data/Processed", exist_ok=True)

    print("\n" + "█" * 70)
    print("  CLOTHING FIT PREDICTION — DATA CLEANING PIPELINE")
    print("█" * 70)

    # ── 1. RentTheRunway ────────────────────────────────────────────────
    rtw_path = "Data/Raw/renttherunway_final_data.json"
    if os.path.exists(rtw_path):
        rtw = clean_renttherunway(rtw_path)
        rtw.to_csv("Data/Processed/renttherunway_clean.csv", index=False)
        print(f"\n  ✓ Saved: Data/Processed/renttherunway_clean.csv")
    else:
        print(f"\n  ✗ Not found: {rtw_path}")

    # ── 2. ModCloth ─────────────────────────────────────────────────────
    mc_path = "Data/Raw/modcloth_final_data.json"
    if os.path.exists(mc_path):
        mc = clean_modcloth(mc_path)
        mc.to_csv("Data/Processed/modcloth_clean.csv", index=False)
        print(f"\n  ✓ Saved: Data/Processed/modcloth_clean.csv")
    else:
        print(f"\n  ✗ Not found: {mc_path}")

    # ── 3. Clothes-Size-Prediction ──────────────────────────────────────
    # Try common file names from Kaggle download
    cs_candidates = [
        "Data/Raw/final_test.csv",
        "Data/Raw/clothes_size.csv",
    ]
    cs_path = None
    for candidate in cs_candidates:
        if os.path.exists(candidate):
            cs_path = candidate
            break

    if cs_path:
        cs = clean_clothes_size(cs_path)
        cs.to_csv("Data/Processed/clothes_size_clean.csv", index=False)
        print(f"\n  ✓ Saved: Data/Processed/clothes_size_clean.csv")
    else:
        print(f"\n  ✗ Clothes-Size-Prediction not found in Data/Raw/")
        print(f"    Download from: https://www.kaggle.com/datasets/tourist55/clothessizeprediction")
        print(f"    Save CSV to Data/Raw/ and re-run this script.")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  PIPELINE COMPLETE")
    print("█" * 70)

    processed = os.listdir("Data/Processed")
    if processed:
        print(f"\n  Files in Data/Processed/:")
        for f in sorted(processed):
            size_mb = os.path.getsize(f"Data/Processed/{f}") / (1024 * 1024)
            print(f"    {f:<35} {size_mb:.1f} MB")
    print()