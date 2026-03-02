# Predicting Clothing Fit: A Machine Learning Approach to Size Recommendation

**ADSP 31017 Machine Learning I — Winter 2026**
**University of Chicago, MS in Applied Data Science**

**Instructor:** Anil Chaturvedi
**Author:** Harleen Kaur Buttar, Skylar Liu, Dora Jiayue Li

---

## Problem Statement

Product size recommendation and fit prediction are critical to improving online shopping experiences and reducing return rates. Customers frequently receive ill-fitting clothing because standard sizing fails to account for variation in body types, brand-specific sizing conventions, and subjective fit preferences. This project builds an end-to-end machine learning pipeline that discovers latent body-type segments, predicts fit outcomes (Small / Fit / Large), and recommends optimal sizes for new customer–product pairs.

## Data

Three datasets support the project pipeline:

### 1. RentTheRunway (PRIMARY) — 192,544 transactions
The richest dataset with fit feedback, body measurements, review text, ratings, and rental occasion context. Used for all three phases.
- **Fit feedback:** Small / Fit / Large
- **Body measurements:** height, weight, bust size, age, body type
- **Review data:** text, summary, rating, date
- **Context:** category, rented-for occasion

### 2. ModCloth (SECONDARY) — 82,790 transactions
Fit feedback with different measurement fields. High sparsity on several columns. Used for cross-retailer generalization testing in Phase 3.
- **Fit feedback:** Small / Fit / Large
- **Body measurements:** height, hips, bra size, cup size (waist/bust >80% null, dropped)
- **Review data:** text, summary, quality rating
- **Context:** category, length feedback

### 3. Clothes-Size-Prediction (SUPPLEMENT) — 119,218 rows
Clean, complete dataset with weight, age, height → size (XXS–XXXL). Used as a baseline benchmark in Phase 2 to compare classifier performance on simple features vs. the richer RentTheRunway feature set.

**Sources:**
- Datasets 1 & 2: [Misra et al., Kaggle](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation)
- Dataset 3: [Dubey, Kaggle](https://www.kaggle.com/datasets/tourist55/clothessizeprediction)

## Project Structure

```
MACHINE-LEARNING-FINAL-PROJECT/
├── README.md
├── download_data.py                # Downloads all 3 datasets from Kaggle
│
├── Data/
│   ├── Raw/
│   │    ├── final_test.csv
│   │    ├── modcloth_final_data.json
│   │    └── renttherunway-final_data.json
│   │              
│   └── Processed/
│        ├── clothes_size_clean.csv
│        ├── modcloth_clean.csv
│        ├── renttherunway_clean.csv
│        └── renttherunway_lda_topics.csv
│                  
├── Notebooks/
│   │
│   │  Phase 1 — Unsupervised Exploration
│   ├── K-means.ipynb               # Body-type segmentation
│   ├── gmm.ipynb                   # Gaussian Mixture Models
│   ├── dbscan.ipynb                # Density-based outlier detection
│   ├── k-modes.ipynb               # Categorical feature clustering
│   ├── lda.ipynb                   # Topic modeling on review text
│   ├── t-SNE.ipynb                 # t-SNE visualization
│   │
│   │  Phase 2 — Supervised Fit Prediction
│   ├── Linear_SVM.ipynb            # Linear Support Vector Machine
│   ├── Kernel_SVM.ipynb            # Kernel-based SVM
│   ├── CART.ipynb                  # Classification and Regression Trees
│   ├── Random_Forest.ipynb         # Random Forest ensemble
│   ├── Bagging.ipynb               # Bagging ensemble
│   ├── KNN.ipynb                   # K-Nearest Neighbors
│   ├── Naive_Bayes.ipynb           # Naive Bayes classifier
│   │
│   │  Phase 3 — Recommender System
│   └── recommender.ipynb           # Collaborative filtering size recommender
│
└── src/
    ├── data_cleaning.py
    ├── download_data.py
    ├── pca.py
    └── t-SNE.py           # Cleans all 3 datasets → Data/Processed/
```

## Methodology

### Phase 1 — Unsupervised Exploration (`01_exploration_clustering.ipynb`)

Discover latent structure in customer body types, product categories, and review language.

| Technique | Application |
|---|---|
| PCA, t-SNE | Dimensionality reduction and visualization of measurement space |
| K-Means, GMM | Body-type segmentation from customer measurements |
| DBSCAN | Outlier body-type detection (customers outside standard sizing) |
| K-Modes | Clustering on categorical features (category, fit feedback) |
| LDA | Topic modeling on review text to extract latent fit-language themes |

**Outputs:** Cleaned dataset with cluster labels, PCA components, LDA topic distributions, and engineered features saved to `data/processed/`.

### Missing Data Handling
During the unsupervised feature discovery phase (DBSCAN, Gaussian Mixture Models, K-modes and K-Means), observations with missing measurement values were excluded prior to model fitting. These clustering methods rely on well-defined distances or likelihoods in the measurement space, and imputing values at this stage would artificially distort the geometry of the data and bias cluster formation. Restricting clustering to complete cases ensured that latent body-type segments were learned from valid, comparable feature vectors, resulting in more meaningful and stable cluster assignments.

For downstream supervised modeling, cluster assignments and topic features were treated as derived inputs rather than core measurements. Because unsupervised models were trained on different subsets of the data, merging their outputs back into the full transaction table naturally reintroduced missing values. Rather than discarding valid observations, missing values were imputed within the modeling pipeline using robust strategies appropriate to each feature type (median imputation for numeric features, most-frequent imputation for categorical features, and constant imputation for binary indicators). This approach preserved sample size, avoided selection bias, and produced models capable of making predictions under realistic conditions where some information may be unavailable.

### Phase 2 — Supervised Fit Prediction (`02_fit_prediction.ipynb`)

Predict fit outcome (Small / Fit / Large) as a multiclass classification problem with imbalanced labels.

**Feature Engineering:**
- Measurement gap features (customer dimension minus product dimension)
- Cluster membership from Phase 1 (GMM, DBSCAN labels)
- LDA topic distributions from review text
- PCA components from measurement space

**Models Compared:**
- Linear SVM and Kernel SVM
- Classification and Regression Trees (CART)
- Random Forest, Bagging, Boosting
- K-Nearest Neighbors
- Naive Bayes

**Evaluation:**
- Stratified K-fold cross-validation
- Precision, Recall, F1-score (per-class and macro-averaged)
- AIC/BIC for mixture model selection
- Class imbalance addressed via SMOTE / class weighting

**Outputs:** Best model saved to `models/`, performance comparison tables and plots saved to `figures/`.

### Phase 3 — Size Recommender (`03_recommender.ipynb`)

Prescriptive system that recommends the size most likely to produce a "Fit" outcome for a new customer–product pair.

**Approach:**
- User-item matrix: customers (by body measurements) × product-size combinations
- Values: fit outcomes (Small=0, Fit=1, Large=2)
- Collaborative filtering identifies customers with similar body types who purchased the same product
- Recommendation: size with highest predicted probability of "Fit"

**Cross-Retailer Generalization:**
- Train on one retailer (e.g., ModCloth), evaluate on the other (RentTheRunway)
- Tests whether learned body-type and fit patterns transfer across retailers

## Key Challenges

- **Sparsity:** Most customers and products have a single transaction, requiring careful imputation and cold-start handling
- **Class imbalance:** Majority of feedback is "Fit," making minority class prediction (Small/Large) methodologically important
- **Subjectivity:** Fit perception varies across customers — the same garment may be "Fit" for one person and "Small" for another
- **Sizing heterogeneity:** Different brands and categories use different sizing conventions, standardized to a numerical scale

## Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
gensim          # LDA topic modeling
nltk            # text preprocessing
```

All notebooks can be executed in **Google Colab** without additional setup.

## Syllabus Alignment

| Session | Topic | Project Coverage |
|---|---|---|
| 1 | Data Discovery | EDA, feature profiling, sparsity analysis |
| 2 | K-Means, GMM, DBSCAN | Body-type clustering, outlier detection |
| 3 | K-Modes, LDA | Categorical clustering, review topic modeling |
| 4 | PCA, Factor Analysis | Measurement dimensionality reduction |
| 5 | Linear SVM, LDA | Fit classification baselines |
| 6 | CART, Random Forest, Boosting | Ensemble fit prediction |
| 7 | Imputation, Outliers | Sparse data handling, DBSCAN noise points |
| 8 | Kernel SVM, KNN, Naive Bayes | Additional classifiers |
| 9 | Recommender Systems | Collaborative filtering size recommender |

## References

- Misra, R., Wan, M., & McAuley, J. (2018). *Decomposing Fit Semantics for Product Size Recommendation in Metric Spaces.* In Proceedings of the 12th ACM Conference on Recommender Systems, pp. 422–426.
- Misra, R., & Grover, J. (2021). *Sculpting Data for ML: The First Act of Machine Learning.* ISBN 9798585463570.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. *An Introduction to Statistical Learning with Applications in Python.* Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. *The Elements of Statistical Learning.* Springer.
- Géron, A. *Hands-On Machine Learning with Scikit-Learn and TensorFlow.* O'Reilly.