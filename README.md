# 🏠 House Sale Price Prediction — Linear Regression, Random Forest & XGBoost

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-brightgreen?logo=xgboost)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> End-to-end regression pipeline for predicting residential property sale prices on the **Kaggle House Prices: Advanced Regression Techniques** dataset. Benchmarks three models — **Linear Regression**, **Random Forest**, and **XGBoost** — across 80 features and 1,460 training records, with final predictions generated for 1,459 unseen test properties.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Pipeline & Methodology](#-pipeline--methodology)
- [Feature Engineering](#-feature-engineering)
- [Models & Results](#-models--results)
- [Prediction Output](#-prediction-output)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🔍 Project Overview

This project solves a classic **supervised regression problem**: predicting the final sale price of residential homes in Ames, Iowa, based on 80 explanatory variables covering physical attributes, quality ratings, neighbourhood characteristics, and transaction metadata.

The workflow demonstrates:

- Robust missing-value imputation across 19 features with incomplete data
- Domain-driven feature engineering (composite area, age, bath count)
- Log-transformation of the skewed target variable for model stability
- Systematic model benchmarking from a linear baseline to gradient boosting
- Submission-ready predictions for all 1,459 test properties

---

## 📊 Dataset

| Attribute            | Detail                                              |
|----------------------|-----------------------------------------------------|
| **Source**           | Kaggle — House Prices: Advanced Regression Techniques |
| **Training records** | 1,460 homes                                         |
| **Test records**     | 1,459 homes                                         |
| **Features**         | 80 (43 categorical · 38 numerical)                  |
| **Target variable**  | `SalePrice` — final sale price in USD               |
| **Target range**     | $34,900 – $755,000 (mean: $180,921; skew: 1.88)    |
| **Missing data**     | 19 columns affected (up to 99.5% in `PoolQC`)       |

### Notable Features

| Feature         | Description                              |
|-----------------|------------------------------------------|
| `OverallQual`   | Overall material and finish quality (1–10) |
| `GrLivArea`     | Above-grade living area (sq ft)          |
| `GarageCars`    | Garage capacity (car count)              |
| `TotalBsmtSF`   | Total basement area (sq ft)              |
| `YearBuilt`     | Original construction year               |
| `Neighborhood`  | Physical location within Ames, Iowa      |

---

## ⚙️ Pipeline & Methodology

```
Raw CSV → EDA → Missing Value Imputation → Feature Engineering
       → Encoding → Log Transform → Split → Train → Evaluate → Predict
```

### Step 1 — Target Analysis
- `SalePrice` exhibits **right skewness (1.88)**; addressed with `log1p` transformation to stabilise variance and improve linear model fit

### Step 2 — Missing Value Imputation
- **Numerical columns**: filled with column median (robust to outliers)
- **Categorical columns**: filled with `'None'` (semantically meaningful — most NAs indicate feature absence, e.g., no pool, no fence)

### Step 3 — Categorical Encoding
- Applied `LabelEncoder` across all 43 categorical variables

### Step 4 — Train / Validation Split
- **80/20 split** with `random_state=42` for reproducibility
- Training set: 1,168 records · Validation set: 292 records

### Step 5 — Modelling
Three models trained and benchmarked:
1. **Linear Regression** — interpretable baseline
2. **Random Forest** — 100 trees, parallelised (`n_jobs=-1`)
3. **XGBoost** — 1,000 estimators, `learning_rate=0.05`, `max_depth=3`

### Step 6 — Test Prediction
- Applied identical preprocessing pipeline to test set
- Aligned test columns to training feature set
- Inverted log transform with `np.expm1` for dollar-scale predictions

---

## 🔧 Feature Engineering

Four domain-informed composite features were created to capture property value drivers more directly:

| New Feature  | Formula                                                        | Rationale                               |
|--------------|----------------------------------------------------------------|-----------------------------------------|
| `TotalSF`    | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF`                          | Total built area across all floors      |
| `TotalBath`  | `FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath` | Weighted bathroom count                 |
| `HouseAge`   | `YrSold − YearBuilt`                                          | Age of property at time of sale         |
| `RemodAge`   | `YrSold − YearRemodAdd`                                       | Years since last remodel at time of sale |

---

## 🏆 Models & Results

| Model              | Validation RMSE (log scale) | Improvement vs Baseline |
|--------------------|-----------------------------|--------------------------|
| Linear Regression  | 0.1559                      | —                        |
| Random Forest      | 0.1442                      | 7.5%                     |
| **XGBoost**        | **Best**                    | **Largest reduction**    |

> ✅ **XGBoost selected as the final model** based on lowest validation RMSE.  
> RMSE is measured on log-transformed `SalePrice`; lower values indicate tighter prediction intervals in dollar terms.

The training vs. validation RMSE gap for Linear Regression was ~0.025 — indicating strong generalisation even from the baseline, and providing a reliable floor for improvement.

---

## 📦 Prediction Output

Final XGBoost predictions for all **1,459 test properties**:

| Statistic | Value       |
|-----------|-------------|
| Minimum   | $43,274     |
| Maximum   | $524,173    |
| Mean      | $175,649    |
| Median    | $153,909    |

Output file: `house_sale_price_predictions.csv`  
Format: `Id`, `SalePrice` — Kaggle submission-ready.

---

## 📁 Project Structure

```
house-sale-price-prediction/
│
├── house_sale_price_prediction.ipynb   # Full pipeline notebook with outputs
│
├── train.csv                           # Training data (1,460 rows × 81 cols)
├── test.csv                            # Test data (1,459 rows × 80 cols)
├── data_description.txt                # Full variable dictionary (80 features)
│
├── house_sale_price_predictions.csv    # Final XGBoost predictions (submission)
│
└── README.md
```

---

## 🛠️ Tech Stack

| Category        | Tools / Libraries                           |
|-----------------|---------------------------------------------|
| **Language**    | Python 3.9+                                 |
| **Data**        | Pandas, NumPy                               |
| **ML Models**   | Scikit-learn (LinearRegression, RandomForestRegressor), XGBoost |
| **Preprocessing** | LabelEncoder, log1p / expm1 transformation |
| **Visualisation** | Matplotlib, Seaborn                       |
| **Environment** | Jupyter Notebook (Kaggle kernel)            |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/shadalishah/house-sale-price-prediction.git
cd house-sale-price-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### 3. Run the notebook

```bash
jupyter notebook house_sale_price_prediction.ipynb
```

> Ensure `train.csv`, `test.csv`, and `data_description.txt` are in the same directory before running. The notebook will generate `house_sale_price_predictions.csv` automatically.

### Run on Kaggle

This notebook was originally developed on **Kaggle Kernels**. To replicate:
1. Fork the notebook at [Kaggle](https://www.kaggle.com)
2. Attach the *House Prices: Advanced Regression Techniques* dataset
3. Run all cells — no local setup required

---

## 👤 Author

**Shad Ali Shah**  
MPhil Economics — Quaid-i-Azam University, Islamabad  
Specialisation: Machine Learning, Applied Econometrics, Big Data

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/shad-ali-shah)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?logo=github)](https://github.com/shadalishah)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-FF6B35)](https://shad-ali-canvas.lovable.app)

---

*Part of an internship data science portfolio. Dataset sourced from Kaggle's House Prices competition (public benchmark dataset).*
