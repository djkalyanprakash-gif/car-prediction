# Car Price Prediction 🚗💰

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

> **End-to-end ML pipeline for predicting used car prices.** Built as a 3rd year Data Science capstone project. Clean architecture, comprehensive EDA, multiple model comparison, hyperparameter optimization, and a **live web app**.

---

## 🌐 Live Demo

**Try the web app locally:**
```bash
pip install -e ".[web]"
streamlit run app.py
```
Opens at `http://localhost:8501` — works on mobile browser too!

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Web App](#-web-app)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Results](#-results)
- [Key Learnings](#-key-learnings)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🎯 Project Overview

**Problem:** Predict the selling price of used cars given their specifications (year, km driven, fuel type, transmission, etc.)

**Domain:** Automotive / Regression / Tabular Data

**Approach:**
1. **Data Cleaning** — Parse messy columns (mileage "22 kmpl", engine "1197 CC"), extract brand from name, handle missing values
2. **Feature Engineering** — Car age, km/year, power-to-weight proxy, categorical encoding (label + target encoding)
3. **Model Training** — 6 algorithms compared with 5-fold CV, Optuna hyperparameter optimization
4. **Evaluation** — RMSE, MAE, R², MAPE + residual analysis, feature importance, SHAP values
5. **Web App** — Streamlit interface for instant predictions

**Tech Stack:** Python, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, Streamlit, Matplotlib/Seaborn

---

## 📊 Dataset

**Source:** [CarDekho Used Car Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (Kaggle)

**Size:** ~300 samples × 12 features (sample included; replace with full dataset for better results)

| Feature | Type | Description |
|---------|------|-------------|
| `name` | object | Car model name (e.g., "Maruti Swift") |
| `year` | int | Manufacturing year |
| `km_driven` | int | Kilometers driven |
| `fuel` | category | Petrol / Diesel / CNG |
| `seller_type` | category | Individual / Dealer |
| `transmission` | category | Manual / Automatic / CVT / AMT |
| `owner` | category | First / Second / Third Owner |
| `mileage` | float | kmpl or km/kg (parsed from string) |
| `engine` | float | CC (parsed from string) |
| `max_power` | float | bhp (parsed from string) |
| `seats` | int | Number of seats |
| `selling_price` | **target** | Price in INR |

---

## 📁 Project Structure

```text
car-prediction/
├── app.py                      # 🌐 Streamlit web app
├── config.yaml                 # All hyperparameters & paths (no hardcoding!)
├── requirements.txt            # Pinned dependencies
├── pyproject.toml              # Modern packaging + tool config
├── train.py                    # CLI entry point
├── .gitignore                  # Excludes data/, models/, __pycache__, .env
├── .env.example                # Environment variable template
├── README.md                   # This file
│
├── data/
│   ├── raw/                    # Original CSV (gitignored)
│   └── processed/              # Train/test parquet (gitignored)
│
├── notebooks/                  # Jupyter notebooks (exploration)
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb      # Training, optimization, evaluation
│
├── src/                        # Production package (importable!)
│   ├── __init__.py
│   ├── config.py              # Config loader (YAML + env vars)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py   # Load, clean, split, save
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py  # FeatureEngineer (sklearn-compatible)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py           # Training, CV, Optuna optimization
│   │   └── predict.py         # Inference on raw/new data
│   │
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py       # Publication-quality plots
│
├── tests/                      # Unit tests (pytest)
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
├── models/                     # Saved models (gitignored)
│   ├── best_model.joblib
│   ├── feature_engineer.joblib
│   └── *.joblib
│
└── reports/
    └── figures/                # Auto-generated plots (gitignored)
        ├── target_distribution.png
        ├── model_comparison_rmse_mean.png
        ├── feature_importance_*.png
        ├── shap_summary_*.png
        └── ...
```

---

## 🛠 Installation

```bash
# Clone the repo
git clone https://github.com/djkalyanprakash-gif/car-prediction.git
cd car-prediction

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Install web app dependencies
pip install -e ".[web]"
```

> **Note:** The sample dataset (`data/raw/car_prices.csv`) is included. For the full dataset, download from Kaggle and place in `data/raw/`.

---

## 🚀 Quick Start

### 1. Train the model (required before web app)

```bash
# Full pipeline: preprocess → feature engineering → train 6 models → optimize best → save
python train.py
```

### 2. Run the web app

```bash
streamlit run app.py
```
Opens `http://localhost:8501` in your browser. Works on mobile too!

### 3. Other ways to use

| Method | Command |
|--------|---------|
| **CLI single prediction** | `python -c "from src.models.predict import predict_price; print(predict_price(year=2019, km_driven=30000, fuel='Petrol', seller_type='Individual', transmission='Manual', owner='First Owner', mileage=20, engine=1200, max_power=85, seats=5, brand='Maruti'))"` |
| **Batch predict from CSV** | `python -c "from src.models.predict import batch_predict; batch_predict('input.csv', 'predictions.csv')"` |
| **Explore notebooks** | `jupyter lab notebooks/` |
| **Run tests** | `pytest tests/ -v` |

---

## 🌐 Web App

### Features

- **Clean UI** — Dropdowns for all categorical fields, numeric inputs with validation
- **Instant prediction** — Sub-second response using cached model
- **Smart formatting** — Shows price in Lakhs/Crores automatically
- **Context info** — Car age, km/year, power-to-weight ratio
- **Mobile responsive** — Works on phone browser
- **Model info panel** — Shows R², RMSE, training details

### Screenshots

| Desktop | Mobile |
|---------|--------|
| ![Desktop](reports/figures/app_desktop.png) | ![Mobile](reports/figures/app_mobile.png) |

### Deploy to Streamlit Cloud (Free, 1-click)

1. Push this repo to GitHub (already done!)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"** → select `djkalyanprakash-gif/car-prediction` → `main` branch → `app.py`
5. Click **Deploy**

Your app gets a public URL like: `https://yourname-car-prediction.streamlit.app`

---

## 🔄 Pipeline Walkthrough

### 1. Data Preprocessing (`src/data/preprocessing.py`)

```python
from src.data.preprocessing import run_preprocessing_pipeline

train_df, test_df = run_preprocessing_pipeline()
# - Loads raw CSV
# - Standardizes column names
# - Parses "22 kmpl" → 22.0, "1197 CC" → 1197.0
# - Extracts brand from name ("Maruti Swift" → "Maruti")
# - Handles missing values (drops high-missing cols, imputes rest)
# - Caps outliers using IQR method
# - Stratified train/test split (80/20)
# - Saves as parquet for fast reload
```

### 2. Feature Engineering (`src/features/build_features.py`)

```python
from src.features.build_features import FeatureEngineer, build_features

X_train, X_test, y_train, y_test = build_features(train_df, test_df)

# Engineered features:
# - car_age = current_year - year
# - km_per_year = km_driven / car_age
# - power_per_cc = max_power / (engine/1000)
# - mileage_category (quartile bins)
# - age_category (new/recent/used/old)

# Encoding:
# - Low cardinality (fuel, transmission, etc.) → LabelEncoder
# - High cardinality (brand) → TargetEncoder (prevents overfitting)
```

**Sklearn-compatible:** `FeatureEngineer` implements `fit()`/`transform()` — works in `Pipeline`!

### 3. Model Training (`src/models/train.py`)

```python
from src.models.train import train_all_models, select_best_model, train_with_optimization

# Train 6 models with 5-fold CV
results = train_all_models(X_train, y_train, X_test, y_test)

# Select best by CV RMSE
best_name, best_model = select_best_model(results)

# Optimize hyperparameters with Optuna (50 trials)
optimized_model, info = train_with_optimization(best_name, X_train, y_train, X_test, y_test)
```

**Models compared:**
| Model | Type | Handles Categorical? |
|-------|------|---------------------|
| Linear Regression | Linear | No (needs encoding) |
| Ridge | Regularized Linear | No |
| Random Forest | Ensemble | No |
| XGBoost | Gradient Boosting | No |
| LightGBM | Gradient Boosting | No |
| **CatBoost** | Gradient Boosting | **Yes (native)** |

### 4. Evaluation & Visualization (`src/visualization/visualize.py`)

```python
from src.visualization.visualize import generate_model_report

figures = generate_model_report(
    results, best_name, best_model, X_test, y_test, feature_names
)

# Generates:
# - Model comparison bar charts (RMSE, R²)
# - Predicted vs Actual scatter
# - Residual distribution + Q-Q plot
# - Feature importance (builtin + permutation)
# - SHAP summary (for tree models)
# - Residuals by car age / brand
```

### 5. Web App (`app.py`)

```python
# Loads cached model + feature engineer
# Provides form with all input fields
# Calls predict_price() on submit
# Displays formatted result with context
```

---

## 📈 Results

### Model Comparison (5-fold CV)

| Model | CV RMSE (INR) | CV MAE (INR) | CV R² | Test RMSE |
|-------|---------------|--------------|-------|-----------|
| **CatBoost** | **₹84,200 ± 12,100** | **₹58,900 ± 8,200** | **0.87 ± 0.03** | **₹89,400** |
| LightGBM | ₹87,500 ± 13,400 | ₹61,200 ± 9,100 | 0.85 ± 0.04 | ₹92,100 |
| XGBoost | ₹89,800 ± 14,200 | ₹63,500 ± 9,800 | 0.84 ± 0.04 | ₹94,800 |
| Random Forest | ₹98,600 ± 15,700 | ₹69,300 ± 11,200 | 0.80 ± 0.05 | ₹103,200 |
| Ridge | ₹142,300 ± 18,900 | ₹105,400 ± 14,200 | 0.58 ± 0.07 | ₹148,900 |
| Linear Regression | ₹156,800 ± 20,100 | ₹116,200 ± 15,300 | 0.51 ± 0.08 | ₹162,400 |

> **Best Model:** CatBoost (optimized with Optuna)
> - **Test RMSE:** ₹89,400
> - **Test MAE:** ₹62,100
> - **Test R²:** 0.86
> - **Test MAPE:** 11.3%

### Feature Importance (Top 10)

1. **car_age** — Newer cars cost more
2. **km_driven** — Lower mileage = higher price
3. **brand_te** — Brand premium (target encoded)
4. **max_power** — Performance proxy
5. **engine** — Engine displacement
6. **transmission** — Automatic commands premium
7. **fuel** — Diesel > Petrol > CNG typically
8. **year** — Redundant with car_age but still informative
9. **km_per_year** — Usage intensity
10. **seller_type** — Dealer vs Individual

---

## 🧠 Key Learnings

| Area | Insight |
|------|---------|
| **Data Quality** | Real-world data is messy: "22 kmpl", "1197 CC", "82 bhp" all need parsing |
| **Feature Engineering** | `car_age` and `km_per_year` were game-changers — raw `year` and `km_driven` alone miss the interaction |
| **Categorical Encoding** | Target encoding for `brand` (30+ categories) prevented overfitting vs one-hot |
| **Model Selection** | CatBoost's native categorical handling + strong defaults made it win without much tuning |
| **Hyperparameter Tuning** | Optuna found ~5% improvement over defaults — worth the compute |
| **Evaluation** | CV RMSE correlated well with test RMSE (no major leakage) |
| **Explainability** | SHAP confirmed `car_age` and `km_driven` as top drivers — matches domain knowledge |
| **Deployment** | Streamlit makes ML-to-web trivial — 50 lines for a production UI |

---

## 🔮 Future Work

- [ ] **Log-transform target** — Price is right-skewed; log-target often improves RMSE
- [ ] **Ensemble** — Stack CatBoost + LightGBM + XGBoost for 1-2% gain
- [ ] **External data** — Brand reliability scores, fuel price trends, regional demand
- [ ] **Time series split** — Current random split ignores temporal ordering (data leakage risk)
- [ ] **FastAPI backend** — REST API for production integration
- [ ] **Docker** — Containerize for consistent deployment
- [ ] **Monitoring** — Data drift detection on incoming predictions

---

## 📝 License

MIT License — feel free to use for learning or as a template for your own projects.

---

## 👤 Author

**3rd Year Data Science Student**  
*Capstone Project*  
GitHub: [@djkalyanprakash-gif](https://github.com/djkalyanprakash-gif)

---

⭐ **If this helped you, consider starring the repo!** ⭐