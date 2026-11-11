# KALYAN™ Car Price Prediction 🚗💰

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A learning project to predict used car prices in India. I am still learning, and I built this step by step — it is not a production system.

## 🌐 Live demo — try it first

**The working web app is live here:**

**[https://djkalyanprakash-gif.github.io/car-prediction/](https://djkalyanprakash-gif.github.io/car-prediction/)**

That is the KALYAN™ Car Price Predictor. It is the file `index.html` in this repo, published with GitHub Pages from the `main` branch, `/` (root). It runs in the browser with no server and no login — just enter the car details and press Predict Price.

The Streamlit app in `app.py` is the original ML version of the same idea. It needs a trained model file to work (see “What I struggled with” below). If the model file is missing, `app.py` now shows a clear message and offers a demo mode with a simple, transparent estimate, so the page still works.

---

## 👤 About me

Hi, I am **Kalyan**.

I am a BTech 3rd-year student in Computer Science Engineering, with a specialisation in Data Science.

I would describe myself honestly as a mid student — not a topper — but I tried very hard on this project. I wanted to learn the full path, not just copy a model: cleaning messy data, making features, comparing models, and then putting the result in an app that someone can actually open and use.

GitHub: [@djkalyanprakash-gif](https://github.com/djkalyanprakash-gif)

---

## 🎯 What this project does

Given a used car's details, it estimates a selling price in INR:

- Brand, year, kilometres driven
- Fuel type, seller type, transmission, owner
- Mileage, engine (CC), max power (bhp), seats

It then shows the price in Lakhs / Crores, plus simple context such as car age, km per year, and power-to-weight.

There are two ways to use it in this repo:

1. **Web app (recommended):** `index.html` — the live KALYAN™ demo above. No Python needed.
2. **Streamlit app (original ML version):** `app.py` — needs the trained model. See “How to run Streamlit locally”.

---

## 📋 Contents

- [About me](#-about-me)
- [What this project does](#-what-this-project-does)
- [Live demo](#-live-demo--try-it-first)
- [Dataset](#-dataset)
- [Project structure](#-project-structure)
- [How to run the web app](#-how-to-run-the-web-app)
- [How to run Streamlit locally](#-how-to-run-streamlit-locally)
- [How the ML pipeline works](#-how-the-ml-pipeline-works)
- [Results on the sample dataset only](#-results-on-the-sample-dataset-only)
- [What I learned](#-what-i-learned)
- [What I struggled with / known limitations](#-what-i-struggled-with--known-limitations)
- [Future work I want to learn next](#-future-work-i-want-to-learn-next)
- [License](#-license)

---

## 📊 Dataset

**Source:** [CarDekho Used Car Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (Kaggle)

**Size:** about 300 samples × 12 features in the sample used here. That is small, so please treat the results as a learning exercise, not as market prices. For better results, use the full dataset from Kaggle and place it in `data/raw/`.

| Feature | Type | Description |
|---------|------|-------------|
| `name` | object | Car model name (e.g., "Maruti Swift") |
| `year` | int | Manufacturing year |
| `km_driven` | int | Kilometres driven |
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

## 📁 Project structure

```text
car-prediction/
├── index.html                  # 🌐 KALYAN™ web app (GitHub Pages demo)
├── app.py                      # 🌐 Streamlit web app (original ML version)
├── config.yaml                 # Settings and hyperparameters
├── requirements.txt            # Pinned dependencies
├── pyproject.toml              # Packaging and tool config
├── train.py                    # CLI entry point for training
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
├── src/
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
│       └── visualize.py       # Plots
│
├── tests/                      # Unit tests (pytest)
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
├── models/                     # Saved models (gitignored, not in the repo)
│   ├── best_model.joblib
│   ├── feature_engineer.joblib
│   └── *.joblib
│
└── reports/
    └── figures/                # Auto-generated plots (gitignored)
```

---

## 🌐 How to run the web app

No installation needed.

1. Open <https://djkalyanprakash-gif.github.io/car-prediction/>
2. Or, locally, open `index.html` in this repo in a browser.

Try the examples on the page first — for example a 2021 Hyundai Creta — then change the year, km driven, fuel or transmission and press Predict Price again.

---

## 🛠 How to run Streamlit locally

```bash
# Clone the repo
git clone https://github.com/djkalyanprakash-gif/car-prediction.git
cd car-prediction

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Install web app dependencies
pip install -e ".[web]"
```

### 1. Train the model (needed before the Streamlit app can use the ML model)

```bash
# Full pipeline: preprocess → feature engineering → train models → optimize best → save
python train.py
```

This needs `data/raw/car_prices.csv`. That file is not in the repo (it is gitignored). If it is missing, training will fail — that is one of the things I learned the hard way, see below.

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser. Works on a phone browser too.

**If there is no trained model yet:** the app now tells you clearly that `models/best_model.joblib` was not found, and offers a demo mode. In demo mode you can still enter car details and get a simple, transparent estimate — it is labelled as a demo estimate, not a trained-model prediction.

### 3. Other ways to use it

| Method | Command |
|--------|---------|
| **CLI single prediction** | `python -c "from src.models.predict import predict_price; print(predict_price(year=2019, km_driven=30000, fuel='Petrol', seller_type='Individual', transmission='Manual', owner='First Owner', mileage=20, engine=1200, max_power=85, seats=5, brand='Maruti'))"` |
| **Batch predict from CSV** | `python -c "from src.models.predict import batch_predict; batch_predict('input.csv', 'predictions.csv')"` |
| **Explore notebooks** | `jupyter lab notebooks/` |
| **Run tests** | `pytest tests/ -v` |

### Deploy to Streamlit Cloud

The original README suggested Streamlit Cloud: sign in at [share.streamlit.io](https://share.streamlit.io), choose this repo, `main` branch, `app.py`, and Deploy. Please note this will only work if the model file and the training data are available in that deployment — they are not in this repo, so I would not recommend that route until the training step is fixed. The GitHub Pages web app above is the demo that actually works right now.

---

## 🔄 How the ML pipeline works

1. **Data cleaning** — Parse messy columns (mileage "22 kmpl", engine "1197 CC"), extract brand from name, handle missing values.
2. **Feature engineering** — Car age, km/year, power-to-weight proxy, categorical encoding (label + target encoding). See `src/features/build_features.py`.
3. **Model training** — Several algorithms compared with 5-fold cross-validation, then hyperparameter search with Optuna for the best one.
4. **Evaluation** — RMSE, MAE, R², MAPE, plus residual analysis, feature importance and SHAP values.
5. **Web app** — Streamlit interface for instant predictions (`app.py`).

**Tech I used:** Python, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, Streamlit, Matplotlib / Seaborn.

---

## 📈 Results on the sample dataset only

Please read this carefully: the numbers below are **as reported in the original README, on the small sample dataset, and unverified here**. I have not re-run them as verified production results, and they should not be quoted as if this were a finished or production model.

The original README reported, on the sample:

- Best model reported: CatBoost (optimised with Optuna)
- Test R² reported: 0.86
- Test RMSE reported: ₹89,400
- Test MAE reported: ₹62,100
- Test MAPE reported: 11.3%

The original README also listed a model comparison table (5-fold CV) with CatBoost first, then LightGBM, XGBoost, Random Forest, Ridge and Linear Regression. I am keeping that table out of this version because I cannot verify those numbers from the sample alone, and I do not want to present them as my own verified results.

What I can say from the work itself is which features seemed to matter most in the exploration: car age, km driven, brand, max power and engine size. That matches what you would expect — newer cars and lower mileage usually cost more.

---

## 🧠 What I learned

- **Real data is messy.** "22 kmpl", "1197 CC" and "82 bhp" all look like text, and I had to parse them into numbers before any model could use them.
- **Features matter more than I expected.** `car_age` and `km_per_year` helped more than just using `year` and `km_driven` on their own.
- **Encoding categories is a real decision.** I learned the difference between label encoding for low-cardinality fields (fuel, transmission) and target encoding for brand, which has many values.
- **A model is not an app.** Training something in a notebook is one thing. Making it open in a browser, handling a missing model file, and showing a clear error instead of a crash is a different skill — and the one I learned most from here.
- **Small data limits everything.** With about 300 samples, the model can only learn so much. More and better data would help more than tuning.
- **Deployment has its own traps.** Files that are gitignored (the model and the CSV) do not exist on a fresh deploy, even if they exist on my laptop.

---

## ⚠️ What I struggled with / known limitations

Honest list — these are real, and I am still working on them:

- **The Streamlit app used to error on a fresh install.** The trained model (`models/best_model.joblib`) is gitignored, so it is not in the repo. On a fresh clone or on Streamlit Cloud, `app.py` used to just show "Model not found" and stop. I have now added a demo mode so the page still works, but the demo estimate is not the trained model.
- **The training data is also not in the repo.** `data/raw/car_prices.csv` is gitignored, so `python train.py` cannot run from a fresh clone without downloading the CarDekho dataset from Kaggle first.
- **Single-row predictions used to crash.** `pd.qcut` on mileage with one row fails with `ValueError: Bin labels must be one fewer than the number of bin edges`. I fixed this in `src/features/build_features.py` by only using `qcut` when there are enough rows and unique values, and otherwise using a safe default category ("medium").
- **Small sample, so prices are estimates only.** The app gives a rough guide, not a market price. Real prices vary by condition, location, service history, negotiation and the market at the time.
- **I am still learning evaluation properly.** I can report RMSE / MAE / R², but I am still learning how to check for leakage, how to split by time, and how to trust a result on such a small dataset.
- **A small CSS typo in `app.py`.** The button hover used `linear-gradient(135deg., …)` — the extra dot. Fixed to `135deg,`.

If you are a teacher or recruiter reading this: I know these limits. I would rather list them honestly than hide them.

---

## 🔮 Future work I want to learn next

- Get the training step working from a fresh clone, with the dataset clearly documented, so the Streamlit app can actually use the trained model.
- Learn log-transforming the price target, since prices are right-skewed.
- Try a simple ensemble (CatBoost + LightGBM + XGBoost) and learn when it helps and when it does not.
- Learn proper time-based splitting, so I do not leak future information into training.
- Learn how to deploy properly — for example a small FastAPI backend, or Docker — instead of relying on files that only exist on my laptop.
- Add better input checks in the app, and clearer explanations of why a price went up or down.

---

## 📝 License

MIT License — feel free to use for learning or as a template for your own projects.

---

## 👤 Author

**Kalyan** — BTech 3rd year, Computer Science Engineering (Data Science specialisation)  
GitHub: [@djkalyanprakash-gif](https://github.com/djkalyanprakash-gif)

---

⭐ If this helped you, consider starring the repo! ⭐
