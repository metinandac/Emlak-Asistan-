# Housing Price ML — Full Pipeline

A full end-to-end machine learning project that predicts Seattle-area housing prices using the King County dataset. It combines an XGBoost regression model with a two-stage RAG system, a local LLM (Ollama / Llama 3.2) for natural language feature extraction, an OpenAI GPT-4o-mini benchmark comparison, and a web interface with a chat UI, live stats page, and analysis dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Architecture](#3-architecture)
4. [RAG System](#4-rag-system)
5. [Models Trained](#5-models-trained)
6. [Results & Charts](#6-results--charts)
   - [Price Distribution](#61-price-distribution)
   - [Price vs Living Area](#62-price-vs-living-area)
   - [Correlation Matrix](#63-correlation-matrix)
   - [Price by Bedrooms](#64-price-by-bedrooms)
   - [Price by Condition](#65-price-by-condition)
   - [Price by Grade](#66-price-by-grade)
   - [Feature Importance](#67-feature-importance)
   - [Actual vs Predicted](#68-actual-vs-predicted)
   - [Residual Analysis](#69-residual-analysis)
   - [Model Comparison](#610-model-comparison)
7. [Our Model vs GPT-4o-mini](#7-our-model-vs-gpt-4o-mini)
8. [Web Interface](#8-web-interface)
9. [Setup & Running](#9-setup--running)
10. [File Structure](#10-file-structure)

---

## 1. Project Overview

### Why this matters

Buying a house is one of the largest financial decisions most people will ever make, yet pricing is almost entirely opaque. Buyers walk in with little intuition for whether a property is fairly valued. Real estate agents hold the information advantage. Tools like Zillow exist but are black boxes — they give you a number with no explanation of what drove it, what was assumed, or what comparable properties actually sold for.

This project is a prototype of what a transparent, explainable pricing tool looks like. It tells you the predicted price, shows you exactly which features it assumed (and labels them as inferred), and surfaces three real comparable sales from the dataset so you can judge the estimate yourself rather than just trusting it.

### What it does

The goal is to predict house prices from structured features using classical ML. The project wraps the trained model in a chat interface where a user describes a house in plain English, a local LLM extracts the features, a RAG system fills missing information from similar houses, and the model returns a price estimate with comparable sold listings.

**Full pipeline:**

```
User (natural language)
       ↓
  Chat UI  (Node.js / port 3000)
       ↓
  Flask API  (Python / port 5000)
       ↓
  Ollama — Llama 3.2  →  extracts only explicitly mentioned features
       ↓
  RAG Option 1 — FAISS index  →  fills missing features from similar neighbours
       ↓
  XGBoost Regressor  →  predicts price
       ↓
  RAG Option 2 — FAISS index  →  retrieves 3 comparable sold listings
       ↓
  Response with price + comparables + inferred labels
```

---

## 2. Dataset

**Source:** [Kaggle — House Pricing Dataset (alyelbadry)](https://www.kaggle.com/datasets/alyelbadry/house-pricing-dataset)

| Property | Value |
|---|---|
| Location | King County, Washington (Seattle area) |
| Rows | 21,613 |
| Features | 17 (after preprocessing) |
| Target | `price` (USD) |
| Min price | $75,000 |
| Max price | $7,700,000 |
| Median price | $450,000 |

### Features

| Feature | Type | Description |
|---|---|---|
| `bedrooms` | Numeric | Number of bedrooms |
| `bathrooms` | Numeric | Number of bathrooms (e.g. 2.25) |
| `sqft_living` | Numeric | Living area in square feet |
| `sqft_lot` | Numeric | Lot size in square feet |
| `floors` | Numeric | Number of floors |
| `waterfront` | Binary | Whether the house has a waterfront view |
| `view` | Numeric | View quality score (0–4) |
| `condition` | Ordinal | House condition (0=Poor → 4=Very Good) |
| `grade` | Numeric | Construction quality grade (1–13) |
| `sqft_above` | Numeric | Square footage above ground |
| `sqft_basement` | Numeric | Basement square footage |
| `yr_built` | Numeric | Year the house was built |
| `was_renovated` | Binary | Whether the house has ever been renovated |
| `lat` | Numeric | Latitude |
| `long` | Numeric | Longitude |
| `sqft_living15` | Numeric | Living area of 15 nearest neighbours |
| `sqft_lot15` | Numeric | Lot size of 15 nearest neighbours |

### Preprocessing

- `id` and `date` dropped (non-predictive)
- `zipcode` dropped (lat/long already captures location)
- `waterfront`: `N/Y` → `0/1`
- `condition`: ordinal encoded (`Poor=0`, `Fair=1`, `Average=2`, `Good=3`, `Very Good=4`)
- `yr_renovated`: converted to binary `was_renovated` flag (0 = never renovated)
- Train/test split: **80% train (17,290 rows) / 20% test (4,323 rows)**, `random_state=42`
- Linear models use `StandardScaler`; tree-based models use raw features

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                              │
│   /              Chat UI (index.html)                       │
│   /stats.html    GPT Comparison (stats.html)                │
│   /analysis.html Charts Dashboard (analysis.html)           │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────────────┐
│              Node.js Express Server  :3000                   │
│   POST /chat                 → proxy to Flask /predict      │
│   POST /api/compare          → proxy to Flask /compare      │
│   POST /api/regenerate-charts → spawns charts.py            │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────────────┐
│              Flask API  :5000  (predict.py)                  │
│   POST /predict   LLM extraction + RAG fill + ML predict    │
│   POST /compare   10-sample GPT vs model benchmark          │
└──────────┬──────────────────────────┬───────────────────────┘
           │                          │
    ┌──────▼──────┐           ┌───────▼───────┐
    │  Ollama     │           │  OpenAI API   │
    │  Llama 3.2  │           │  GPT-4o-mini  │
    │  (local)    │           │  (benchmark)  │
    └──────┬──────┘           └───────────────┘
           │ extracted features
    ┌──────▼──────────────────────────┐
    │  FAISS RAG Index (rag.py)       │
    │  21,613 houses embedded via     │
    │  sentence-transformers          │
    │  Option 1: fill missing fields  │
    │  Option 2: get comparables      │
    └──────┬──────────────────────────┘
           │ complete feature vector
    ┌──────▼──────────────┐
    │  XGBoost Regressor  │
    │  R²=0.88  MAE=$70k  │
    └─────────────────────┘
```

---

## 4. RAG System

The RAG system (`rag.py`) uses FAISS with `sentence-transformers/all-MiniLM-L6-v2` embeddings. Each of the 21,613 houses is converted to a text description and embedded. At query time:

**Option 1 — Feature Filling (pre-prediction)**

When a user says *"3 bed waterfront house 2000 sqft grade 9"*, Llama 3.2 only extracts what was explicitly mentioned (4 features). The remaining 13 features need values. Instead of using dumb dataset medians, RAG embeds the user's query, finds the 5 most similar houses in FAISS, and averages their feature values. A grade-9 waterfront house will borrow from other grade-9 waterfront houses — so the inferred bathrooms, view, and basement are contextually appropriate.

Inferred fields are labelled with an amber `inferred` badge in the chat UI so users can see exactly what the model assumed.

**Option 2 — Comparable Listings (post-prediction)**

After predicting a price, RAG retrieves the 3 most similar actual sold listings from the dataset. This grounds the prediction in real sales data and gives the user a reference range, rather than a single opaque number.

---

## 5. Models Trained

Six regression models were trained and evaluated on the same 80/20 split.

| Model | MAE | R² |
|---|---|---|
| **XGBoost** ✅ | **$70,134** | **0.8806** |
| Random Forest | $72,846 | 0.8572 |
| Gradient Boosting | $77,911 | 0.8621 |
| Linear Regression | $128,161 | 0.6957 |
| Ridge | $128,144 | 0.6957 |
| Lasso | $127,929 | 0.6950 |

With 21,613 rows, tree-based models now dramatically outperform linear ones (R² 0.88 vs 0.70). XGBoost wins because it has the most regularisation and is best able to exploit the larger dataset. On the old 546-row dataset, XGBoost was the worst performer — dataset size is critical for complex ensemble models.

---

## 6. Results & Charts

### 6.1 Price Distribution

![Price Distribution](docs/charts/1_price_distribution.png)

Prices are **right-skewed** with most houses in the $200k–$700k range and a long tail of luxury properties up to $7.7M. The skew is more pronounced than the old dataset because the King County market includes multi-million dollar waterfront properties. The model is trained on raw prices — log-transforming the target would likely reduce error on high-value outliers.

---

### 6.2 Price vs Living Area

![Price vs Sqft](docs/charts/2_price_vs_sqft.png)

Living area (`sqft_living`) is the strongest single predictor of price. Waterfront properties (green) command a **significant premium** at all size levels — a 2,000 sqft waterfront house typically sells for $200k–$400k more than an equivalent inland property. The scatter is wide, confirming that area alone is insufficient and other features (grade, location, condition) add substantial explanatory power.

---

### 6.3 Correlation Matrix

![Correlation Heatmap](docs/charts/3_correlation_heatmap.png)

Key correlations with `price`:
- `sqft_living` (~0.70) — strongest predictor
- `grade` (~0.67) — construction quality matters almost as much as size
- `sqft_above`, `sqft_living15` (~0.60) — neighbourhood size confirms location quality
- `bathrooms` (~0.53) — correlated with overall house size
- `waterfront` (~0.27) — lower raw correlation but represents massive absolute price difference
- `yr_built` (~-0.05) — almost no linear relationship; older houses in desirable areas are just as expensive

---

### 6.4 Price by Bedrooms

![Price by Bedrooms](docs/charts/4_price_by_bedrooms.png)

Price peaks around **4–5 bedrooms** but the relationship is noisy. Very large bedroom counts (8+) don't reliably command higher prices — these tend to be subdivided or lower-grade properties. The wide interquartile ranges reflect that location and grade dominate over bedroom count alone.

---

### 6.5 Price by Condition

![Price by Condition](docs/charts/5_price_by_condition.png)

Condition has a **weaker effect than expected**. Very Good condition houses do command a premium, but Poor and Fair condition houses overlap significantly with Average. This is likely because buyers in King County often purchase lower-condition houses for renovation in desirable neighbourhoods — location overrides condition.

---

### 6.6 Price by Grade

![Price by Grade](docs/charts/6_price_by_grade.png)

Construction grade shows a **steep, near-exponential relationship** with price. The jump from grade 7 (average) to grade 10+ (luxury) more than triples the mean price. This makes grade one of the most important features in the model — and explains why the feature importance chart ranks it highly alongside `sqft_living`.

---

### 6.7 Feature Importance

![Feature Importance](docs/charts/7_feature_importance.png)

XGBoost feature importances confirm the correlation analysis:
- **`sqft_living`** and **`grade`** dominate
- **`lat`** and **`long`** are highly important — location within King County is a massive price driver
- **`sqft_living15`** (neighbourhood living area) captures micro-location effects
- **`yr_built`** and **`was_renovated`** contribute modestly
- **`waterfront`** appears lower than expected due to its rarity (only ~1% of houses), but its absolute price impact is huge

---

### 6.8 Actual vs Predicted

![Actual vs Predicted](docs/charts/8_actual_vs_predicted.png)

With R²=0.88, the model tracks actual prices closely across the full range. The fit is visibly tighter than the old dataset (R²=0.66). The main failure mode is **luxury outliers above $2M** — the model under-predicts these because they are rare in training data and involve unique features (waterfront, grade 12–13) that the model hasn't seen enough of.

---

### 6.9 Residual Analysis

![Residuals](docs/charts/9_residuals.png)

**Left — Residual distribution:** Well-centred around zero, roughly symmetric, indicating no systematic bias. The tails are heavier than a normal distribution — these are the luxury outliers the model misses.

**Right — Residuals vs Fitted:** The fanning pattern (heteroscedasticity) is visible but mild. Residual variance grows with predicted price, which is expected — a $100k error on a $200k house is severe, but a $100k error on a $2M house is acceptable. Log-transforming the target would flatten this pattern.

---

### 6.10 Model Comparison

![Model Comparison](docs/charts/10_model_comparison.png)

XGBoost leads on both MAE and R². All three tree-based models substantially outperform the linear baselines — the non-linear interactions between grade, sqft, and location are strong enough that linear models leave significant accuracy on the table. The gap between linear (R²~0.70) and tree-based (R²~0.88) models is much larger here than on the old 546-row dataset, because more data allows trees to learn finer-grained decision boundaries.

---

## 7. Our Model vs GPT-4o-mini

`compare.py` samples 10 houses from the held-out test set, describes each in plain English including lat/long coordinates, and asks GPT-4o-mini to predict the price in USD. Unlike the old Indian dataset where GPT had no market knowledge, GPT-4o-mini does know Seattle real estate — making this a fairer and more interesting comparison.

Run it via the `/stats.html` page or directly:

```bash
python compare.py
```

**Why the comparison is interesting now:** GPT has genuine knowledge of King County prices, so it can sometimes outperform on individual samples — especially when the description strongly implies a neighbourhood GPT recognises. But our trained model has seen 17,290 actual transactions and wins on average MAE.

---

## 8. Web Interface

### Chat — `/`
Describe a house in plain English. Llama 3.2 extracts only what you explicitly mention. RAG fills in the rest from similar neighbours (labelled with amber `inferred` badges). The response shows the predicted price plus 3 comparable sold listings from the dataset.

### Stats — `/stats.html`
Click **Run Comparison** to benchmark our model vs GPT-4o-mini on 10 real test samples. Shows per-row winner, MAE comparison, and overall winner card.

### Analysis — `/analysis.html`
10 publication-style charts across 6 sections with downloadable PNGs. The **↺ Regenerate** button re-runs `charts.py` in place.

---

## 9. Setup & Running

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) with `llama3.2` pulled
- OpenAI API key (for GPT comparison)
- Kaggle API credentials (to download dataset)

### Install

```bash
python -m venv env
source env/bin/activate
pip install scikit-learn pandas numpy joblib xgboost flask openai \
            python-dotenv matplotlib seaborn faiss-cpu sentence-transformers

ollama pull llama3.2
npm install
echo 'OPENAI_API_KEY=your_key_here' > .env
```

### Download dataset

```bash
kaggle datasets download -d alyelbadry/house-pricing-dataset --path . --unzip
```

### Train & build index

```bash
python main.py          # trains models, saves model.pkl / scaler.pkl / features.pkl / dataset.pkl
python rag.py           # builds FAISS index, saves rag_index.faiss / rag_rows.pkl
python charts.py        # generates all 10 charts to public/charts/
```

### Run

```bash
# Terminal 1
python predict.py       # Flask API on :5000

# Terminal 2
node server.js          # Express on :3000
```

Open [http://localhost:3000](http://localhost:3000).

---

## 10. File Structure

```
.
├── main.py              # Train all 6 models, save best (XGBoost) as model.pkl
├── predict.py           # Flask API: /predict and /compare
├── rag.py               # RAG index: build + query FAISS with sentence-transformers
├── compare.py           # CLI: 10-sample model vs GPT-4o-mini comparison
├── charts.py            # Generate all 10 analysis charts as PNGs
├── server.js            # Node.js Express server (port 3000)
├── package.json
├── .env                 # OPENAI_API_KEY (gitignored)
├── house_prices.csv     # Dataset (gitignored — download from Kaggle)
│
├── model.pkl            # Saved XGBoost model
├── scaler.pkl           # StandardScaler (for linear models)
├── features.pkl         # Ordered feature name list
├── dataset.pkl          # Preprocessed dataframe (used by RAG builder)
├── rag_index.faiss      # FAISS vector index (gitignored — build with rag.py)
├── rag_rows.pkl         # Raw house rows aligned to FAISS index (gitignored)
│
├── public/
│   ├── index.html       # Chat UI
│   ├── stats.html       # GPT comparison page
│   ├── analysis.html    # Charts & analysis dashboard
│   └── charts/          # Generated chart PNGs (gitignored)
│
└── docs/
    └── charts/          # Chart PNGs tracked in git (embedded in this README)
```
