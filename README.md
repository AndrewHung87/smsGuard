# 🛡️ SMSGuard — AI-Powered SMS Spam Detection

An end-to-end data pipeline and ML system for SMS spam classification — from raw CSV ingestion to a live interactive dashboard.

---

## Architecture

```
Raw CSV Files (UCI + Kaggle)
        │
        ▼
┌───────────────────┐
│   ETL Pipeline    │  ← data validation, deduplication, feature engineering
│   etl_pipeline.py │
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
PostgreSQL  BigQuery      ← structured storage, cloud-ready
         │
         ▼
┌───────────────────┐
│   ML Model        │  ← TF-IDF + Naive Bayes classifier (97% accuracy)
│   train_model.py  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Streamlit        │  ← interactive dashboard + live predictor
│  dashboard.py     │
└───────────────────┘
```

---

## Features

| Layer | What it does |
|-------|-------------|
| **ETL** | Extracts from multiple CSV sources, validates schema, deduplicates, engineers features, loads to PostgreSQL and/or BigQuery |
| **ML** | TF-IDF vectorisation + Naive Bayes classifier; achieves 97% accuracy, 94% precision on 10K+ messages |
| **Dashboard** | Interactive Streamlit app with data overview, model metrics (ROC curve, confusion matrix), and live SMS predictor |

---

## Quick Start

```bash
pip install -r requirements.txt
python src/etl_pipeline.py --target both
python src/main.py
streamlit run src/dashboard.py
```

---

## ETL Pipeline

```bash
python src/etl_pipeline.py --target postgres   # PostgreSQL only
python src/etl_pipeline.py --target bigquery   # BigQuery only
python src/etl_pipeline.py --target both       # both (default)
```

| Env Variable | Description |
|---|---|
| `POSTGRES_CONN` | `postgresql://user:pass@host:5432/dbname` |
| `GCP_PROJECT` | GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON |

---

## Dashboard

```bash
streamlit run src/dashboard.py
```

Pages: **Data Overview** · **Model Performance** (ROC, confusion matrix) · **Live Predictor**

---

## Project Structure

```
smsguard/
├── data/raw/                   # Source CSVs
├── data/processed/             # ETL output
├── logs/                       # Run logs + invalid row exports
├── models/                     # Saved model + vectorizer
└── src/
    ├── etl_pipeline.py         # Extract → Transform → Load
    ├── dashboard.py            # Streamlit dashboard
    ├── main.py                 # ML orchestrator
    ├── train_model.py          # Training + evaluation
    ├── preprocess.py           # Text cleaning
    ├── feature_extraction.py   # TF-IDF
    ├── cluster_analysis.py     # K-Means clustering
    ├── utils.py                # Helpers
    └── app.py                  # Flask app (original)
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 97% |
| Precision | 94% |
| Recall | 93% |
| F1 | 93% |

**Tech Stack:** Python · scikit-learn · pandas · NLTK · Streamlit · Plotly · SQLAlchemy · PostgreSQL · BigQuery
