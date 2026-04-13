"""
SMSGuard ETL Pipeline
Extract → Transform → Load (PostgreSQL + BigQuery)

Usage:
    python etl_pipeline.py --target postgres   # Load to PostgreSQL
    python etl_pipeline.py --target bigquery   # Load to BigQuery
    python etl_pipeline.py --target both       # Load to both (default)
"""

import os
import re
import logging
import argparse
import hashlib
from datetime import datetime

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACT
# ═════════════════════════════════════════════════════════════════════════════

def extract(uci_path: str, kaggle_path: str) -> pd.DataFrame:
    """Load raw CSV files and combine into a single DataFrame."""
    logger.info("── EXTRACT ──────────────────────────────────────")
    frames = []

    # UCI dataset: tab-separated, no header
    if os.path.exists(uci_path):
        df_uci = pd.read_csv(uci_path, sep="\t", names=["label", "message"], header=None)
        df_uci["source"] = "uci"
        logger.info(f"UCI dataset loaded: {len(df_uci):,} rows from '{uci_path}'")
        frames.append(df_uci)
    else:
        logger.warning(f"UCI file not found: {uci_path}")

    # Kaggle dataset: has header row (v1, v2, ...)
    if os.path.exists(kaggle_path):
        df_kaggle = pd.read_csv(kaggle_path, encoding="latin-1")[["v1", "v2"]]
        df_kaggle.columns = ["label", "message"]
        df_kaggle["source"] = "kaggle"
        logger.info(f"Kaggle dataset loaded: {len(df_kaggle):,} rows from '{kaggle_path}'")
        frames.append(df_kaggle)
    else:
        logger.warning(f"Kaggle file not found: {kaggle_path}")

    if not frames:
        raise FileNotFoundError("No source files found. Check data/raw/ directory.")

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined raw rows: {len(df):,}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ═════════════════════════════════════════════════════════════════════════════

_stop_words = set(stopwords.words("english"))
_stemmer = PorterStemmer()


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = [_stemmer.stem(w) for w in text.split() if w not in _stop_words]
    return " ".join(tokens)


def _validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split rows into valid and invalid.
    Invalid = missing label/message, unrecognised label, or empty after cleaning.
    """
    invalid_mask = (
        df["label"].isna()
        | df["message"].isna()
        | ~df["label"].isin(["spam", "ham"])
        | (df["message"].str.strip() == "")
    )
    invalid = df[invalid_mask].copy()
    valid   = df[~invalid_mask].copy()

    if len(invalid):
        logger.warning(f"Validation: {len(invalid):,} invalid rows dropped")
        invalid.to_csv("logs/invalid_rows.csv", index=False)
    return valid, invalid


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, validate, deduplicate, and enrich the raw DataFrame."""
    logger.info("── TRANSFORM ────────────────────────────────────")

    # Normalise labels (handle mixed case, extra spaces)
    df["label"] = df["label"].str.strip().str.lower()

    # Validate
    df, invalid = _validate(df)
    logger.info(f"Valid rows after validation: {len(df):,}")

    # Deduplicate on (label, message)
    before = len(df)
    df.drop_duplicates(subset=["label", "message"], inplace=True)
    logger.info(f"Duplicates removed: {before - len(df):,} | Remaining: {len(df):,}")

    # Clean text
    df["cleaned_message"] = df["message"].apply(_clean_text)

    # Drop rows that are empty after cleaning
    empty_after = df["cleaned_message"].str.strip() == ""
    if empty_after.sum():
        logger.warning(f"Dropped {empty_after.sum()} rows that were empty after cleaning")
        df = df[~empty_after]

    # Enrich: binary label, word count, char count, row hash
    df["label_binary"]    = (df["label"] == "spam").astype(int)
    df["word_count"]      = df["message"].str.split().str.len()
    df["char_count"]      = df["message"].str.len()
    df["message_hash"]    = df["message"].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()
    )
    df["processed_at"]    = datetime.utcnow().isoformat()

    logger.info(
        f"Transform complete | spam: {df['label_binary'].sum():,} | "
        f"ham: {(df['label_binary'] == 0).sum():,}"
    )
    return df.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD — PostgreSQL
# ═════════════════════════════════════════════════════════════════════════════

def load_postgres(df: pd.DataFrame, conn_str: str | None = None) -> None:
    """
    Load transformed data into a PostgreSQL table.

    Set the connection string via the POSTGRES_CONN env var, or pass it directly:
        postgresql://user:password@host:5432/dbname
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        logger.error("sqlalchemy not installed. Run: pip install sqlalchemy psycopg2-binary")
        return

    conn_str = conn_str or os.getenv("POSTGRES_CONN")
    if not conn_str:
        logger.error(
            "No PostgreSQL connection string provided. "
            "Set the POSTGRES_CONN environment variable."
        )
        return

    logger.info("── LOAD → PostgreSQL ────────────────────────────")
    engine = create_engine(conn_str)

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sms_messages (
                id              SERIAL PRIMARY KEY,
                label           VARCHAR(4)   NOT NULL,
                label_binary    SMALLINT     NOT NULL,
                message         TEXT         NOT NULL,
                cleaned_message TEXT,
                word_count      INTEGER,
                char_count      INTEGER,
                message_hash    VARCHAR(32)  UNIQUE,
                source          VARCHAR(20),
                processed_at    TIMESTAMP
            )
        """))

    # Use upsert to avoid duplicate hash inserts on re-runs
    rows_before = pd.read_sql("SELECT COUNT(*) AS n FROM sms_messages", engine).iloc[0, 0]

    df_pg = df[[
        "label", "label_binary", "message", "cleaned_message",
        "word_count", "char_count", "message_hash", "source", "processed_at"
    ]]

    df_pg.to_sql(
        "sms_messages_staging", engine,
        if_exists="replace", index=False, method="multi", chunksize=1000
    )

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sms_messages
                (label, label_binary, message, cleaned_message,
                 word_count, char_count, message_hash, source, processed_at)
            SELECT label, label_binary, message, cleaned_message,
                   word_count, char_count, message_hash, source, processed_at::timestamp
            FROM   sms_messages_staging
            ON CONFLICT (message_hash) DO NOTHING
        """))
        conn.execute(text("DROP TABLE IF EXISTS sms_messages_staging"))

    rows_after = pd.read_sql("SELECT COUNT(*) AS n FROM sms_messages", engine).iloc[0, 0]
    logger.info(f"PostgreSQL: {rows_after - rows_before:,} new rows inserted | total: {rows_after:,}")


# ═════════════════════════════════════════════════════════════════════════════
# LOAD — BigQuery
# ═════════════════════════════════════════════════════════════════════════════

def load_bigquery(df: pd.DataFrame, project_id: str | None = None,
                  dataset: str = "smsguard", table: str = "sms_messages") -> None:
    """
    Load transformed data into BigQuery.

    Requirements:
        pip install google-cloud-bigquery pandas-gbq

    Auth:
        gcloud auth application-default login
        -- OR --
        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json

    Set project via GCP_PROJECT env var or pass project_id directly.
    """
    try:
        from google.cloud import bigquery
        import pandas_gbq
    except ImportError:
        logger.error(
            "BigQuery libraries not installed. "
            "Run: pip install google-cloud-bigquery pandas-gbq"
        )
        return

    project_id = project_id or os.getenv("GCP_PROJECT")
    if not project_id:
        logger.error(
            "No GCP project ID provided. Set the GCP_PROJECT environment variable."
        )
        return

    logger.info("── LOAD → BigQuery ──────────────────────────────")
    destination = f"{project_id}.{dataset}.{table}"

    bq_client = bigquery.Client(project=project_id)

    # Ensure dataset exists
    bq_client.create_dataset(bigquery.Dataset(f"{project_id}.{dataset}"), exists_ok=True)

    schema = [
        bigquery.SchemaField("label",           "STRING"),
        bigquery.SchemaField("label_binary",    "INTEGER"),
        bigquery.SchemaField("message",         "STRING"),
        bigquery.SchemaField("cleaned_message", "STRING"),
        bigquery.SchemaField("word_count",      "INTEGER"),
        bigquery.SchemaField("char_count",      "INTEGER"),
        bigquery.SchemaField("message_hash",    "STRING"),
        bigquery.SchemaField("source",          "STRING"),
        bigquery.SchemaField("processed_at",    "TIMESTAMP"),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",   # Replace on each run; change to WRITE_APPEND if preferred
    )

    df_bq = df[[
        "label", "label_binary", "message", "cleaned_message",
        "word_count", "char_count", "message_hash", "source", "processed_at"
    ]].copy()
    df_bq["processed_at"] = pd.to_datetime(df_bq["processed_at"])

    job = bq_client.load_table_from_dataframe(df_bq, destination, job_config=job_config)
    job.result()  # Wait for completion

    table_ref = bq_client.get_table(destination)
    logger.info(f"BigQuery: {table_ref.num_rows:,} rows in `{destination}`")


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATE
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    uci_path:   str = "data/raw/SMSSpamCollection",
    kaggle_path: str = "data/raw/spam.csv",
    target:     str = "both",          # "postgres" | "bigquery" | "both"
    save_csv:   bool = True,
) -> pd.DataFrame:
    """End-to-end ETL: extract → transform → load."""
    start = datetime.now()
    logger.info("══════════════════════════════════════════════════")
    logger.info("SMSGuard ETL Pipeline started")
    logger.info(f"Target: {target.upper()}")
    logger.info("══════════════════════════════════════════════════")

    # Extract
    raw_df = extract(uci_path, kaggle_path)

    # Transform
    clean_df = transform(raw_df)

    # Optionally save cleaned CSV (useful for Tableau / quick inspection)
    if save_csv:
        os.makedirs("data/processed", exist_ok=True)
        out_path = "data/processed/sms_cleaned.csv"
        clean_df.to_csv(out_path, index=False)
        logger.info(f"Cleaned data saved → {out_path}")

    # Load
    if target in ("postgres", "both"):
        load_postgres(clean_df)
    if target in ("bigquery", "both"):
        load_bigquery(clean_df)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Pipeline completed in {elapsed:.1f}s | {len(clean_df):,} rows processed")
    logger.info("══════════════════════════════════════════════════")
    return clean_df


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMSGuard ETL Pipeline")
    parser.add_argument(
        "--target",
        choices=["postgres", "bigquery", "both"],
        default="both",
        help="Where to load the data (default: both)",
    )
    parser.add_argument("--uci",    default="data/raw/SMSSpamCollection")
    parser.add_argument("--kaggle", default="data/raw/spam.csv")
    parser.add_argument("--no-csv", action="store_true", help="Skip saving cleaned CSV")
    args = parser.parse_args()

    run_pipeline(
        uci_path=args.uci,
        kaggle_path=args.kaggle,
        target=args.target,
        save_csv=not args.no_csv,
    )
