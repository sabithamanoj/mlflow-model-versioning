"""
explore_data.py

Performs basic dataset exploration.

Logs:
- Dataset shape
- Column names
- Data types
- Missing values
- Target distribution
"""

import logging

import pandas as pd

from pathlib import Path

# ---------------------------------------------------------
# Configure logging
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "explore_data.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="%(message)s",
)

logging.getLogger().addHandler(
    logging.StreamHandler()
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"

df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# Dataset information
# ---------------------------------------------------------

logging.info("\n========== DATASET SHAPE ==========\n")

logging.info(df.shape)


logging.info("\n========== COLUMN NAMES ==========\n")

for column in df.columns:
    logging.info(column)


logging.info("\n========== FIRST FIVE ROWS ==========\n")

logging.info("\n%s", df.head().to_string())


logging.info("\n========== DATA TYPES ==========\n")

logging.info("\n%s", df.dtypes)


logging.info("\n========== MISSING VALUES ==========\n")

logging.info("\n%s", df.isnull().sum())


logging.info("\n========== TARGET DISTRIBUTION ==========\n")

logging.info("\n%s", df["Churn"].value_counts())

logging.info(
    "\n========== BLANK TOTALCHARGES ==========\n"
)

blank_totalcharges = (
    df["TotalCharges"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

logging.info(
    "Blank TotalCharges values: %d",
    blank_totalcharges
)