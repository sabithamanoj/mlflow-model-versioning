"""
preprocess.py

Performs all preprocessing required
before model training.
"""

from pathlib import Path
import logging

import pandas as pd


def preprocess_data(df):
    """
    Clean and prepare dataset for training.
    """

    logging.info("\n========== PREPROCESSING ==========\n")

    # ---------------------------------------
    # Remove customerID
    # ---------------------------------------

    df = df.drop(columns=["customerID"])

    logging.info("customerID removed")

    # ---------------------------------------
    # Convert TotalCharges to numeric
    # ---------------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    logging.info("Converted TotalCharges to numeric")

    # ---------------------------------------
    # Missing values after conversion
    # ---------------------------------------

    logging.info("\nMissing values after conversion")

    logging.info(df.isnull().sum())

    # ---------------------------------------
    # Remove rows containing NaN
    # ---------------------------------------

    before = len(df)

    df = df.dropna()

    after = len(df)

    logging.info(
        "\nRemoved %d rows",
        before - after
    )

    # ---------------------------------------
    # One-hot encode
    # ---------------------------------------

    df = pd.get_dummies(
        df,
        drop_first=True
    )

    logging.info(
        "Dataset encoded"
    )

    logging.info(
        "Final dataset shape: %s",
        df.shape
    )

    return df