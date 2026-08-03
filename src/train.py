"""
train.py

Trains a Customer Churn prediction model
and tracks the experiment using MLflow.

Workflow:

Customer Churn Dataset
        |
        v
Load Dataset
        |
        v
Preprocessing
        |
        v
Train/Test Split
        |
        v
Random Forest Training
        |
        v
Model Evaluation
        |
        v
Log Parameters, Metrics and Model to MLflow
"""


from pathlib import Path
import logging

import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from preprocess import preprocess_data


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


DATA_PATH = (
    PROJECT_ROOT /
    "data" /
    "customer_churn.csv"
)


LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(
    exist_ok=True
)


LOG_FILE = (
    LOG_DIR /
    "train.log"
)


# ---------------------------------------------------------
# Configure logging
# ---------------------------------------------------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="%(message)s",
    force=True
)


# Also show logs in terminal
logging.getLogger().addHandler(
    logging.StreamHandler()
)



# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

logging.info(
    "\n========== LOADING DATASET ==========\n"
)


df = pd.read_csv(
    DATA_PATH
)


logging.info(
    "Original dataset shape: %s",
    df.shape
)



# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

logging.info(
    "\n========== PREPROCESSING ==========\n"
)


df = preprocess_data(
    df
)


logging.info(
    "Processed dataset shape: %s",
    df.shape
)



# ---------------------------------------------------------
# Separate Features and Target
# ---------------------------------------------------------

logging.info(
    "\n========== FEATURES / TARGET ==========\n"
)


X = df.drop(
    columns=[
        "Churn_Yes"
    ]
)


y = df[
    "Churn_Yes"
]


logging.info(
    "Number of features: %d",
    X.shape[1]
)



# ---------------------------------------------------------
# Train Test Split
# ---------------------------------------------------------

logging.info(
    "\n========== TRAIN TEST SPLIT ==========\n"
)


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


logging.info(
    "Training samples: %d",
    len(X_train)
)


logging.info(
    "Testing samples: %d",
    len(X_test)
)



# ---------------------------------------------------------
# Configure MLflow Experiment
# ---------------------------------------------------------

mlflow.set_experiment(
    "Customer_Churn_Experiment"
)



# ---------------------------------------------------------
# Start MLflow Run
# ---------------------------------------------------------

with mlflow.start_run():


    logging.info(
        "\n========== MODEL TRAINING ==========\n"
    )


    # -----------------------------------------------------
    # Create Model
    # -----------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=100,

        max_depth=10,

        random_state=42

    )


    # -----------------------------------------------------
    # Train Model
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    logging.info(
        "Model training completed"
    )



    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    predictions = model.predict(
        X_test
    )



    # -----------------------------------------------------
    # Calculate Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions
    )


    recall = recall_score(
        y_test,
        predictions
    )


    f1 = f1_score(
        y_test,
        predictions
    )


    logging.info(
        "\n========== MODEL METRICS ==========\n"
    )


    logging.info(
        "Accuracy : %.4f",
        accuracy
    )


    logging.info(
        "Precision: %.4f",
        precision
    )


    logging.info(
        "Recall   : %.4f",
        recall
    )


    logging.info(
        "F1 Score : %.4f",
        f1
    )



    # -----------------------------------------------------
    # Log Parameters to MLflow
    # -----------------------------------------------------

    mlflow.log_param(
        "model",
        "RandomForestClassifier"
    )


    mlflow.log_param(
        "n_estimators",
        100
    )


    mlflow.log_param(
        "max_depth",
        10
    )



    # -----------------------------------------------------
    # Log Metrics to MLflow
    # -----------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )


    mlflow.log_metric(
        "precision",
        precision
    )


    mlflow.log_metric(
        "recall",
        recall
    )


    mlflow.log_metric(
        "f1_score",
        f1
    )



    # -----------------------------------------------------
    # Log Model to MLflow
    # -----------------------------------------------------

    mlflow.sklearn.log_model(
        model,
        "customer_churn_model"
    )


    logging.info(
        "\nModel logged to MLflow successfully"
    )



logging.info(
    "\n========== TRAINING COMPLETED ==========\n"
)