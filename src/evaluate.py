"""
evaluate.py

Evaluates the trained customer churn model.

Generates:

1. Confusion matrix
2. Feature importance plot
3. Classification report

and logs them as MLflow artifacts.
"""


from pathlib import Path
import logging


import pandas as pd

import mlflow
import mlflow.sklearn


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


from sklearn.ensemble import RandomForestClassifier


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


ARTIFACT_DIR = (
    PROJECT_ROOT /
    "artifacts"
)


ARTIFACT_DIR.mkdir(
    exist_ok=True
)



# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(
    exist_ok=True
)


logging.basicConfig(
    filename=LOG_DIR / "evaluate.log",
    level=logging.INFO,
    filemode="w",
    format="%(message)s",
    force=True
)


logging.getLogger().addHandler(
    logging.StreamHandler()
)



# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

logging.info(
    "\n========== LOADING DATA ==========\n"
)


df = pd.read_csv(
    DATA_PATH
)



# ---------------------------------------------------------
# Preprocess dataset
# ---------------------------------------------------------

df = preprocess_data(
    df
)
# ---------------------------------------------------------
# Split features and target
# ---------------------------------------------------------

X = df.drop(
    columns=[
        "Churn_Yes"
    ]
)


y = df[
    "Churn_Yes"
]



X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# ---------------------------------------------------------
# Train model
#
# NOTE:
# Later we will load the MLflow model instead.
# For now we reproduce the model.
# ---------------------------------------------------------

model = RandomForestClassifier(

    n_estimators=100,

    max_depth=10,

    random_state=42

)


model.fit(
    X_train,
    y_train
)



# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

predictions = model.predict(
    X_test
)



# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

logging.info(
    "\n========== CONFUSION MATRIX ==========\n"
)


cm = confusion_matrix(

    y_test,

    predictions

)


plt.figure(
    figsize=(6,5)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.title(
    "Customer Churn Confusion Matrix"
)


cm_path = (
    ARTIFACT_DIR /
    "confusion_matrix.png"
)


plt.savefig(
    cm_path,
    bbox_inches="tight"
)


plt.close()



logging.info(
    "Saved confusion matrix: %s",
    cm_path
)



# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

report = classification_report(
    y_test,
    predictions
)


report_path = (
    ARTIFACT_DIR /
    "classification_report.txt"
)


with open(
    report_path,
    "w"
) as file:

    file.write(
        report
    )


logging.info(
    "Saved classification report"
)



# ---------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------

importance = pd.DataFrame(

    {

        "feature": X.columns,

        "importance": model.feature_importances_

    }

)


importance = importance.sort_values(

    by="importance",

    ascending=False

)



plt.figure(
    figsize=(10,6)
)


sns.barplot(

    data=importance.head(15),

    x="importance",

    y="feature"

)


plt.title(
    "Top 15 Feature Importance"
)


importance_path = (
    ARTIFACT_DIR /
    "feature_importance.png"
)


plt.savefig(
    importance_path,
    bbox_inches="tight"
)


plt.close()



logging.info(
    "Saved feature importance"
)



# ---------------------------------------------------------
# MLflow Logging
# ---------------------------------------------------------

with mlflow.start_run(

    run_name="random_forest_evaluation"

):


    mlflow.log_artifact(
        cm_path
    )


    mlflow.log_artifact(
        report_path
    )


    mlflow.log_artifact(
        importance_path
    )


logging.info(
    "\nEvaluation completed successfully"
)