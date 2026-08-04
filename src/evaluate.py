"""
evaluate.py

Evaluates the registered MLflow model.

Workflow:

MLflow Model Registry
        |
        v
CustomerChurnModel Version 1
        |
        v
Load Test Data
        |
        v
Preprocess
        |
        v
Prediction
        |
        v
Evaluation Artifacts
        |
        v
Log artifacts to MLflow
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


LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(
    exist_ok=True
)



# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

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
# MLflow Model Registry Configuration
# ---------------------------------------------------------

MODEL_NAME = "CustomerChurnModel"

MODEL_VERSION = 1

'''
MODEL_URI = (
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)
'''
MODEL_URI = "models:/CustomerChurnModel@champion"



# ---------------------------------------------------------
# Load registered model
# ---------------------------------------------------------

logging.info(
    "\n========== LOADING REGISTERED MODEL ==========\n"
)


logging.info(
    "Loading model: %s",
    MODEL_URI
)


model = mlflow.sklearn.load_model(
    MODEL_URI
)


logging.info(
    "Model loaded successfully"
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
# Preprocessing
# ---------------------------------------------------------

logging.info(
    "\n========== PREPROCESSING ==========\n"
)


df = preprocess_data(
    df
)



# ---------------------------------------------------------
# Prepare features and target
# ---------------------------------------------------------

X = df.drop(
    columns=[
        "Churn_Yes"
    ]
)


y = df[
    "Churn_Yes"
]



# ---------------------------------------------------------
# Create test data
#
# Same split used during training
# ---------------------------------------------------------

_, X_test, _, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



logging.info(
    "Test samples: %d",
    len(X_test)
)



# ---------------------------------------------------------
# Prediction
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



# ---------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------

if hasattr(
    model,
    "feature_importances_"
):

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



else:

    importance_path = None



# ---------------------------------------------------------
# Log artifacts to MLflow
# ---------------------------------------------------------

with mlflow.start_run(

    run_name="RandomForest_Model_Evaluation"

):


    mlflow.set_tag(

        "Evaluation_Model",

        MODEL_URI

    )


    mlflow.log_artifact(
        cm_path
    )


    mlflow.log_artifact(
        report_path
    )


    if importance_path:

        mlflow.log_artifact(
            importance_path
        )



logging.info(
    "\n========== EVALUATION COMPLETED ==========\n"
)