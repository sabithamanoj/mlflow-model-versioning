"""
train_models.py

Multi-model experimentation using MLflow.

Models compared:

1. Random Forest
2. Logistic Regression
3. Gradient Boosting


Each model:

- gets its own MLflow run
- logs parameters
- logs metrics
- logs model artifact
- registers model version
"""


from pathlib import Path
import logging


import pandas as pd


import mlflow
import mlflow.sklearn


from sklearn.model_selection import train_test_split


from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)


from sklearn.linear_model import LogisticRegression


from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score

)


from preprocess import preprocess_data



# ---------------------------------------------------------
# Paths
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



# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(

    filename=LOG_DIR / "train_models.log",

    level=logging.INFO,

    filemode="w",

    format="%(message)s",

    force=True

)


logging.getLogger().addHandler(
    logging.StreamHandler()
)



# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

logging.info(
    "\n========== LOADING DATASET ==========\n"
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
# Features and target
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
# Train test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



logging.info(
    "Training samples: %s",
    len(X_train)
)


logging.info(
    "Testing samples: %s",
    len(X_test)
)



# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

models = {


    "RandomForest_Baseline":

        RandomForestClassifier(

            n_estimators=100,

            max_depth=10,

            random_state=42

        ),



    "LogisticRegression_Baseline":

        LogisticRegression(

            max_iter=1000,

            random_state=42

        ),



    "GradientBoosting_Baseline":

        GradientBoostingClassifier(

            n_estimators=100,

            learning_rate=0.1,

            random_state=42

        )

}



# ---------------------------------------------------------
# MLflow Experiment
# ---------------------------------------------------------

mlflow.set_experiment(
    "Customer_Churn_Experiment"
)



# ---------------------------------------------------------
# Train each model
# ---------------------------------------------------------

for model_name, model in models.items():


    logging.info(
        "\n================================"
    )


    logging.info(
        "Training %s",
        model_name
    )


    logging.info(
        "================================\n"
    )



    with mlflow.start_run(

        run_name=model_name

    ):



        # -------------------------------
        # Training
        # -------------------------------

        model.fit(

            X_train,

            y_train

        )



        predictions = model.predict(

            X_test

        )


        probabilities = model.predict_proba(

            X_test

        )[:,1]



        # -------------------------------
        # Metrics
        # -------------------------------

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


        roc_auc = roc_auc_score(

            y_test,

            probabilities

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


        logging.info(
            "ROC AUC  : %.4f",
            roc_auc
        )



        # -------------------------------
        # MLflow Logging
        # -------------------------------

        mlflow.set_tag(

            "Model",

            model_name

        )


        mlflow.set_tag(

            "Dataset",

            "Telco Customer Churn"

        )



        mlflow.log_params(

            model.get_params()

        )



        mlflow.log_metrics(

            {

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1_score": f1,

                "roc_auc": roc_auc

            }

        )



        mlflow.sklearn.log_model(

            model,

            name="model"

        )



    logging.info(

        "%s completed",

        model_name

    )



logging.info(

    "\n========== ALL MODELS COMPLETED ==========\n"

)