"""
promote_model.py

Manages MLflow model promotion.

Version workflow:

Version 1
    |
    v
candidate
    |
    v
champion
"""


import mlflow
from mlflow import MlflowClient
import logging



# ---------------------------------------------------------
# MLflow Configuration
# ---------------------------------------------------------

MODEL_NAME = "CustomerChurnModel"

MODEL_VERSION = 1



# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)



client = MlflowClient()



# ---------------------------------------------------------
# Assign candidate alias
# ---------------------------------------------------------

logging.info(
    "Assigning candidate alias..."
)


client.set_registered_model_alias(

    name=MODEL_NAME,

    alias="candidate",

    version=MODEL_VERSION

)


logging.info(
    "Model Version %s promoted to candidate",
    MODEL_VERSION
)



# ---------------------------------------------------------
# Promote to champion
# ---------------------------------------------------------

logging.info(
    "Promoting model to champion..."
)


client.set_registered_model_alias(

    name=MODEL_NAME,

    alias="champion",

    version=MODEL_VERSION

)


logging.info(
    "Model Version %s is now champion",
    MODEL_VERSION
)