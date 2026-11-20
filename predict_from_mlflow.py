"""Loads the Champion by ALIAS from the MLflow Model Registry and predicts on new data.

There is no Hugging Face code and no version number in this script.
Optional: python predict_from_mlflow.py challenger
"""
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_NAME = "sentiment-model"
ALIAS = sys.argv[1] if len(sys.argv) > 1 else "champion"

mlflow.set_tracking_uri(TRACKING_URI)
model_uri = f"models:/{REGISTERED_NAME}@{ALIAS}"
model = mlflow.pyfunc.load_model(model_uri)

# which version and Hugging Face model sit behind the alias right now
version = MlflowClient().get_model_version_by_alias(REGISTERED_NAME, ALIAS)
hf_model_id = version.tags.get("hf_model_id", "unknown")
print(f"Loaded {model_uri}")
print(f"  model version : {version.version}")
print(f"  Hugging Face  : {hf_model_id}")

new_data = pd.read_csv("data/new_data.csv")
predictions = model.predict(new_data[["text"]])
predictions["alias"] = ALIAS
predictions["model_version"] = version.version
predictions["hf_model_id"] = hf_model_id

Path("outputs").mkdir(exist_ok=True)
predictions.to_csv("outputs/new_data_predictions.csv", index=False)
print("\nPredictions on new data:")
print(predictions[["text", "label", "score"]].to_string(index=False))
print("\nSaved outputs/new_data_predictions.csv")