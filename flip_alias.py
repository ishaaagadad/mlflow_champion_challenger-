"""Stretch goal: swap the champion and challenger aliases.

Run it once, then run predict_from_mlflow.py again WITHOUT editing it: the output changes.
Run flip_alias.py a second time afterwards to put the aliases back to the decision.
"""
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
NAME = "sentiment-model"
client = MlflowClient()

champion = client.get_model_version_by_alias(NAME, "champion").version
challenger = client.get_model_version_by_alias(NAME, "challenger").version
client.set_registered_model_alias(NAME, "champion", challenger)
client.set_registered_model_alias(NAME, "challenger", champion)
print(f"champion is now version {challenger}, challenger is now version {champion}")