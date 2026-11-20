"""Registers both models and sets the aliases champion / challenger.

Reads the model URIs and the winner from the latest 'compare_sentiment' run,
so no run ID is typed by hand. Run evaluate.py first.
"""
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "hf-sentiment-champion-challenger"
REGISTERED_NAME = "sentiment-model"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT)
if experiment is None:
    raise SystemExit("Experiment not found. Run evaluate.py first.")

runs = client.search_runs([experiment.experiment_id],
                          order_by=["attributes.start_time DESC"], max_results=50)
summary = next((r for r in runs if r.info.run_name == "compare_sentiment"), None)
if summary is None:
    raise SystemExit("No 'compare_sentiment' run found. Run evaluate.py first.")

params, tags = summary.data.params, summary.data.tags
winner = tags["winner_key"]
other = "B" if winner == "A" else "A"

versions = {}
for key in ("A", "B"):
    mv = mlflow.register_model(params[f"model_uri_{key}"], REGISTERED_NAME)
    versions[key] = mv.version
    model_run = client.get_run(params[f"run_id_{key}"])
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "hf_model_id", params[f"hf_model_id_{key}"])
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "macro_f1",
                                 f"{model_run.data.metrics['macro_f1']:.4f}")

client.set_registered_model_alias(REGISTERED_NAME, "champion", versions[winner])
client.set_registered_model_alias(REGISTERED_NAME, "challenger", versions[other])

table = pd.DataFrame([
    {"version": versions[k], "alias": "champion" if k == winner else "challenger",
     "hf_model_id": params[f"hf_model_id_{k}"]} for k in ("A", "B")
]).sort_values("alias")
print(f"Registered model: {REGISTERED_NAME}")
print(table.to_string(index=False))
print("\nDecision:", tags["decision_reason"])