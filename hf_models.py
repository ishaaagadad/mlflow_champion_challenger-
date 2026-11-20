"""Shared definitions: the two Hugging Face models and the MLflow pyfunc wrapper.

Only pandas and mlflow are imported at the top. transformers and torch are imported
inside functions, so a script that loads the logged model through MLflow never has to
call the Hugging Face pipeline itself.
"""
import json

import pandas as pd
import mlflow.pyfunc

TASK = "sentiment-analysis"
MAX_LENGTH = 128          # longer sentences are cut to this many tokens
BATCH_SIZE = 16           # sentences processed together
PACKAGE_WEIGHTS = False   # True: copy the model files into MLflow (about 1 GB more disk)
                          # False: MLflow stores only the Hugging Face model ID

# Model A is the current Champion, Model B is the Challenger.
MODELS = {
    "A": {
        "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "short_name": "distilbert-sst2",
        "role": "champion",
        "label_map": {"NEGATIVE": "NEGATIVE", "POSITIVE": "POSITIVE"},
    },
    "B": {
        "hf_model_id": "textattack/bert-base-uncased-SST-2",
        "short_name": "bert-sst2",
        "role": "challenger candidate",
        # this model answers LABEL_0 / LABEL_1, so the labels are mapped
        "label_map": {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"},
    },
}


def device_id():
    """0 = first GPU, -1 = CPU."""
    import torch
    return 0 if torch.cuda.is_available() else -1


def build_pipeline(model_path_or_id):
    """Create the Hugging Face sentiment pipeline (downloads the model the first time)."""
    from transformers import pipeline
    return pipeline(TASK, model=model_path_or_id, device=device_id())


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow model. Input: DataFrame with a 'text' column.
    Output: DataFrame with the columns text, label and score."""

    def load_context(self, context):
        with open(context.artifacts["config"], encoding="utf-8") as f:
            self.config = json.load(f)
        # use the packaged weights if they exist, otherwise the Hugging Face model ID
        source = context.artifacts.get("weights", self.config["hf_model_id"])
        self.pipe = build_pipeline(source)

    def predict(self, context, model_input, params=None):
        texts = pd.DataFrame(model_input)["text"].astype(str).tolist()
        results = self.pipe(texts, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LENGTH)
        label_map = self.config["label_map"]
        return pd.DataFrame({
            "text": texts,
            "label": [label_map.get(r["label"], r["label"]) for r in results],
            "score": [round(float(r["score"]), 4) for r in results],
        })