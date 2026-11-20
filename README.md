# MLflow Champion-Challenger: Sentiment Classification

Student: Isha Gadad | USN: 1MSC25CI056

## 1. Task and models

- **Task:** Binary sentiment classification (`POSITIVE` / `NEGATIVE`) on individual sentences.
- **Dataset:** A fixed random subset of 200 sentences (seed 42) from the GLUE SST-2 validation split, stored in `data/eval_inputs.csv`.
- **Model A (existing Champion prior to evaluation):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters).
- **Model B (Challenger):** `textattack/bert-base-uncased-SST-2` (~110M parameters), which produces `LABEL_0` and `LABEL_1` that are converted to `NEGATIVE` and `POSITIVE` inside the custom MLflow wrapper (`hf_models.py`).
- **Execution setup:** Both models process the exact same sentences in identical order on CPU.

## 2. How to run

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

# terminal 1 (keep it open)
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

# terminal 2
python prepare_data.py             # builds data/eval_inputs.csv
python check_models.py             # sanity check of the label mapping
python evaluate.py                 # logs both models, decides the Champion
python register_champion.py        # registers the models, sets the aliases
python predict_from_mlflow.py      # loads models:/sentiment-model@champion, predicts data/new_data.csv
