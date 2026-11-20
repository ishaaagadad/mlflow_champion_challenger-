"""Step 4: build the shared evaluation set (data/eval_inputs.csv).

Takes a fixed random sample of the SST-2 validation set. Run this once.
"""
import json
from pathlib import Path

SEED = 42
N_SAMPLES = 200
DATA_DIR = Path("data")


def load_sst2_validation():
    from datasets import load_dataset
    try:
        ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
        name = "glue/sst2 validation"
    except Exception:                       # fallback if the first source is unavailable
        ds = load_dataset("stanfordnlp/sst2", split="validation")
        name = "stanfordnlp/sst2 validation"
    return ds.to_pandas(), name


df, dataset_name = load_sst2_validation()
df = df.rename(columns={"sentence": "text"})

sample = df.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)
sample["label"] = sample["label"].map({0: "NEGATIVE", 1: "POSITIVE"})
sample.insert(0, "id", range(len(sample)))

DATA_DIR.mkdir(exist_ok=True)
sample[["id", "text", "label"]].to_csv(DATA_DIR / "eval_inputs.csv", index=False)
(DATA_DIR / "eval_meta.json").write_text(
    json.dumps({"dataset_name": dataset_name, "seed": SEED, "n_samples": N_SAMPLES}, indent=2),
    encoding="utf-8",
)

print("Saved data/eval_inputs.csv with", len(sample), "rows from", dataset_name)
print(sample["label"].value_counts().to_string())