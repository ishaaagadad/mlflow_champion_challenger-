"""Runs both models on the same saved inputs and logs everything to MLflow.

Creates three runs in one experiment:
    A_sentiment_distilbert-sst2   (Champion)
    B_sentiment_bert-sst2         (Challenger)
    compare_sentiment             (summary run with the decision)
Start the tracking server first (see README).
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # draw plots without opening a window
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import pandas as pd
import sklearn
import torch
import transformers
from mlflow.models import infer_signature
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from hf_models import (BATCH_SIZE, MAX_LENGTH, MODELS, PACKAGE_WEIGHTS, TASK,
                       HFSentimentModel, build_pipeline, device_id)

# ----------------------------- settings -----------------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "hf-sentiment-champion-challenger"
STUDENT_USN = "1MSC25CI056"

# Champion rule (must match the README)
PRIMARY_METRIC = "macro_f1"        # higher is better
MARGIN = 0.01                      # Challenger must win by at least 1 percentage point
LATENCY_GUARDRAIL = 5.0            # reject a Challenger more than 5x slower than the Champion

LABELS = ["NEGATIVE", "POSITIVE"]
DATA_DIR, OUT_DIR, BUILD_DIR = Path("data"), Path("outputs"), Path("build")
OUT_DIR.mkdir(exist_ok=True)
BUILD_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

# the same evaluation set, in the same order, for both models
eval_df = pd.read_csv(DATA_DIR / "eval_inputs.csv")
meta = json.loads((DATA_DIR / "eval_meta.json").read_text(encoding="utf-8"))
device = "cuda" if device_id() == 0 else "cpu"
library_versions = (f"transformers={transformers.__version__}, torch={torch.__version__}, "
                    f"mlflow={mlflow.__version__}, scikit-learn={sklearn.__version__}")


def evaluate_model(key, cfg):
    """One run for one model: log the model, score it, save files and plots."""
    short = cfg["short_name"]
    run_name = f"{key}_sentiment_{short}"

    # what MLflow packages together with the Python wrapper
    build = BUILD_DIR / key
    build.mkdir(parents=True, exist_ok=True)
    config_path = build / "config.json"
    config_path.write_text(json.dumps(
        {"hf_model_id": cfg["hf_model_id"], "task": TASK, "label_map": cfg["label_map"]}, indent=2),
        encoding="utf-8")
    artifacts = {"config": str(config_path)}
    if PACKAGE_WEIGHTS:
        weights = build / "weights"
        build_pipeline(cfg["hf_model_id"]).save_pretrained(str(weights))
        artifacts["weights"] = str(weights)

    example = eval_df[["text"]].head(2)
    signature = infer_signature(example, pd.DataFrame(
        {"text": example["text"], "label": ["POSITIVE", "NEGATIVE"], "score": [0.99, 0.99]}))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "hf_model_id": cfg["hf_model_id"], "task": TASK, "role": cfg["role"],
            "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
            "eval_samples": len(eval_df), "seed": meta["seed"], "device": device,
        })
        mlflow.set_tags({"student_usn": STUDENT_USN, "library_versions": library_versions,
                         "dataset_name": meta["dataset_name"]})

        # log the model through MLflow, then evaluate the logged copy
        info = mlflow.pyfunc.log_model(
            name="sentiment_model", python_model=HFSentimentModel(), artifacts=artifacts,
            code_paths=["hf_models.py"], signature=signature, input_example=example,
            pip_requirements=["mlflow", "transformers", "torch"])
        model = mlflow.pyfunc.load_model(info.model_uri)

        model.predict(eval_df[["text"]].head(1))             # warm-up, not timed
        start = time.perf_counter()
        preds = model.predict(eval_df[["text"]])
        total_s = time.perf_counter() - start

        # metrics on the evaluation set
        y_true, y_pred = eval_df["label"].tolist(), preds["label"].tolist()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1,
            "macro_precision": precision, "macro_recall": recall,
            "avg_latency_ms": total_s / len(eval_df) * 1000, "total_inference_s": total_s,
        }
        mlflow.log_metrics(metrics)

        # artifacts: predictions, shared inputs, metrics file, confusion matrix
        pred_df = pd.DataFrame({
            "id": eval_df["id"], "text": eval_df["text"], "reference": y_true,
            "prediction": y_pred, "score": preds["score"],
            "correct": [t == p for t, p in zip(y_true, y_pred)],
        })
        pred_path = OUT_DIR / f"predictions_{short}.csv"
        pred_df.to_csv(pred_path, index=False)

        summary_path = OUT_DIR / f"metrics_summary_{short}.json"
        summary_path.write_text(json.dumps({
            "run_name": run_name, "hf_model_id": cfg["hf_model_id"],
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "direction": {"accuracy": "higher is better", "macro_f1": "higher is better",
                          "macro_precision": "higher is better", "macro_recall": "higher is better",
                          "avg_latency_ms": "lower is better", "total_inference_s": "lower is better"},
        }, indent=2), encoding="utf-8")

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=LABELS),
                               display_labels=LABELS).plot(ax=ax, colorbar=False)
        ax.set_title(f"{key}: {short}")
        fig.tight_layout()
        cm_path = OUT_DIR / f"confusion_matrix_{short}.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)

        for path in (pred_path, summary_path, cm_path, DATA_DIR / "eval_inputs.csv"):
            mlflow.log_artifact(str(path))

    print(f"{run_name}: accuracy={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}, "
          f"avg_latency={metrics['avg_latency_ms']:.1f} ms")
    return {"key": key, "short": short, "cfg": cfg, "run_id": run.info.run_id,
            "model_uri": info.model_uri, "metrics": metrics, "pred_df": pred_df}


def decide(delta, latency_ratio, a_ms, b_ms):
    """Apply the Champion rule. Returns (winner key, reason)."""
    if latency_ratio > LATENCY_GUARDRAIL:
        return "A", f"Challenger rejected: {latency_ratio:.1f}x slower than the Champion (limit {LATENCY_GUARDRAIL}x)."
    if delta >= MARGIN:
        return "B", f"Challenger wins: {PRIMARY_METRIC} is higher by {delta:.4f} (margin {MARGIN})."
    if delta <= -MARGIN:
        return "A", f"Champion stays: Challenger is lower by {abs(delta):.4f} (margin {MARGIN})."
    winner = "A" if a_ms <= b_ms else "B"
    return winner, (f"Difference {delta:+.4f} is within the margin of {MARGIN}; "
                    f"the faster model ({winner}) is chosen.")


results = {key: evaluate_model(key, cfg) for key, cfg in MODELS.items()}
A, B = results["A"], results["B"]
a_m, b_m = A["metrics"], B["metrics"]

delta = b_m[PRIMARY_METRIC] - a_m[PRIMARY_METRIC]            # Challenger minus Champion
latency_ratio = b_m["avg_latency_ms"] / a_m["avg_latency_ms"]
winner_key, reason = decide(delta, latency_ratio, a_m["avg_latency_ms"], b_m["avg_latency_ms"])
winner = results[winner_key]

# 5 (or more) cases where the models disagree; add your own comments in the README
disagree = pd.DataFrame({
    "id": eval_df["id"], "text": eval_df["text"], "reference": eval_df["label"],
    "pred_A": A["pred_df"]["prediction"], "pred_B": B["pred_df"]["prediction"],
})
disagree = disagree[disagree["pred_A"] != disagree["pred_B"]].copy()
disagree["correct_model"] = ["A" if a == ref else "B"
                             for a, ref in zip(disagree["pred_A"], disagree["reference"])]
disagree["comment"] = ""
disagree_path = OUT_DIR / "disagreements.csv"
disagree.to_csv(disagree_path, index=False)

# comparison chart for every metric
quality = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={"width_ratios": [3, 1, 1]})
x, w = range(len(quality)), 0.38
axes[0].bar([i - w / 2 for i in x], [a_m[q] for q in quality], w, color="tab:blue", label=f"A {A['short']}")
axes[0].bar([i + w / 2 for i in x], [b_m[q] for q in quality], w, color="tab:orange", label=f"B {B['short']}")
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(quality, rotation=15)
axes[0].set_ylim(0, 1.2)
axes[0].set_title("Quality (higher is better)")
axes[0].legend(loc="upper left", ncol=2)
for ax, metric, title in [(axes[1], "avg_latency_ms", "Avg latency, ms (lower is better)"),
                          (axes[2], "total_inference_s", "Total time, s (lower is better)")]:
    ax.bar(["A", "B"], [a_m[metric], b_m[metric]], color=["tab:blue", "tab:orange"])
    ax.set_title(title, fontsize=9)
fig.tight_layout()
chart_path = OUT_DIR / "comparison_chart.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)

# combined metrics file with the decision
summary = {
    "primary_metric": PRIMARY_METRIC, "margin": MARGIN, "latency_guardrail_x": LATENCY_GUARDRAIL,
    "models": {k: {"hf_model_id": r["cfg"]["hf_model_id"],
                   "metrics": {m: round(v, 4) for m, v in r["metrics"].items()}}
               for k, r in results.items()},
    "delta_primary_metric": round(delta, 4), "latency_ratio_B_over_A": round(latency_ratio, 2),
    "winner": winner_key, "reason": reason,
}
summary_path = OUT_DIR / "metrics_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

# summary run: the decision
with mlflow.start_run(run_name="compare_sentiment"):
    mlflow.log_params({
        "primary_metric": PRIMARY_METRIC, "margin": MARGIN, "latency_guardrail_x": LATENCY_GUARDRAIL,
        "direction": f"{PRIMARY_METRIC}: higher is better",
        "eval_samples": len(eval_df), "seed": meta["seed"],
        "hf_model_id_A": A["cfg"]["hf_model_id"], "hf_model_id_B": B["cfg"]["hf_model_id"],
        "model_uri_A": A["model_uri"], "model_uri_B": B["model_uri"],
        "run_id_A": A["run_id"], "run_id_B": B["run_id"],
    })
    mlflow.set_tags({
        "winner": f"{winner['key']}_sentiment_{winner['short']}", "winner_key": winner_key,
        "winner_hf_model_id": winner["cfg"]["hf_model_id"], "decision_reason": reason,
        "student_usn": STUDENT_USN, "library_versions": library_versions,
        "dataset_name": meta["dataset_name"],
    })
    mlflow.log_metrics({
        "delta_primary_metric": delta,
        "delta_accuracy": b_m["accuracy"] - a_m["accuracy"],
        "latency_ratio_B_over_A": latency_ratio,
    })
    for path in (summary_path, chart_path, disagree_path, DATA_DIR / "eval_inputs.csv",
                 Path("README.md"), Path("requirements.txt")):
        if path.exists():
            mlflow.log_artifact(str(path))

print("\nResults (Challenger minus Champion in the last column)")
table = pd.DataFrame({"A": a_m, "B": b_m})
table["B - A"] = table["B"] - table["A"]
print(table.round(4).to_string())
print(f"\nDecision: {reason}")
print(f"Winner: {winner['key']}_sentiment_{winner['short']} ({winner['cfg']['hf_model_id']})")
print(f"Models that disagree on {len(disagree)} sentences (see outputs/disagreements.csv)")