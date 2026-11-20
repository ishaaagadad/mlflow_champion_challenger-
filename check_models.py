"""Step 5: quick sanity check of both models and of the label mapping.

Run it before evaluate.py. The mapped labels must match the meaning of each sentence.
"""
from hf_models import BATCH_SIZE, MAX_LENGTH, MODELS, build_pipeline

SENTENCES = [
    "I loved this film, it was wonderful.",
    "A boring, terrible waste of time.",
    "What a brilliant and moving story.",
    "The acting was awful and the plot made no sense.",
]

for key, cfg in MODELS.items():
    print(f"\nModel {key}: {cfg['hf_model_id']}")
    pipe = build_pipeline(cfg["hf_model_id"])
    results = pipe(SENTENCES, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LENGTH)
    for text, r in zip(SENTENCES, results):
        mapped = cfg["label_map"].get(r["label"], r["label"])
        print(f"  raw={r['label']:<9} mapped={mapped:<8} score={r['score']:.3f}  | {text}")