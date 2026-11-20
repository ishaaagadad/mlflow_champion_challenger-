# MLflow Champion-Challenger: Sentiment Classification

Student: Isha Gadad | USN: 1MSC25CI056

## 1. Task and models
- **Task:** binary sentiment classification (POSITIVE / NEGATIVE) of single sentences.
- **Dataset:** SST-2 validation set (GLUE), a fixed random sample of **200 sentences** (seed 42), saved in `data/eval_inputs.csv`.
- **Model A (current Champion before comparison):** `distilbert-base-uncased-finetuned-sst-2-english` (about 67M parameters).
- **Model B (Challenger):** `textattack/bert-base-uncased-SST-2` (about 110M parameters). It outputs `LABEL_0` / `LABEL_1`, which are mapped to NEGATIVE / POSITIVE inside the MLflow wrapper (`hf_models.py`).
- Both models run on the same inputs, in the same order, on CPU.

## 2. How to run
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

# terminal 1 (keep it open)
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

# terminal 2
python prepare_data.py          # builds data/eval_inputs.csv
python check_models.py          # sanity check of the label mapping
python evaluate.py              # logs both models, decides the Champion
python register_champion.py     # registers the models, sets the aliases
python predict_from_mlflow.py   # loads models:/sentiment-model@champion, predicts data/new_data.csv
```
MLflow UI: http://127.0.0.1:5000

## 3. Champion rule (written before the final run)
- **Primary metric:** macro-F1 on the evaluation set. **Higher is better.** (Accuracy, macro-precision and macro-recall: higher is better. `avg_latency_ms` and `total_inference_s`: lower is better.)
- **Minimum improvement margin:** the Challenger must beat the Champion by at least **0.01 (1 percentage point)** in macro-F1. 
This is because with 200 sentences, one sentence is 0.5 percentage points. Hence, a smaller difference could be insignificant.
- **Tie-breaker:** if the difference is within the margin (between -0.01 and +0.01), the **faster model** (lower average latency) is chosen.
- **Guardrail:** a Challenger whose average latency is more than **5x** the Champion's is rejected.
- If the Challenger is lower by more than the margin, the Champion stays.

## 4. Results (Logged after evaluation was done)
| Metric | A: distilbert-sst2 | B: bert-sst2 | B - A | Better |
|---|---|---|---|---|
| accuracy | 0.9350 | 0.9450 | +0.0100 | higher |
| macro_f1 (primary) | 0.9343 | 0.9446 | +0.0103 | higher |
| macro_precision | 0.9359 | 0.9442 | +0.0083 | higher |
| macro_recall | 0.9331 | 0.9450 | +0.0119 | higher |
| avg_latency_ms | 29.58 | 57.72 | +28.14 | lower |
| total_inference_s | 5.92 | 11.54 | +5.63 | lower |

**Outcome of the rule:** the Challenger (B) improves macro-F1 by 0.0103, which is just above the 0.01 margin. The latency ratio is 1.95x, below the 5x guardrail. By the rule, **Model B wins and becomes the Champion**, and Model A becomes the Challenger in the registry.


**Examples where the models disagree** (8 of 200 sentences, from `outputs/disagreements.csv`):

| ID | Sentence | Reference | A | B | Correct | Comment |
|---|---|---|---|---|---|---|
| 4 | it moves quickly , adroitly , and without fuss ; it does n't give you time to reflect on the inanity -- and the cold war datedness -- of its premise . | POSITIVE | POSITIVE | NEGATIVE | A | Praise is mixed with criticism of the premise; B was pulled by negative words like "inanity". |
| 44 | it has all the excitement of eating oatmeal . | NEGATIVE | NEGATIVE | POSITIVE | A | Sarcastic comparison: "excitement" sounds positive but the meaning is negative. A understood it, B did not. |
| 56 | oh come on . | NEGATIVE | POSITIVE | NEGATIVE | B | Very short, with no sentiment words, so it is hard to judge. B got it right, A guessed wrong. |
| 94 | a bloated gasbag thesis grotesquely impressed by its own gargantuan aura of self-importance ... | NEGATIVE | POSITIVE | NEGATIVE | B | Negative tone is carried by unusual words ("bloated", "self-importance"). B caught it, A missed it. |
| 112 | writer/director joe carnahan 's grimy crime drama is a manual of precinct cliches , but it moves fast enough to cover its clunky dialogue and lapses in logic . | POSITIVE | NEGATIVE | POSITIVE | B | Mixed review whose final verdict is positive. B followed the overall verdict, A reacted to "cliches" and "clunky". |
| 116 | sustains its dreamlike glide through a succession of cheesy coincidences and voluptuous cheap effects , not the least of which is rebecca romijn-stamos . | NEGATIVE | POSITIVE | NEGATIVE | B | Positive-sounding words ("dreamlike", "glide") are used ironically; "cheesy" and "cheap" show the real meaning. |
| 161 | outer-space buffs might love this film , but others will find its pleasures intermittent . | NEGATIVE | POSITIVE | NEGATIVE | B | Hedged negative: "might love" is followed by "but others", so the overall view is negative. |
| 183 | not since tom cruise in risky business has an actor made such a strong impression in his underwear . | POSITIVE | POSITIVE | NEGATIVE | A | Indirect praise; the word "not" probably misled B. |

## 5. Decision (150-250 words)
Model B (textattack/bert-base-uncased-SST-2) becomes the Champion and replaces Model A (distilbert-base-uncased-finetuned-sst-2-english). Under the rule written before the run, the Challenger had to beat the Champion's macro-F1 by at least 0.01. B reached 0.9446 against 0.9343 for A, a difference of 0.0103, so the margin was cleared by a very small amount. The latency guardrail was also passed, since B is 1.95 times slower, well under the 5x limit.

However, this improvement is not strong evidence. With 200 sentences, the accuracy gap of 0.010 is only two sentences (189 correct against 187). The models disagreed on just 8 sentences, where B was right on 5 and A on 3, so a different sample could easily reverse the result. The decision follows the rule, but it should be read as a weak preference for B and not a proven gain.

The trade-offs are clear. B needs about 57.7 ms per sentence against 29.6 ms for A, and it is larger (about 110M against 67M parameters). If speed matters more than a gain of about one point, A would be the practical choice. Most disagreements involved sarcastic, mixed or very short sentences, which both models find difficult. A larger evaluation set or repeated samples would be needed to confirm the difference.

## 6. Limitations and declaration
- **Sample size:** 200 sentences from one dataset; both models were fine-tuned on SST-2 training data, so the scores are probably higher than on other text.
- **Tools used:** I took reference from the work done during the lab to further my understanding of the topic and an AI assistant was used for basic structure and for helping in writing scripts. I can explain all codes and I understand every line.
