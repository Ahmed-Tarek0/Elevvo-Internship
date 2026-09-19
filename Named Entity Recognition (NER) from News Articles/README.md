# Named Entity Recognition (NER) from News Articles

Identifying named entities (people, organizations, locations) in news text using the [CoNLL-2003](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion) dataset, comparing a **rule-based** approach against a **pretrained model-based** approach (spaCy).

## Dataset

- **Source:** Kaggle — `alaakhaled/conll003-englishversion`
- **Files:** `train.txt`, `valid.txt`, `test.txt` in the standard CoNLL-2003 format — one token per line (`word POS-tag chunk-tag NER-tag`), sentences separated by blank lines, IOB2-tagged entities (`B-PER`, `I-PER`, `B-ORG`, `B-LOC`, `B-MISC`, `O`).
- Gold-standard entity labels are used to evaluate both approaches quantitatively, not just to inspect predictions by eye.

## Pipeline

1. **Download data** via `kagglehub`.
2. **Parse the CoNLL-2003 format** into per-sentence lists of `(token, tag)` pairs.
3. **Reconstruct raw text and gold entity spans**: join tokens back into plain sentences (tracking character offsets) and convert IOB2 tags into `(start, end, type)` entity spans.
4. **Model-based NER**: run spaCy's pretrained `en_core_web_sm` pipeline, mapping its labels to CoNLL's (`PERSON`→`PER`, `ORG`→`ORG`, `GPE`/`LOC`→`LOC`).
5. **Bonus**: visualize predicted entities inline with `displacy`.
6. **Rule-based NER**: a simple heuristic — runs of consecutive capitalized words (excluding the first word of a sentence) are flagged as candidate entities, with no type prediction.
7. **Evaluation**: precision, recall, and F1 for both approaches on **entity boundary detection**, run over the full CoNLL-2003 test set.
8. **Bonus**: comparing two spaCy models (`en_core_web_sm` vs. `en_core_web_md`), including entity-type accuracy.

## Why evaluate on PER / ORG / LOC only

CoNLL-2003 has a fourth entity type, `MISC`, which has no clean equivalent in spaCy's pretrained label set. `MISC` entities are excluded from both the gold spans and the evaluation so the comparison stays fair and meaningful.

## Results (full test set)

| Approach | Precision | Recall | F1 |
|---|---|---|---|
| Rule-based | 0.636 | 0.617 | 0.627 |
| `en_core_web_sm` | 0.846 | 0.677 | 0.752 |
| `en_core_web_md` | 0.834 | 0.738 | **0.783** |

| Model | Type Accuracy (on correctly-bounded entities) |
|---|---|
| `en_core_web_sm` | **0.815** |
| `en_core_web_md` | 0.794 |

The pretrained models clearly outperform the rule-based heuristic, especially in precision — the rule-based approach flags many false positives (e.g. capitalized words that aren't real entities). Between the two spaCy models, `md` achieves a better overall F1 thanks to higher recall (it catches more entities), while `sm` is slightly more accurate at typing the entities it does bound correctly — there's no single model that wins on every metric.
