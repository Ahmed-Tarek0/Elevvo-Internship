# Question Answering with Transformers

An **extractive question-answering** system built on pretrained transformer models. Given a passage (context) and a question, the model extracts the exact answer as a span of text from within the passage — it does not generate free-form text, it points to where the answer already lives in the source.

## Overview

- **Dataset:** [SQuAD v1.1](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset) (Stanford Question Answering Dataset) — 100,000+ question–answer pairs over 500+ Wikipedia articles.
- **Approach:** No training from scratch. Pretrained transformer models fine-tuned for SQuAD-style QA are loaded directly via Hugging Face's `pipeline("question-answering")`.
- **Deliverables:**
  - `QA_with_Transformers.ipynb` — the full notebook (data loading, evaluation, model comparison).
  - `qa_app.py` — a standalone Streamlit app for interactively asking questions.
  - `requirements.txt` — dependencies for running/deploying the app.

## Dataset

SQuAD ships as two JSON files:

| File | Purpose |
|---|---|
| `train-v1.1.json` | Training split (not used here — no fine-tuning is performed) |
| `dev-v1.1.json` | Evaluation split — used to test the pretrained models |

Each entry nests `article → paragraph → context`, with each `context` paired with several `questions`, each question having a gold `answer` (text + its character position in the context). The notebook flattens this structure into a simple table of `(context, question, answer_text, answer_start)` rows before doing anything else with it.

## Pipeline

1. **Download** the dataset via `kagglehub`.
2. **Parse & flatten** the SQuAD JSON format into a plain table.
3. **Load a pretrained QA pipeline** — `distilbert-base-cased-distilled-squad` — and run it on a few example questions to sanity-check it.
4. **Implement the official SQuAD metrics** from scratch:
   - **Exact Match (EM)** — 1 if the predicted answer matches the gold answer exactly after normalization (lowercasing, removing punctuation/articles/extra whitespace), else 0.
   - **F1** — token-level overlap between predicted and gold answers, more forgiving of partial matches.
5. **Evaluate** the pretrained model on a sample of the dev set.
6. **Bonus — compare base models:** run the same evaluation with `bert-large-uncased-whole-word-masking-finetuned-squad` and `deepset/roberta-base-squad2`.
7. **Bonus — interactive interface:** a simple `ask(context, question)` helper in the notebook, plus a full Streamlit app (`qa_app.py`) for a proper UI.

## Results

Evaluated on a 300-question sample of the SQuAD v1.1 dev set:

| Model | Exact Match | F1 |
|---|---|---|
| DistilBERT (`distilbert-base-cased-distilled-squad`) | 64.00 | 79.05 |
| **BERT (large)** (`bert-large-uncased-whole-word-masking-finetuned-squad`) | **72.33** | **85.85** |
| RoBERTa (`deepset/roberta-base-squad2`) | 69.00 | 84.09 |

**Takeaway:** BERT-large performs best, which tracks with its much larger parameter count. RoBERTa (base-sized, but with an improved pretraining recipe) lands in between. DistilBERT trails both — expected, since it's a distilled, compressed model that trades some accuracy for speed and a smaller footprint. There's a clear accuracy-vs-efficiency trade-off across the three, not a single "best" choice for every use case.

## Try it live

🔗 **[Live demo](https://question-answering0.streamlit.app/)**

## Running the Streamlit app locally

```bash
pip install -r requirements.txt
streamlit run qa_app.py
```

`requirements.txt` pins `transformers==4.46.3` — the latest release (5.x) has a bug where the `"question-answering"` pipeline task is missing from its task registry, causing a `KeyError: "Unknown task question-answering"`. This is a bug in the library itself, not in this project's code, and pinning to a known-stable version avoids it.

The app exposes two inputs:
- **Context / Passage** — the text to search for an answer in.
- **Question** — anything you want to ask about that passage.

The first run downloads the DistilBERT model (~250 MB), so it may take a minute before the first answer appears.
