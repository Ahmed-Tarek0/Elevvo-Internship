# Sentiment Analysis on Amazon Product Reviews

A binary sentiment classifier (Positive / Negative) trained on the [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) dataset from Kaggle, using classic NLP preprocessing + TF-IDF + Logistic Regression / Naive Bayes.

## Dataset

- **Source:** Kaggle — `arhamrumi/amazon-product-reviews`
- **Size:** ~568K reviews
- **Columns used:** `Text` (review body), `Score` (1–5 star rating)
- **Label construction:** `Score > 3` → Positive (1), `Score < 3` → Negative (0). Neutral reviews (`Score == 3`) are dropped to keep the task strictly binary.

## Pipeline

1. **Download data** via `kagglehub`.
2. **Load & label** the dataset from `Score`.
3. **Clean text**: lowercasing, HTML/URL removal, punctuation & digit removal, stopword removal, short-token filtering.
4. **Train/test split**: 80/20, stratified on the label.
5. **Vectorize**: `TfidfVectorizer` (unigrams + bigrams, `max_features=20000`).
6. **Train classifiers**:
   - Logistic Regression (`solver='saga'`, `class_weight='balanced'`)
   - Multinomial Naive Bayes (with `sample_weight` computed via `compute_sample_weight(class_weight='balanced', ...)`, since `MultinomialNB` has no native `class_weight` support)
7. **Evaluate**: accuracy, precision/recall/F1 per class, confusion matrix.
8. **Bonus visualizations**: top-20 most frequent words per class (bar charts) and word clouds.

## Handling class imbalance

The raw dataset is heavily skewed toward positive reviews. Training naively on the full dataset without addressing this inflates overall accuracy while hiding poor recall on the negative class — the model just learns to favor the majority class. Applying `class_weight='balanced'` (Logistic Regression) and equivalent `sample_weight` (Naive Bayes) trades a small drop in overall accuracy for a much better, more honest recall on the negative class.

## Results

| Model | Accuracy | Negative Recall (before balancing) | Negative Recall (after balancing) |
|---|---|---|---|
| Logistic Regression | 92.5% | 0.74 | **0.92** |
| Naive Bayes | 89.5% | 0.47 | **0.88** |

Logistic Regression outperforms Naive Bayes on every metric, and both models improve substantially on the minority (negative) class once imbalance is corrected. Accuracy alone would have been a misleading metric here — per-class precision/recall tells the real story.


