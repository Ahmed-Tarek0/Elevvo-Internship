# Fake News Detection

A binary text classifier that labels a news article as **Fake** or **Real**, using the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.

## Dataset

- **Source:** Kaggle — `clmentbisaillon/fake-and-real-news-dataset`
- **Files:** `Fake.csv` (23,502 articles) and `True.csv` (21,417 articles)
- **Columns used:** `title` and `text` (combined into one field). `subject` and `date` are excluded — see below.

## Handling dataset shortcuts (leakage)

An exploratory check on the raw data revealed two strong non-textual shortcuts that would let a model reach high accuracy without actually reading the content:

- **`subject`** uses a completely disjoint set of values between `Fake.csv` and `True.csv` — a near-perfect label leak on its own.
- The **`(Reuters)`** byline appears in almost every real article and essentially never in fake ones — an easy giveaway token.

To keep the task meaningful, `subject` and `date` are dropped entirely, and the `(Reuters)` byline is stripped out during text cleaning, so the model has to learn from genuine writing-style patterns instead.

## Pipeline

1. **Download data** via `kagglehub`.
2. **Load & label** `Fake.csv` (0) and `True.csv` (1), then shuffle and combine into one dataframe.
3. **Text preprocessing**: lowercasing, byline/HTML/URL removal, punctuation & digit removal, tokenization (`nltk.word_tokenize`), stopword removal, lemmatization.
4. **Train/test split**: 80/20, stratified on the label.
5. **Vectorize**: `TfidfVectorizer` (unigrams + bigrams, `sublinear_tf=True`, `max_features=50000`, `min_df=2`).
6. **Train the classifier**: `LinearSVC`, tuned via `GridSearchCV` over `C`.
7. **Evaluate**: accuracy, F1-score, classification report, confusion matrix, and a train-vs-test accuracy check for overfitting.
8. **Bonus**: word clouds comparing the most common terms in fake vs. real articles.

## Why Linear SVM

The task suggests either Logistic Regression or SVM. `LinearSVC` is used here since it's typically the stronger choice on sparse, high-dimensional TF-IDF text features — which fits this dataset well.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 99.53% |
| Test F1-score | 99.51% |
| Train Accuracy | 100% |
| Train–Test Gap | 0.47% |

The small train–test gap indicates the model is genuinely generalizing rather than overfitting. High performance here is expected even after removing the obvious leaks (`subject`, `(Reuters)`), since fake and real articles in this dataset come from distinctly different writing sources with a consistently different style.
