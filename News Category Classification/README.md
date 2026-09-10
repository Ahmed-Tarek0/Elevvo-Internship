# News Category Classification (AG News)

A multiclass text classifier that categorizes news articles into **World, Sports, Business, Sci/Tech**, using the [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) from Kaggle.

## Dataset

- **Source:** Kaggle — `amananandrai/ag-news-classification-dataset`
- **Size:** 120,000 training articles / 7,600 test articles (perfectly balanced, 30,000 / 1,900 per class)
- **Columns used:** `Title` and `Description` (combined into a single `text` field), `Class Index` (1–4, mapped to World / Sports / Business / Sci/Tech)

## Pipeline

1. **Download data** via `kagglehub`.
2. **Load & label** `train.csv` / `test.csv`, mapping `Class Index` to category names.
3. **Text preprocessing**: lowercasing, punctuation/digit removal, tokenization (`nltk.word_tokenize`), stopword removal, lemmatization.
4. **Train/validation split**: 90/10 from the training set (stratified), with `test.csv` kept fully held out as the final test set.
5. **Vectorize**: `TfidfVectorizer` (unigrams + bigrams, `sublinear_tf=True`, `max_features=40000`).
6. **Train the main classifier**: `LinearSVC`, tuned via `GridSearchCV` over `C`.
7. **Compare** against Logistic Regression and Random Forest on the same test set.
8. **Bonus visualizations**: top words per category (bar charts + word clouds).
9. **Bonus neural network**: a simple feedforward network in Keras — `Embedding → GlobalAveragePooling1D → Dense → Dropout → Dense → Dropout → Dense(softmax)` — trained on tokenized/padded sequences (not on dense TF-IDF, to keep memory usage reasonable).

## Why Linear SVM as the main model

The dataset is clean, perfectly balanced, and high-dimensional after TF-IDF vectorization — the setting where linear SVMs typically outperform other classic models on sparse text features, while Random Forest tends to struggle with that same sparsity/dimensionality.

## Results

| Model | Test Accuracy |
|---|---|
| **Linear SVM** | **91.99%** |
| Logistic Regression | 91.53% |
| Neural Network (Keras) | 91.43% |
| Random Forest | 89.75% |

Linear SVM performs best overall, with Logistic Regression and the Keras neural network close behind, and Random Forest trailing — consistent with expectations for high-dimensional sparse text data. The most common confusion across all models is between **Business** and **Sci/Tech**, which makes sense given the overlap between tech-company and business news (e.g. Microsoft, Google earnings reports).
