# Sentiment Analysis: TF-IDF vs SBERT

Comparing a classical TF-IDF + Logistic Regression pipeline against a modern 
SBERT + SGDClassifier pipeline for tweet sentiment classification on the 
Sentiment140 dataset (1.6M tweets).

**Result:** TF-IDF (80.56% test accuracy) outperformed SBERT (77.26% 
validation accuracy). See "Key Finding" below for why, and what closed most 
of the gap.

**Stack:** Python, scikit-learn, sentence-transformers, NLTK, pandas

## Approach

- **TF-IDF + Logistic Regression:** classic bag-of-words pipeline — regex 
  cleaning, tokenization, stopword removal, Porter stemming, then 
  `TfidfVectorizer` + `LogisticRegression`, tuned via `GridSearchCV`
- **SBERT + SGDClassifier:** tweets encoded into 384-dim embeddings using 
  `all-MiniLM-L6-v2`. Switched from `LogisticRegression` to `SGDClassifier` 
  after repeated RAM crashes on Colab's free tier — `SGDClassifier` trains 
  incrementally in mini-batches instead of needing the full dataset in 
  memory at once

## Key Finding

SBERT initially underperformed badly (73% vs TF-IDF's 80%). Rather than 
accept "embeddings lose to classical methods," I investigated: ruled out 
regularization choice (L1 vs L2) and linear-boundary limitations (tested an 
RBF SVM), then found the real cause — I was feeding SBERT the same 
heavily-stemmed, stopword-stripped text built for TF-IDF. SBERT was trained 
on natural, grammatically intact sentences, so fragments like 
"movi great not slow" were out-of-distribution input for its tokenizer.

I built a second, lighter cleaning function preserving natural sentence 
structure for SBERT specifically. This raised SBERT's validation accuracy 
from 73% to **77.26%** — a substantial, measurable fix.

I also caught a second bug: NLTK's default English stopword list removes 
negation words ("not", "no", "never"), which flips sentiment meaning during 
cleaning ("not good" → "good"). Fixed by excluding negation words from the 
stopword set.

## Results

| Model | Validation Accuracy | Validation F1 | Encoding Time (train) |
|---|---|---|---|
| TF-IDF + Logistic Regression | 80.48% | 80.66% | 21.74s |
| SBERT + SGDClassifier | 77.26% | 77.09% | 496.76s |

**Final model (TF-IDF) — test set evaluation:**
- Accuracy: 80.56% | F1: 80.81% | Precision: 79.88% | Recall: 81.75%

## Limitations

- Binary classifier — cannot represent mixed-sentiment tweets (e.g. "food 
  was great, but service was slow"); TF-IDF's bag-of-words approach has no 
  way to recognize that "but" often signals the second clause dominates
- SBERT embeddings are frozen, not fine-tuned for this task/domain
- Hyperparameter tuning was done on a 30,000-row stratified subsample for 
  tractability; final models were retrained on the full dataset

## How to Run

1. Requires a Kaggle API key (for downloading Sentiment140), set as Colab 
   secrets: `kaggle_username`, `kaggleAPI`
2. Open `sentiment_analysis.ipynb` in Google Colab (GPU runtime recommended 
   for SBERT encoding)
3. Run all cells top to bottom

## Example Usage

After running the notebook:
```python
predict_sentiment_vec("I absolutely loved this!")
# → "Positive"
```
