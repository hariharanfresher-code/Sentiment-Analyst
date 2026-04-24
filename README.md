# 🎭 Twitter Sentiment Analysis

An **intermediate-level NLP project** that trains and compares multiple machine learning models to classify tweet sentiment as **Positive**, **Negative**, or **Neutral**.

## 📦 Dataset

| File | Description | Rows |
|------|-------------|------|
| `train.csv` | Kaggle Twitter Sentiment (3-class) | 27,481 |
| `test.csv` | Kaggle Twitter Sentiment test set | 4,815 |
| `testdata_manual_2009_06_14.csv` | Sentiment140 manual test (binary) | 498 |
| `training_1600000_processed_noemoticon.csv` | Full Sentiment140 dataset | 1.6 M |

> 📌 Place all CSVs inside the `data/` folder before running.

## 🗂 Project Structure

```
sentiment_analysis/
├── sentiment_analysis.ipynb   ← Main notebook (run this)
├── requirements.txt           ← Python dependencies
├── README.md
├── data/                      ← Put your CSVs here
├── models/                    ← Auto-created; saves trained pipeline
└── outputs/                   ← Auto-created; saves all plots
```

## 🔬 Methodology

1. **EDA** — class distribution, tweet length analysis, top-country chart
2. **Preprocessing** — lowercase, URL/mention removal, stopword filter, lemmatization
3. **Visualisation** — word clouds per class, top bigrams
4. **Feature Engineering** — TF-IDF (unigram + bigram, 50 K features) vs Bag-of-Words
5. **Model Training** — Logistic Regression, Naive Bayes, Linear SVC, Random Forest
6. **Evaluation** — accuracy, F1-macro/weighted, confusion matrix, cross-validation
7. **Error Analysis** — inspect and understand misclassifications
8. **Inference Demo** — run predictions on any custom text

## 🚀 Quick Start

### 1 — Clone & install
```bash
git clone https://github.com/<your-username>/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
```

### 2 — Add data
```bash
mkdir data
# Copy your CSVs into data/
```

### 3 — Open in VS Code
```bash
code .
# Open sentiment_analysis.ipynb
# Select Python kernel → Run All Cells
```

### 4 — Or run from terminal
```bash
jupyter notebook sentiment_analysis.ipynb
```

## 📊 Results

| Model | Accuracy | F1-Macro | F1-Weighted |
|-------|----------|----------|-------------|
| Logistic Regression | ~0.81 | ~0.81 | ~0.81 |
| Linear SVC | ~0.81 | ~0.81 | ~0.81 |
| Naive Bayes | ~0.74 | ~0.74 | ~0.74 |
| Random Forest | ~0.73 | ~0.72 | ~0.73 |

> Results may vary slightly due to random seeds and NLTK version.

## 📁 Outputs

After running the notebook, `outputs/` will contain 10 plots:

| File | Description |
|------|-------------|
| `01_eda_overview.png` | Bar + pie + length histogram |
| `02_length_boxplot.png` | Tweet length per class |
| `03_top_countries.png` | Top 10 countries by tweet count |
| `04_wordclouds.png` | Word clouds per sentiment |
| `05_top_bigrams.png` | Most frequent bigrams |
| `06_model_comparison.png` | Side-by-side model metrics |
| `07_confusion_matrix.png` | Counts + normalized CM |
| `08_top_features.png` | LR coefficient analysis |
| `09_cross_validation.png` | 5-fold CV boxplot |
| `10_prediction_confidence.png` | Per-tweet confidence bars |

## 🔮 Custom Predictions

```python
from sentiment_analysis import predict_sentiment   # or copy the function from the notebook

tweets = ["I love this!", "This is the worst!", "Just had lunch."]
df = predict_sentiment(tweets)
print(df)
```

## 🛠 Tech Stack

- **Python 3.10+**
- pandas · numpy · scikit-learn · NLTK
- matplotlib · seaborn · wordcloud
- Jupyter Notebook / VS Code

## 📜 License

MIT
