# 📈 Stock Movement Prediction using Price + FinBERT Sentiment  

## 🎯 Goal  
Train a supervised machine learning model to predict **whether a stock’s next-day movement will be positive or negative** by combining:  
- **Financial news sentiment** (from FinViz headlines, analyzed using FinBERT)  
- **Daily stock price data** (Open, High, Low, Close, Volume, and technical indicators)  

This project demonstrates the full end-to-end workflow:  
**Data → NLP (FinBERT) → Feature Engineering → Model Training → Evaluation → Explainability (SHAP).**

---

## 🧠 Project Overview  

| Step | Description |
|------|--------------|
| **1. Data Collection** | - FinViz headlines scraped for a stock ticker (e.g., AAPL, GOOG).<br>- Daily price data downloaded from Yahoo Finance.<br>- Kaggle sentiment dataset used to benchmark FinBERT accuracy. |
| **2. Sentiment Analysis (FinBERT)** | Headlines are analyzed by **FinBERT**, a finance-specific NLP model, to compute negative, neutral, positive, and compound sentiment scores. |
| **3. Feature Engineering** | Combine OHLC price features (returns, moving averages, volatility) with sentiment features and rolling (3-day, 7-day) sentiment averages. |
| **4. Model Training** | Use a **RandomForestClassifier** to predict whether the next day’s closing price will be higher (`1`) or lower (`0`). |
| **5. Evaluation** | Compute Accuracy and F1-score on a time-based train/test split. |
| **6. Explainability (SHAP)** | Use SHAP to visualize which features most influence predictions. |
| **7. Prediction** | After training, the model outputs an “UP” or “DOWN” forecast for the next trading day. |

---

## 🧰 Tech Stack  

**Language:** Python 3  
**Libraries:**  
`pandas`, `numpy`, `scikit-learn`, `yfinance`, `transformers`, `torch`, `matplotlib`, `shap`, `joblib`  
**Sentiment Model:** [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)  
**ML Model:** RandomForest (supervised binary classifier)  
**Explainability:** SHAP (SHapley Additive exPlanations)

---

## ⚙️ Folder Structure  

# 📈 Stock Movement Prediction using Price + FinBERT Sentiment  

## 🎯 Goal  
Train a supervised machine learning model to predict **whether a stock’s next-day movement will be positive or negative** by combining:  
- **Financial news sentiment** (from FinViz headlines, analyzed using FinBERT)  
- **Daily stock price data** (Open, High, Low, Close, Volume, and technical indicators)  

This project demonstrates the full end-to-end workflow:  
**Data → NLP (FinBERT) → Feature Engineering → Model Training → Evaluation → Explainability (SHAP).**

---

## 🧠 Project Overview  

| Step | Description |
|------|--------------|
| **1. Data Collection** | - FinViz headlines scraped for a stock ticker (e.g., AAPL, GOOG).<br>- Daily price data downloaded from Yahoo Finance.<br>- Kaggle sentiment dataset used to benchmark FinBERT accuracy. |
| **2. Sentiment Analysis (FinBERT)** | Headlines are analyzed by **FinBERT**, a finance-specific NLP model, to compute negative, neutral, positive, and compound sentiment scores. |
| **3. Feature Engineering** | Combine OHLC price features (returns, moving averages, volatility) with sentiment features and rolling (3-day, 7-day) sentiment averages. |
| **4. Model Training** | Use a **RandomForestClassifier** to predict whether the next day’s closing price will be higher (`1`) or lower (`0`). |
| **5. Evaluation** | Compute Accuracy and F1-score on a time-based train/test split. |
| **6. Explainability (SHAP)** | Use SHAP to visualize which features most influence predictions. |
| **7. Prediction** | After training, the model outputs an “UP” or “DOWN” forecast for the next trading day. |

---

## 🧰 Tech Stack  

**Language:** Python 3  
**Libraries:**  
`pandas`, `numpy`, `scikit-learn`, `yfinance`, `transformers`, `torch`, `matplotlib`, `shap`, `joblib`  
**Sentiment Model:** [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)  
**ML Model:** RandomForest (supervised binary classifier)  
**Explainability:** SHAP (SHapley Additive exPlanations)

---

## ⚙️ Folder Structure  

```
stock-sentiment-project/
├── data/
│ ├── Sentiment_Stock_data.csv # Kaggle dataset (labeled sentiment)
│ ├── finviz_AAPL.csv # FinViz headlines for AAPL (example)
│ └── finviz_GOOG.csv # FinViz headlines for GOOG (example)
│
├── outputs/
│ ├── shap_summary.png # SHAP visualization
│ ├── model_randomforest.pkl # trained model
│ ├── scaler.pkl # saved feature scaler
│
├── stock_sentiment_pipeline.py # main training & evaluation script
├── scrape_finviz.py # FinViz scraper
├── demo.py # FinBERT headline demo script
├── requirements.txt # dependencies list
└── README.md
```

---

## 🚀 How to Run  

### Install dependencies  
```bash
pip install -r requirements.txt

### Place the kaggle sentiment dataset in:  
data/Sentiment_Stock_data.csv
### Place your FinViz news data in:
data/finviz_<TICKER>.csv

### Run the training pipeline:
python stock_sentiment_pipeline.py
Change ticker symbol in stock_sentiment_pipeline.py file to get analysis for stock of choice

### Example Output
=== Stock movement prediction with price + FinBERT sentiment ===
Loading Kaggle sentiment dataset...
Loading FinBERT model: ProsusAI/finbert ...
FinBERT vs Kaggle labels - Accuracy: 0.5130, F1: 0.1601

=== Stock movement prediction evaluation ===
Train Accuracy: 0.7575
Train F1 Score: 0.8111
Test Accuracy:  0.5030
Test F1 Score:  0.6640

=== Next-Day Movement Prediction ===
GOOG on 2023-12-28: UP ▲  (P(up) = 0.652)
Note: This predicts the move for the NEXT trading day after the date shown.
