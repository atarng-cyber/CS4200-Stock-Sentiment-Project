import os
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# =========================
# CONFIGURATION
# =========================

TICKER = "GOOG"

PRICE_START_DATE = "2014-01-01"
PRICE_END_DATE = "2024-01-01"

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

KAGGLE_SENTIMENT_PATH = os.path.join(DATA_DIR, "Sentiment_Stock_data.csv")
FINVIZ_CSV_PATH = os.path.join(DATA_DIR, f"finviz_{TICKER}.csv")


# =========================
# NLTK SETUP (VADER)
# =========================

def setup_vader():
    """
    Ensure VADER lexicon is downloaded and return a SentimentIntensityAnalyzer.
    """
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

# =========================
# FINBERT SENTIMENT MODEL
# =========================

class FinBertSentimentModel:
    """
    Wrapper around a FinBERT checkpoint for financial sentiment.
    Robust to label order by reading model.config.id2label.
    Outputs:
      - neg, neu, pos probabilities
      - compound score = pos - neg
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        print(f"Loading FinBERT model: {model_name} (first time may take a bit)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Build stable name→index map from the model’s own config
        # Example id2label could be {0:'positive', 1:'negative', 2:'neutral'}
        self.name_to_idx = {
            self.model.config.id2label[i].lower(): i
            for i in range(len(self.model.config.id2label))
        }
        # Ensure keys exist even if model uses odd names
        for k in ["positive", "negative", "neutral"]:
            self.name_to_idx.setdefault(k, None)

        print("id2label:", self.model.config.id2label)

    def score_texts(self, texts, batch_size: int = 32, max_length: int = 128):
        """
        Run FinBERT over a list of texts.
        Returns arrays: neg, neu, pos, compound.
        """
        import numpy as np

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**enc)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        probs = np.vstack(all_probs) if all_probs else np.zeros((0, 3))

        def col(label):
            idx = self.name_to_idx.get(label)
            return probs[:, idx] if idx is not None and probs.size else np.zeros(len(texts))

        neg = col("negative")
        neu = col("neutral")
        pos = col("positive")
        compound = pos - neg

        return neg, neu, pos, compound


# =========================
# UTILS
# =========================

def ensure_file(path: str, description: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} file not found at: {path}\n"
            f"Please create it as described in the instructions."
        )


# =========================
# KAGGLE SENTIMENT (LABELED)
# =========================

def load_kaggle_sentiment(csv_path: str) -> pd.DataFrame:
    """
    Load labeled Kaggle sentiment dataset in the format:

        ,Sentiment,Sentence
        0,0,"text..."
        1,1,"text..."
        ...

    - Drops the unnamed index column
    - Renames columns to: Sentiment (int), text (str)
    """
    ensure_file(csv_path, "Kaggle sentiment")
    print(f"Loading Kaggle sentiment dataset from {csv_path}...")

    df = pd.read_csv(csv_path)

    # Drop any unnamed index columns like "Unnamed: 0"
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    expected_cols = {"Sentiment", "Sentence"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Kaggle file must contain columns {expected_cols}, found {df.columns}"
        )

    # Keep only relevant columns
    df = df[["Sentiment", "Sentence"]].copy()

    # Ensure correct types
    df["Sentiment"] = df["Sentiment"].astype(int)
    df["text"] = df["Sentence"].astype(str)

    # Drop original Sentence column to avoid confusion
    df.drop(columns=["Sentence"], inplace=True)

    return df



def preprocess_text(text: str) -> str:
    """
    Very simple preprocessing:
    - lowercase
    - strip whitespace
    """
    return text.lower().strip()


def evaluate_finbert_on_kaggle(kaggle_df: pd.DataFrame,
                               finbert_model: FinBertSentimentModel,
                               sample_size: int = 10000):
    """
    Evaluate FinBERT against the Kaggle 0/1 sentiment labels.
    To keep runtime reasonable, use a random subset of up to sample_size rows.
    """
    print("\n=== Evaluating FinBERT on Kaggle sentiment dataset (subset) ===")

    if len(kaggle_df) > sample_size:
        df_sample = kaggle_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Using a random subset of {sample_size} rows out of {len(kaggle_df)}.")
    else:
        df_sample = kaggle_df.reset_index(drop=True)
        print(f"Using all {len(kaggle_df)} rows (small enough).")

    texts = df_sample["text"].astype(str).tolist()
    true_labels = df_sample["Sentiment"].astype(int).values  # 0/1

    neg, neu, pos, compound = finbert_model.score_texts(texts, batch_size=32, max_length=128)

    # Simple mapping: if positive prob is highest -> label 1, else 0
    probs = np.vstack([neg, neu, pos]).T
    pred_class = probs.argmax(axis=1)  # 0=neg, 1=neu, 2=pos
    pred_labels = (pred_class == 2).astype(int)

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print(f"FinBERT vs Kaggle labels - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print("Classification report:")
    print(classification_report(true_labels, pred_labels))



# =========================
# FINVIZ LOADING
# =========================

def load_text_source(csv_path: str, source_name: str) -> pd.DataFrame:
    """
    Generic loader for finviz CSVs with columns:
    - date (YYYY-MM-DD)
    - text
    """
    ensure_file(csv_path, source_name)
    print(f"Loading {source_name} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Try to be lenient about column names
    cols_lower = {c.lower(): c for c in df.columns}
    if "date" not in cols_lower or "text" not in cols_lower:
        raise ValueError(
            f"{source_name} file must have 'date' and 'text' columns. "
            f"Found: {list(df.columns)}"
        )

    date_col = cols_lower["date"]
    text_col = cols_lower["text"]

    df = df[[date_col, text_col]].copy()
    df.rename(columns={date_col: "Date", text_col: "text"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["text"] = df["text"].astype(str)
    df["source"] = source_name
    return df



# =========================
# SENTIMENT ON NEWS (FINVIZ)
# =========================

def add_finbert_sentiment(df: pd.DataFrame,
                          finbert_model: FinBertSentimentModel) -> pd.DataFrame:
    """
    Add FinBERT sentiment scores to a dataframe with a 'text' column.
    Produces columns: neg, neu, pos, compound.
    """
    print("Computing FinBERT sentiment scores...")
    texts = df["text"].astype(str).tolist()
    neg, neu, pos, compound = finbert_model.score_texts(texts, batch_size=32, max_length=128)

    out = df.copy()
    out["neg"] = neg
    out["neu"] = neu
    out["pos"] = pos
    out["compound"] = compound
    return out




def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment per day (mean of scores).
    """
    print("Aggregating daily sentiment...")
    daily = (
        df.groupby("Date")[["neg", "neu", "pos", "compound"]]
          .mean()
          .reset_index()
    )
    daily = daily.sort_values("Date")
    return daily

def add_rolling_sentiment_features(daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    From daily sentiment (neg, neu, pos, compound), add 3/7-day rolling
    mean and std features. Fills early NaNs with expanding means to keep rows.
    """
    df = daily_sentiment.sort_values("Date").copy()
    for col in ["neg", "neu", "pos", "compound"]:
        df[f"{col}_mean_3"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_std_3"]  = df[col].rolling(3, min_periods=2).std().fillna(0.0)
        df[f"{col}_mean_7"] = df[col].rolling(7, min_periods=1).mean()
        df[f"{col}_std_7"]  = df[col].rolling(7, min_periods=2).std().fillna(0.0)

        # Optional: expanding fallback to reduce early-window bias
        df[f"{col}_exp_mean"] = df[col].expanding(min_periods=1).mean()
    return df



# =========================
# PRICE DATA + FEATURES
# =========================

def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"Downloading price data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError("No price data downloaded. Check ticker or dates.")

    # Handle possible MultiIndex columns (e.g., ('Adj Close', 'AAPL'))
    if isinstance(df.columns, pd.MultiIndex):
        # If ticker is a level in the columns, select that slice
        if ticker in df.columns.levels[-1]:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            # Otherwise just drop the top level
            df.columns = df.columns.get_level_values(0)

    # Ensure expected columns exist; if Adj Close missing, fall back to Close
    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = list(df.columns)

    if "Adj Close" not in cols:
        if "Close" in cols:
            df["Adj Close"] = df["Close"]
            print("Warning: 'Adj Close' not found; using 'Close' as a proxy.")
        else:
            raise KeyError(
                f"Neither 'Adj Close' nor 'Close' found in downloaded data. "
                f"Columns present: {cols}"
            )

    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df.index.name = "Date"
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df



def make_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple technical features.
    """
    print("Engineering price features...")
    df = price_df.copy()
    df = df.sort_values("Date")

    df["return_1d"] = df["Adj Close"].pct_change()

    df["return_1d_lag1"] = df["return_1d"].shift(1)
    df["return_1d_lag2"] = df["return_1d"].shift(2)
    df["return_1d_lag3"] = df["return_1d"].shift(3)

    df["sma_5"] = df["Adj Close"].rolling(window=5).mean()
    df["sma_10"] = df["Adj Close"].rolling(window=10).mean()

    df["volatility_5"] = df["return_1d"].rolling(window=5).std()

    return df


def build_merged_dataset(price_df: pd.DataFrame,
                         daily_sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily price features with aggregated daily sentiment.
    Also create a binary target: 1 if next-day Close > today Close, else 0.
    """
    print("Merging price and sentiment data...")
    price_features = make_price_features(price_df)

    merged = pd.merge(
        price_features,
        daily_sentiment_df,
        on="Date",
        how="left"
    )

    # Fill missing sentiment (days with no news)
    sentiment_cols = ["neg", "neu", "pos", "compound"]
    merged[sentiment_cols] = merged[sentiment_cols].fillna(0.0)

    merged = merged.sort_values("Date")
    merged["Close_next"] = merged["Close"].shift(-1)
    merged["target_up"] = (merged["Close_next"] > merged["Close"]).astype(int)

    merged = merged.dropna(subset=["Close_next"])
    return merged


# =========================
# MODELING
# =========================

def make_feature_label_arrays(merged: pd.DataFrame):
    base_price = [
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "return_1d", "return_1d_lag1", "return_1d_lag2", "return_1d_lag3",
        "sma_5", "sma_10", "volatility_5",
    ]
    base_sent = ["neg", "neu", "pos", "compound"]

    # Rolling features we just created
    roll_feats = []
    for col in base_sent:
        roll_feats += [
            f"{col}_mean_3", f"{col}_std_3",
            f"{col}_mean_7", f"{col}_std_7",
            f"{col}_exp_mean",
        ]

    feature_cols = base_price + base_sent + roll_feats

    data = merged.dropna(subset=["target_up"]).copy()
    # If some rolling columns are missing (e.g., no news at all), fill with 0
    for c in feature_cols:
        if c not in data.columns:
            data[c] = 0.0

    X = data[feature_cols].values
    y = data["target_up"].values
    dates = data["Date"].values
    return X, y, dates, feature_cols

def latest_feature_row(merged: pd.DataFrame, feature_cols):
    """
    Return the last available feature row (fills missing sentiment with 0s).
    """
    df = merged.sort_values("Date").copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    # fill NaNs so we always have a valid row
    df[feature_cols] = df[feature_cols].fillna(0.0)

    last_row = df.iloc[-1]
    X_last = last_row[feature_cols].to_numpy().reshape(1, -1)
    last_date = last_row["Date"]
    return X_last, last_date


def print_next_day_prediction_message(ticker, model, scaler, merged, feature_cols):
    """
    Scale the latest feature row, predict next-day UP/DOWN, and print a friendly message.
    """
    X_last, last_date = latest_feature_row(merged, feature_cols)
    X_last_scaled = scaler.transform(X_last)
    proba = getattr(model, "predict_proba", None)
    if proba is not None:
        p = model.predict_proba(X_last_scaled)[0, 1]  # probability of "UP" class (1)
    else:
        # fallback: use decision function if available; otherwise 0.5
        p = 0.5
    pred = model.predict(X_last_scaled)[0]
    direction = "UP" if pred == 1 else "DOWN"
    arrow = "▲" if pred == 1 else "▼"
    print("\n=== Next-Day Movement Prediction ===")
    print(f"{ticker} on {last_date}: {direction} {arrow}  (P(up) = {p:.3f})")
    print("Note: This predicts the move for the NEXT trading day after the date shown.")



def time_based_train_test_split(X, y, dates, test_size=0.2):
    """
    Split train/test by time (no shuffling).
    """
    n = len(X)
    split_index = int(n * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    dates_train, dates_test = dates[:split_index], dates[split_index:]
    return X_train, X_test, y_train, y_test, dates_train, dates_test


def train_model_classifier(X_train, y_train):
    print("Training RandomForest classifier ...")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_train, y_train, X_test, y_test):
    print("\n=== Stock movement prediction evaluation ===")

    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Train F1 Score: {train_f1:.4f}")

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test F1 Score:  {test_f1:.4f}")

    print("\nClassification report (test set):")
    print(classification_report(y_test, y_test_pred))


# =========================
# SHAP INTERPRETABILITY
# =========================

def explain_with_shap(model, X_train, feature_names):
    print("Running SHAP analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure()
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    shap_plot_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(shap_plot_path)
    plt.close()
    print(f"Saved SHAP summary plot to: {shap_plot_path}")


# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=== Stock movement prediction with price + FinBERT sentiment ===")

    # 1. Load Kaggle dataset (Sentiment + Sentence -> text)
    kaggle_df = load_kaggle_sentiment(KAGGLE_SENTIMENT_PATH)

    # 2. Set up FinBERT (one-time load)
    finbert = FinBertSentimentModel()

    # 3. Evaluate FinBERT vs Kaggle labels
    evaluate_finbert_on_kaggle(kaggle_df, finbert, sample_size=10000)

    # 4. Load FinViz news only (since you removed Reddit)
    finviz_df = load_text_source(FINVIZ_CSV_PATH, "finviz")

    # 5. Compute FinBERT sentiment for FinViz headlines
    news_with_sentiment = add_finbert_sentiment(finviz_df, finbert)

    # 6. Aggregate daily sentiment and rolling sentiment
    daily_sentiment = aggregate_daily_sentiment(news_with_sentiment)
    daily_sentiment = add_rolling_sentiment_features(daily_sentiment)


    # 7. Load price data
    price_df = load_price_data(TICKER, PRICE_START_DATE, PRICE_END_DATE)

    # 8. Merge price + sentiment and build target
    merged_df = build_merged_dataset(price_df, daily_sentiment)

    # 9. Features + labels
    X, y, dates, feature_names = make_feature_label_arrays(merged_df)

    # 10. Time-based split + scaling
    X_train, X_test, y_train, y_test, _, _ = time_based_train_test_split(
        X, y, dates, test_size=0.2
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 11. Train RandomForest classifier
    model = train_model_classifier(X_train_scaled, y_train)   # your RF function

    # 12. Evaluate
    evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # 13. SHAP interpretability
    explain_with_shap(model, X_train_scaled, feature_names)

    # 14. Save model & scaler
    model_path = os.path.join(OUTPUT_DIR, "model_randomforest.pkl")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nSaved trained RandomForest model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")

    # 15) Print next-day prediction message (relative to the last date in merged_df)
    print_next_day_prediction_message(TICKER, model, scaler, merged_df, feature_names)


    print("\nPipeline complete.")



if __name__ == "__main__":
    main()
