from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

# Pretrained FinBERT model name
MODEL_NAME = "yiyanghkust/finbert-tone"
# Batch size for FinBERT scoring loop
BATCH_SIZE = 32
# Max token length when tokenizing headlines
MAX_LEN = 128

def Load_data():
    # Load DJIA prices
    df = pd.read_csv("Taskc7/upload_DJIA_table.csv")
    # Standardize column name for adjusted close
    df = df.rename(columns={"Adj Close": "Adjclose"})
    # Make all column names lowercase
    df.columns = df.columns.str.lower()
    # Ensure proper datetime type
    df["date"] = pd.to_datetime(df["date"])
    # Sort by date and clean
    df = df.sort_values(by="date", ascending=True).dropna().reset_index(drop=True)

    # Load Reddit news
    df_1 = pd.read_csv("Taskc7/RedditNews.csv")
    # Keep a tidy, lowercase schema     # => 'date', 'news'
    df_1.columns = df_1.columns.str.lower()
    df_1["date"] = pd.to_datetime(df_1["date"])
    # Sort and remove rows with missing news text
    df_1 = df_1.sort_values(by="date", ascending=True).dropna(subset=["news"]).reset_index(drop=True)
    # Strip leading b' and trailing quotes if present
    df_1["news"] = df_1["news"].astype(str).str.replace(r"^b['\"]|['\"]$", "", regex=True)
    return df, df_1

#   Generate Vader Sentiment Score 
def generate_sent_score(df: pd.DataFrame) -> pd.DataFrame:
    # VADER analyzer for compound score
    sid = SentimentIntensityAnalyzer()
    # Enable progress bar for apply
    tqdm.pandas()
    
    # Compute VADER compound score per headline
    df['compound'] = df['news'].progress_apply(
        lambda x: sid.polarity_scores(str(x))['compound']
    )
    # Save raw headline-level scores
    df.to_csv("Taskc7/news_plus_sentiment.csv")
    return df
# Genrate finbert sentiment scores
def finbert_sent_score(df: pd.DataFrame) -> pd.DataFrame:
    # Pick MPS because I am using a mac.(apple gpu)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Use FP16 for MPS, FP32 otherwise to keep it stable
    torch_dtype = torch.float16 if device.type == "mps" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    def score_batch(texts):
        # Tokenize a batch of headlines
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN,  
        )
        # Move to device
        enc = {k: v.to(device) for k, v in enc.items()}

        # Forward pass with no grads
        with torch.inference_mode():
            out = model(**enc)
            # Convert logits to probabilities
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
        # FinBERT order: [negative, neutral, positive]
        # We use a simple sentiment score: pos - neg
        return probs[:, 2] - probs[:, 0]  # pos - neg

    # Preallocate result array
    scores = np.zeros(len(df), dtype=np.float32)
    # Score in batches for speed
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Scoring with FinBERT (MPS-ready)"):
        end = min(start + BATCH_SIZE, len(df))
        batch_texts = df["news"].iloc[start:end].astype(str).tolist()
        scores[start:end] = score_batch(batch_texts)

    # Attach to a copy to avoid side effects
    df = df.copy()
    df["finbert_score"] = scores
   
    
    return df

    
        

if __name__ == "__main__":
    # Load prices and raw news
    df_prices, df_news = Load_data()
    # VADER per-headline scores
    df_news_scored = generate_sent_score(df_news)
    # FinBERT per-headline scores
    finbert_score = finbert_sent_score(df_news)

    # Save headline-level FinBERT output
    finbert_score.to_csv("Taskc7/Vader_&_finbert_score.csv")
    # Aggregate VADER by date for quick daily stats
    daily_sentiment = (
        df_news_scored.groupby('date')['compound']
        .agg(['mean', 'median', 'count'])
        .reset_index()
        .rename(columns={
            'mean': 'sent_mean',
            'median': 'sent_median',
            'count': 'headline_count'
        })
    )
    # Save daily prices as-is
    df_prices.to_csv("Taskc7/daily_prices")
    # Save daily VADER summary
    daily_sentiment.to_csv("Taskc7/Vader_daily_sentiment.csv")
    # Quick sanity check on distribution
    print(daily_sentiment['sent_mean'].describe())

    # VADER counts by sign
    v_pos_count = (df_news_scored["compound"]>0).sum()
    v_neg_count = (df_news_scored["compound"]<0).sum()
    v_neu_count = (df_news_scored["compound"]==0).sum()
    # FinBERT counts by sign of score
    pos_count = (finbert_score["finbert_score"] > 0).sum()
    neg_count = (finbert_score["finbert_score"] < 0).sum()
    neu_count = (finbert_score["finbert_score"] == 0).sum()  # optional
    


    print ("===================Vader-Sentiment================" )
    print("Positive:", v_pos_count)
    print("Negative:", v_neg_count)
    print("Neutral:", v_neu_count)
    print ("===================Finbert-Sentiment================" )
    print("Positive:", pos_count)
    print("Negative:", neg_count)
    print("Neutral:", neu_count)

    # Build a daily table with both signals
    daily = (
    finbert_score.groupby("date")
           .agg(vader_mean   = ("compound", "mean"),
                finbert_mean   = ("finbert_score", "mean"))
           .reset_index()
    )

    
    # Save daily combined sentiment
    daily.to_csv("Taskc7/daily_vader_finbert.csv", index=False)
    # Join with prices on date
    final_df = pd.merge(df_prices,daily, on="date", how="inner")
    # Save merged table for modeling
    final_df.to_csv("Taskc7/final_prices_&_sentiment.csv",index=False)
    
    # Simple histogram of daily VADER means
    plt.figure(figsize=(8,5))
    plt.hist(daily_sentiment['sent_mean'], bins=30, color='steelblue', edgecolor='black')
    plt.axvline(daily_sentiment['sent_mean'].mean(), color='red', linestyle='dashed', linewidth=2, label=f"Mean = {daily_sentiment['sent_mean'].mean():.2f}")
    plt.axvline(daily_sentiment['sent_mean'].median(), color='green', linestyle='dotted', linewidth=2, label=f"Median = {daily_sentiment['sent_mean'].median():.2f}")
    plt.title("Distribution of Daily Average Sentiment Scores")
    plt.xlabel("Daily Mean Sentiment (VADER)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()
    # Peek at the head and tail for quick inspection
    print(df_news_scored.head())
    print(df_news_scored.tail())
    print(daily_sentiment.head())