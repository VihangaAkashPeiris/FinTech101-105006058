from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

MODEL_NAME = "yiyanghkust/finbert-tone"
BATCH_SIZE = 32
MAX_LEN = 128

def Load_data():
    df = pd.read_csv("Taskc7/upload_DJIA_table.csv")
    df = df.rename(columns={"Adj Close": "Adjclose"})
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date", ascending=True).dropna().reset_index(drop=True)

    df_1 = pd.read_csv("Taskc7/RedditNews.csv")
    df_1.columns = df_1.columns.str.lower()     # => 'date', 'news'
    df_1["date"] = pd.to_datetime(df_1["date"])
    df_1 = df_1.sort_values(by="date", ascending=True).dropna(subset=["news"]).reset_index(drop=True)
    df_1["news"] = df_1["news"].astype(str).str.replace(r"^b['\"]|['\"]$", "", regex=True)
    return df, df_1

def generate_sent_score(df: pd.DataFrame) -> pd.DataFrame:
    sid = SentimentIntensityAnalyzer()
    tqdm.pandas()
    
    df['compound'] = df['news'].progress_apply(
        lambda x: sid.polarity_scores(str(x))['compound']
    )
    df.to_csv("Taskc7/news_plus_sentiment.csv")
    return df

def finbert_sent_score(df: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # FP16 on MPS, FP32 otherwise
    torch_dtype = torch.float16 if device.type == "mps" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    def score_batch(texts):
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,   # try 64 if your texts are short; use 128 if needed
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
        return probs[:, 2] - probs[:, 0]  # pos - neg

    scores = np.zeros(len(df), dtype=np.float32)
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Scoring with FinBERT (MPS-ready)"):
        end = min(start + BATCH_SIZE, len(df))
        batch_texts = df["news"].iloc[start:end].astype(str).tolist()
        scores[start:end] = score_batch(batch_texts)

    df = df.copy()
    df["finbert_score"] = scores
   
    
    return df

    
        

if __name__ == "__main__":
    df_prices, df_news = Load_data()
    df_news_scored = generate_sent_score(df_news)
    finbert_score = finbert_sent_score(df_news)

    finbert_score.to_csv("Taskc7/Vader_&_finbert_score.csv")
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
    df_prices.to_csv("Taskc7/daily_prices")
    daily_sentiment.to_csv("Taskc7/Vader_daily_sentiment.csv")
    print(daily_sentiment['sent_mean'].describe())

    v_pos_count = (finbert_score["compound"]>0).sum()
    v_neg_count = (finbert_score["compound"]<0).sum()
    v_neu_count = (finbert_score["compound"]==0).sum()
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

    daily = (
    finbert_score.groupby("date")
           .agg(vader_mean   = ("compound", "mean"),
                finbert_mean   = ("finbert_score", "mean"))
           .reset_index()
    )

    
    daily.to_csv("Taskc7/daily_vader_finbert.csv", index=False)
    final_df = pd.merge(df_prices,daily, on="date", how="inner")
    final_df.to_csv("Taskc7/final_prices_&_sentiment.csv",index=False)
    
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
    print(df_news_scored.head())
    print(df_news_scored.tail())
    print(daily_sentiment.head())