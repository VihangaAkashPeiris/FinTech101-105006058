import requests
import pandas as pd

API_KEY = "dfedfc76d914430ab44d3ec6f39c501a"

query = "Commonwealth Bank OR CBA.AX"
from_date = "2025-10-01"   # start of your range
to_date   = "2025-10-17"   # end of your range

endpoint = "https://newsapi.org/v2/everything"

all_articles = []
page = 1

while True:
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,           # max per request
        "page": page,
        "from": from_date,
        "to": to_date,
        "apiKey": API_KEY
    }

    response = requests.get(endpoint, params=params)
    data = response.json()

    articles = data.get("articles", [])
    if not articles:
        break  # stop if no more results

    all_articles.extend(articles)
    page += 1

df = pd.DataFrame([
    {
        "date": a["publishedAt"][:10],
        "headline": a["title"],
        "source": a["source"]["name"]
    }
    for a in all_articles
])

df.to_csv("CBA_news.csv", index=False)
print(f"Total articles: {len(df)}")
print(df.head())
