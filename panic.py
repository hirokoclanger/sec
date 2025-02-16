import pandas as pd
import numpy as np
import kagglehub
import requests
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# -------------------------
# 1. Download and Load the S&P 500 Dataset
# -------------------------
# Download latest version from Kaggle Hub
path = kagglehub.dataset_download("henryhan117/sp-500-historical-data")
print("Path to dataset files:", path)

# List CSV files in the downloaded path
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print("CSV files found:", csv_files)

# Assume the dataset contains a CSV file with historical price data.
# (If the file name is different, adjust the filename below accordingly.)
data_file = os.path.join(path, csv_files[0])
market = pd.read_csv(data_file, parse_dates=['Date'])

# Compute daily returns and volume change (example: percentage change)
market['Return'] = market['Close'].pct_change()
market['VolumeChange'] = market['Volume'].pct_change()
market.dropna(inplace=True)

# Define thresholds (customize as needed)
price_drop_threshold = -0.02  # e.g., price drops more than 2%
volume_increase_threshold = 0.10  # e.g., volume increases more than 10%

# Flag days with significant price drop and volume spike
market['EventFlag'] = ((market['Return'] < price_drop_threshold) &
                       (market['VolumeChange'] > volume_increase_threshold)).astype(int)

# --- Step 2: Load and Process News Data via GDELT ---
# Define the date for which you want to retrieve news (adjust as needed)
# --- Step 2: Load and Process News Data via GDELT ---
# Define the target date (make sure it's defined!)
# --- Step 2: Load and Process News Data via GDELT ---
# Define the target date (make sure it's defined!)
date = "2021-09-01"

query = "market"
startdatetime = date.replace("-", "") + "000000"  # "20210901000000"
enddatetime = date.replace("-", "") + "235959"    # "20210901235959"

# Build the GDELT API URL; adjust parameters if needed
url = (f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}"
       f"&mode=ArtList&startdatetime={startdatetime}&enddatetime={enddatetime}&format=JSON")

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    # The API is expected to return a list of articles under the key "articles"
    articles = data.get("articles", [])
    
    # Build a list of dictionaries with keys "Date" and "Headline"
    news_list = []
    for article in articles:
        headline = article.get("title", "")
        seendate = article.get("seendate", None)
        if seendate is not None:
            # Convert seendate to a pandas datetime object, then take just the date
            news_date = pd.to_datetime(seendate).date()
            news_list.append({
                "Date": news_date,
                "Headline": headline
            })
    
    # Create DataFrame from the news list and sort by Date
    news = pd.DataFrame(news_list)
    news.sort_values("Date", inplace=True)
    
    # Use NLTK VADER for sentiment analysis on each headline
    sid = SentimentIntensityAnalyzer()
    news["Sentiment"] = news["Headline"].apply(
        lambda x: sid.polarity_scores(x)["compound"] if x else 0
    )
    
    print("News data sample:")
    print(news.head())
else:
    print("Error retrieving GDELT data:", response.status_code)
    news = pd.DataFrame(columns=["Date", "Headline", "Sentiment"])

# Aggregate news sentiment by date (average sentiment)
daily_sentiment = news.groupby('Date')['Sentiment'].mean().reset_index()
daily_sentiment.rename(columns={'Sentiment': 'AvgSentiment'}, inplace=True)

# Define negative sentiment as having an average score below a threshold
sentiment_threshold = -0.2
daily_sentiment['NegativeNews'] = (daily_sentiment['AvgSentiment'] < sentiment_threshold).astype(int)

# --- Step 3: Merge Data ---
# Ensure that the market DataFrame's Date column is a date object (without time)
market['Date'] = pd.to_datetime(market['Date']).dt.date

# Merge market and daily sentiment data on Date
data = pd.merge(market, daily_sentiment, on='Date', how='inner')

# Now we have features from market data and a label from news sentiment (NegativeNews)
features = ['Return', 'VolumeChange']  # You could add more engineered features
X = data[features]
y = data['NegativeNews']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Train a Classifier ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate performance
print(classification_report(y_test, y_pred))

# --- Optional: Visualize Events and News Sentiment ---
plt.figure(figsize=(12, 6))
plt.scatter(data['Date'], data['Return'], c=data['NegativeNews'], cmap='coolwarm', alpha=0.6)
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.title("Market Returns Colored by Negative News Flag")
plt.colorbar(label="NegativeNews (1=Yes, 0=No)")
plt.show()