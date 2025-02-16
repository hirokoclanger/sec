import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pennylane as qml
import torch

# File paths
orderbook_path = "data/AAPL_2012-06-21_34200000_37800000_orderbook_30.csv"
message_path = "data/AAPL_2012-06-21_34200000_37800000_message_30.csv"

# Load order book (4 columns per level)
num_levels = 10
columns = []
for i in range(1, num_levels + 1):
    columns += [f'ask_price_{i}', f'ask_size_{i}', f'bid_price_{i}', f'bid_size_{i}']

orderbook_df = pd.read_csv(orderbook_path, header=None, names=columns)

# Load message file (LOBSTER format)
message_cols = ["Time", "EventType", "OrderID", "Size", "Price", "Direction"]
message_df = pd.read_csv(message_path, header=None, names=message_cols)

# Ensure both datasets have the same length
min_length = min(len(orderbook_df), len(message_df))
orderbook_df = orderbook_df.iloc[:min_length]
message_df = message_df.iloc[:min_length]

# Merge order book and message data
merged_df = pd.concat([message_df.reset_index(drop=True), orderbook_df.reset_index(drop=True)], axis=1)

# Mark cancellations
merged_df['Is_Cancellation'] = (merged_df['EventType'] == 3).astype(int)

# Count cancellations per second (to detect spoofing clusters)
merged_df['Cancellation_Count'] = merged_df['Is_Cancellation'].rolling(window=50, min_periods=1).sum()

# Identify high spoofing likelihood events
spoofing_threshold = merged_df['Cancellation_Count'].quantile(0.95)
merged_df['Potential_Spoofing'] = (merged_df['Cancellation_Count'] > spoofing_threshold).astype(int)

print("Potential spoofing cases detected:", merged_df['Potential_Spoofing'].sum())

# Drop NaN values before clustering
merged_df.dropna(inplace=True)

# Feature selection for clustering
features = ['bid_price_1', 'ask_price_1', 'bid_size_1', 'ask_size_1', 'Cancellation_Count']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(merged_df[features])

# Apply K-Means
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
merged_df['Cluster'] = kmeans.fit_predict(data_scaled)

# Analyze spoofing per cluster
spoofing_distribution = merged_df.groupby("Cluster")["Potential_Spoofing"].sum()
print("Spoofing Events per Cluster:\n", spoofing_distribution)



# Additional Analysis
# 1. Correlation between spoofing clusters and cancellation rate
correlation = merged_df[['Cluster', 'Cancellation_Count']].corr()
print("Correlation between Cluster and Cancellation Count:\n", correlation)

# 2. Distribution of clusters over large price movements
price_movement_threshold = merged_df['bid_price_1'].std() * 2
merged_df['Large_Price_Movement'] = (merged_df['bid_price_1'].diff().abs() > price_movement_threshold).astype(int)
large_movements_distribution = merged_df.groupby("Cluster")["Large_Price_Movement"].sum()
print("Clusters associated with large price movements:\n", large_movements_distribution)

# Visualization of clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=merged_df['bid_price_1'], y=merged_df['ask_price_1'], hue=merged_df['Cluster'], palette="viridis", alpha=0.5)
plt.title("K-Means Clustering with Spoofing Highlights")
plt.show()
# 3. Cluster evolution over time
plt.figure(figsize=(10,6))
sns.lineplot(x=merged_df['Time'], y=merged_df['Cluster'], hue=merged_df['Cluster'], palette="viridis", alpha=0.5)
plt.title("Cluster Evolution Over Time")
plt.xlabel("Time (Seconds After Midnight)")
plt.ylabel("Cluster ID")
plt.show()


# Final Performance Comparison
print(f'Classical K-Means Silhouette Score: {silhouette_score(data_scaled, merged_df["Cluster"])}')

