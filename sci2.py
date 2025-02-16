import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# File path to the Excel file
file_path = 'data/combined_data2.xlsx'

# Read each sheet into a DataFrame, parsing the Date column as datetime
df_ONI = pd.read_excel(file_path, sheet_name='ONI', parse_dates=['Date'])
df_CONS = pd.read_excel(file_path, sheet_name='CONSUMPTION', parse_dates=['Date'])
df_PRICES = pd.read_excel(file_path, sheet_name='PRICES', parse_dates=['Date'])

# Merge the dataframes on 'Date' using an outer join to keep all dates
df = pd.merge(df_ONI, df_CONS, on='Date', how='outer')
df = pd.merge(df, df_PRICES, on='Date', how='outer')

# Sort the merged dataframe by Date
df.sort_values('Date', inplace=True)

# Check for missing values and drop any rows with missing values for simplicity
print("Missing values in each column:")
print(df.isna().sum())
df = df.dropna()

# Display basic statistics
print("\nData Summary:")
print(df.describe())

# --------------------
# CLUSTERING
# --------------------
# We will cluster based on all three features: ONIChange, Cons, and Price.
# Since they are on different scales, we standardize the data first.
features = ['ONIChange', 'Cons', 'Price']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Apply KMeans clustering with 3 clusters (adjust the number of clusters as needed)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# --------------------
# PLOTS
# --------------------

# Plot 1: Scatter plot of Price vs Consumption with clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Cons', y='Price', hue='Cluster', palette='viridis', s=100, edgecolor='k')
plt.title('Price vs Consumption (with Clusters)')
plt.xlabel('Consumption')
plt.ylabel('Price')
plt.legend(title='Cluster')
plt.show()

# Plot 2: Scatter plot of ONIChange vs Price with clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='ONIChange', y='Price', hue='Cluster', palette='viridis', s=100, edgecolor='k')
plt.title('Price vs ONI Change (with Clusters)')
plt.xlabel('ONI Change')
plt.ylabel('Price')
plt.legend(title='Cluster')
plt.show()

# Plot 3: Scatter plot of ONIChange vs Consumption with clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='ONIChange', y='Cons', hue='Cluster', palette='viridis', s=100, edgecolor='k')
plt.title('Consumption vs ONI Change (with Clusters)')
plt.xlabel('ONI Change')
plt.ylabel('Consumption')
plt.legend(title='Cluster')
plt.show()

# Additional Plot: Pair Plot of all three variables colored by cluster
sns.pairplot(df, vars=features, hue='Cluster', palette='viridis', diag_kind='kde', height=3)
plt.suptitle('Pair Plot of ONIChange, Consumption, and Price (with Clusters)', y=1.02)
plt.show()

# Additional Plot: Regression plots to visualize potential trends
# ONIChange vs Price with a regression line and clusters
plt.figure(figsize=(10, 6))
sns.lmplot(data=df, x='ONIChange', y='Price', hue='Cluster', palette='viridis', markers=["o", "s", "D"],
           height=6, aspect=1.2, scatter_kws={'s': 80, 'edgecolor': 'k'})
plt.title('Regression: Price vs ONI Change (with Clusters)')
plt.xlabel('ONI Change')
plt.ylabel('Price')
plt.show()

# --------------------
# Correlation Analysis
# --------------------
# Calculate and display the correlation matrix
corr_matrix = df[['ONIChange', 'Cons', 'Price']].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap')
plt.show()