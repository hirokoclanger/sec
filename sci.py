import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Check for missing values
print("Missing values in each column:")
print(df.isna().sum())

# For simplicity, drop any rows with missing values (or alternatively, impute them)
df = df.dropna()

# Display basic statistics
print("\nData Summary:")
print(df.describe())

# Plot 1: Scatter plot of Consumption vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Cons', y='Price')
plt.title('Price vs Consumption')
plt.xlabel('Consumption')
plt.ylabel('Price')
plt.show()

# Plot 2: Scatter plot of ONIChange vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='ONIChange', y='Price')
plt.title('Price vs ONI Change')
plt.xlabel('ONI Change')
plt.ylabel('Price')
plt.show()

# Plot 3: Scatter plot of ONIChange vs Consumption
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='ONIChange', y='Cons')
plt.title('Consumption vs ONI Change')
plt.xlabel('ONI Change')
plt.ylabel('Consumption')
plt.show()

# Plot 4: Time series plot of Price and Consumption over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Price', label='Price')
sns.lineplot(data=df, x='Date', y='Cons', label='Consumption')
plt.title('Price and Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate and display the correlation matrix
corr_matrix = df[['ONIChange', 'Cons', 'Price']].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Optionally, plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()