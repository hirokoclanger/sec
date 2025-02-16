import os
import re
import pandas as pd
import numpy as np
import yfinance as yf
from sec_edgar_downloader import Downloader
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

ticker = 'MVST'

def download_ionq_10q_filings(limit, download_folder):
    """
    Uses sec-edgar-downloader to fetch up to `limit` "10-Q" filings for ticker IONQ.
    Filings are saved in `download_folder` (e.g., "data" directory).
    Returns a list of dictionaries with basic metadata:
        [
          {
            "filing_date": <datetime.date>,
            "form_type": "10-Q",
            "accession_no": <folder_name or parsed value>
          },
          ...
        ]
    """
    # 1) Initialize the downloader with your 'company_name' and 'email_address'
    dl = Downloader("MyCompanyName", "my.email@domain.com")

    # 2) Download the specified number of 10-Q filings for IONQ
    dl.get("10-Q", ticker, limit=limit)

    # 3) The directory structure is: <download_folder>/IONQ/10-Q/<subfolder>/
    base_10q_path = os.path.join(download_folder, ticker, "10-Q")
    print("Looking for 10-Q folders in:", base_10q_path)

   # Regex to find e.g. "CONFORMED PERIOD OF REPORT: 2023-06-30"
    date_pattern = re.compile(r"FILED AS OF DATE:\s*(\d{4}\d{2}\d{2})", re.IGNORECASE)
    form_pattern = re.compile(r"FORM TYPE:\s*(10-Q)", re.IGNORECASE)
    # Define the regex pattern to match EPS values
    eps_pattern = re.compile(
    r'<us-gaap:EarningsPerShareDiluted[^>]*id=["\'][^"\']+["\'][^>]*>(-?\d+\.\d+)</us-gaap:EarningsPerShareDiluted>',
    re.IGNORECASE
    )
    metadata = []
    for folder_name in os.listdir(base_10q_path):
        folder_path = os.path.join(base_10q_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        full_sub_path = os.path.join(folder_path, "full-submission.txt")
        if not os.path.exists(full_sub_path):
            continue

        with open(full_sub_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        date_match = date_pattern.search(content)
        form_match = form_pattern.search(content)
        eps_match = eps_pattern.findall(content)

        if date_match and form_match:
            raw_date = date_match.group(1)  # e.g. "20230810"
            date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            filing_date = pd.to_datetime(date_str).date()
            form_type = "10-Q"
            accession_no = folder_name  # use folder name as accession number

        
        eps_value = float(eps_match[0]) if eps_match else None

        metadata.append({
            "filing_date": filing_date,
            "form_type": form_type,
            "accession_no": accession_no,
            "eps": eps_value
        })

    return metadata

def get_ionq_stock_data(start="2021-01-01", end="2025-01-01"):
    """
    Fetch daily data for IONQ from yfinance and compute daily returns.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.index = pd.to_datetime(df.index)
    if df.empty:
        return df

    df["Return"] = df["Adj Close"].pct_change()
    return df


def analyze_pre_filing_windows(price_df, filings_info, pre_days=14):
    """
    For each 10-Q filing, look at the window [filing_date - pre_days, filing_date - 1]
    and compute stats: cumulative return, average daily return, and average volume.
    
    Parameters:
      price_df: DataFrame with daily stock data (must include 'Adj Close', 'Return', and 'Volume').
      filings_info: List of dicts with at least 'filing_date' key (as a date or string convertible to date).
      pre_days: Number of days before the filing_date to analyze (default is 14).
    
    Returns:
      A DataFrame summarizing each event's stats.
    """
    results = []
    for filing in filings_info:
        filing_dt = pd.to_datetime(filing["filing_date"])
        start_date = filing_dt - pd.Timedelta(days=pre_days)
        end_date = filing_dt - pd.Timedelta(days=1)
        window_data = price_df.loc[start_date:end_date]

        if len(window_data) < 1:
            continue
        
        cum_return = (window_data["Adj Close"].iloc[-1] / window_data["Adj Close"].iloc[0]) - 1
        avg_daily_return = window_data["Return"].mean()
        avg_volume = window_data["Volume"].mean() if "Volume" in window_data.columns else None

        results.append({
            "filing_date": filing_dt.date(),
            "pre_window_start": start_date.date(),
            "pre_window_end": end_date.date(),
            "days_in_window": len(window_data),
            "cumulative_return_%": round(cum_return * 100, 2),
            "avg_daily_return_%": round(avg_daily_return * 100, 2),
            "avg_volume": round(avg_volume, 2) if avg_volume is not None else None
        })

    return pd.DataFrame(results)

def check_pre_move(price_df, filings_info):
    """
    Checks the net move in the [filing_date - 7, filing_date - 3] window
    to see if the stock moved up or down.
    
    Parameters:
      price_df: DataFrame with a DateTimeIndex of daily prices (must include 'Adj Close').
      filings_info: List of dicts, each with a 'filing_date' key (and optionally 'eps' for EPS data).
    
    Returns:
      A DataFrame summarizing for each filing:
        - filing_date, start_date, end_date, days_in_window,
        - net_move_% (net move in percentage),
        - direction ("up" or "down"),
        - eps_basic (the EPS value from the filing metadata, if available)
    """
    results = []
    for filing in filings_info:
        filing_dt = pd.to_datetime(filing['filing_date'])
        # Define the window as [filing_date - 7, filing_date - 3]
        start_date = filing_dt - pd.Timedelta(days=15)
        end_date = filing_dt - pd.Timedelta(days=1)

        window_data = price_df.loc[start_date:end_date]
        if len(window_data) < 1:
            results.append({
                'filing_date': filing_dt.date(),
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'days_in_window': 0,
                'net_move_%': None,
                'direction': None,
                'eps_diluted': filing.get("eps", None)
            })
            continue

        # Convert to scalar float values to avoid ambiguous Series comparisons.
        start_price = float(window_data["Adj Close"].iloc[0])
        end_price = float(window_data["Adj Close"].iloc[-1])
        net_move = (end_price / start_price) - 1
        net_move_pct = round(net_move * 100, 2)
        direction = 'down' if net_move < 0 else 'up'

        results.append({
            'filing_date': filing_dt.date(),
            'start_date': start_date.date(),
            'end_date': end_date.date(),
            'days_in_window': len(window_data),
            'net_move_%': net_move_pct,
            'direction': direction,
            'eps_basic': filing.get("eps", None)
        })

    return pd.DataFrame(results)



def main():
    # 1. Download up to 5 10-Q filings, parse the filing dates
    ionq_filings = download_ionq_10q_filings(limit=5, download_folder="sec-edgar-filings")


    # 2. Get daily stock data
    df_ionq = get_ionq_stock_data(start="2021-01-01", end="2025-01-01")
 

    # 3. Pre-Filing Analysis (7-day window)
    df_results = analyze_pre_filing_windows(df_ionq, ionq_filings, pre_days=14)
    #print("\nPre-Filing Analysis:\n", df_results)

    # 4. Check 3–7 day move before filing
    df_pre_move = check_pre_move(df_ionq, ionq_filings)
    df_pre_move = df_pre_move.sort_values(by="filing_date", ascending=False)
    print("\n3–7 Day Move Before Filing:\n", df_pre_move)


if __name__ == "__main__":
    main()