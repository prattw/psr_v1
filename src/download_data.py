import json
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

def fetch_stock_data(symbol, interval='1min', outputsize='full'):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    # Ensure you're passing the interval and outputsize to the URL
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}&datatype=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        time_series_key = f"Time Series ({interval})"
        time_series = data.get(time_series_key, {})
        
        if not time_series:
            print(f"No data found for {symbol}. Check the API response for error messages.")
            return None
        
        # Parse the JSON data into a DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index').astype(float)
        df.index = pd.to_datetime(df.index)
        df.rename(columns=lambda s: s[3:], inplace=True)  # Removing the numerical prefix from column names.

        # Save the DataFrame as a CSV file.
        df.to_csv(f'/Users/williampratt/Library/Mobile Documents/com~apple~CloudDocs/Documents/project_sea_ranch/data/raw/{symbol}_intraday_{interval}.csv')
        
        print(f"Data for {symbol} fetched and saved successfully.")
        return df
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        return None
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
        return None
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
        return None
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
        return None

# Specify the stock symbols
symbols = ['SPY', 'META', 'TSLA', 'AMZN', 'MSFT', 'AAPL', 'GOOG', 'NVDA', 'VOO']
# Fetch and save data for each stock symbol
for symbol in symbols:
    print(f"Fetching data for {symbol}...")
    stock_data = fetch_stock_data(symbol)
    if stock_data is None:
        print(f"Failed to fetch data for {symbol}.")
    
    # Sleep for 1 minute before the next API call
    print(f"Waiting for 1 minute before fetching the next symbol...")
    time.sleep(60)  # Sleeps for 60 seconds