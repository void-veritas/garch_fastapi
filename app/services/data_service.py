# Placeholder for data fetching logic from FMP

import requests
import pandas as pd
from io import StringIO


def fetch_data(symbol: str, api_key: str, period: str = "daily") -> pd.DataFrame | None:
    """Fetches historical price data for a given symbol from FMP.

    Args:
        symbol: The stock ticker symbol (e.g., 'SPY').
        api_key: Your FinancialModelingPrep API key.
        period: The frequency of the data ('daily' is typical for GARCH).
                 FMP also supports '1min', '5min', '15min', '30min', '1hour', '4hour'.
                 However, the free tier often limits historical intraday data.

    Returns:
        A pandas DataFrame with the historical data (date, open, high, low, close, volume),
        sorted by date ascending, or None if fetching fails.
    """
    # Use the historical daily price endpoint
    # FMP provides CSV download link for historical data which is often more reliable
    # Check FMP documentation for the exact endpoint structure
    # Example using a common pattern (verify against current FMP docs):
    # url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}"

    # Using the CSV endpoint for potentially more data
    # This often requires specific date ranges or limits depending on your FMP plan
    # Let's try fetching a reasonable amount of daily data first, e.g., last 5 years
    # For daily data, sometimes just the symbol is enough for the endpoint
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Check if the response content is valid JSON or potentially CSV
        content_type = response.headers.get('content-type')

        if 'application/json' in content_type:
            data = response.json()
            # FMP often wraps historical data in a 'historical' key
            if isinstance(data, dict) and 'historical' in data:
                historical_data = data['historical']
            elif isinstance(data, list):
                # Sometimes the list is directly returned
                historical_data = data
            else:
                print(f"Unexpected JSON structure for {symbol}: {data}")
                return None

            if not historical_data:
                print(f"No historical data found for {symbol} in JSON response.")
                return None
            df = pd.DataFrame(historical_data)

        else:
             # Attempt to read as CSV if not JSON (some FMP endpoints return CSV)
             # This might need adjustment based on the specific endpoint's CSV format
             csv_data = StringIO(response.text)
             df = pd.read_csv(csv_data)
             if df.empty:
                 print(f"No historical data found for {symbol} in CSV response.")
                 return None

        # Standardize columns (FMP column names can vary slightly)
        df = df.rename(columns={'date': 'Date', 'adjClose': 'Adj Close', 
                                'open': 'Open', 'high': 'High', 'low': 'Low', 
                                'close': 'Close', 'volume': 'Volume'})

        # Ensure 'Date' column exists and is datetime
        if 'Date' not in df.columns:
            print(f"'Date' column not found in data for {symbol}.")
            return None
        df['Date'] = pd.to_datetime(df['Date'])

        # Select relevant columns and sort
        # Ensure all expected columns are present after potential rename/fetch variations
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        # Keep only columns that actually exist in the DataFrame
        cols_to_keep = [col for col in required_cols if col in df.columns]
        if 'Date' not in cols_to_keep:
             print(f"'Date' column missing after processing for {symbol}.")
             return None # Cannot proceed without Date

        df = df[cols_to_keep]
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.set_index('Date') # Set Date as index for time series analysis

        print(f"Successfully fetched {len(df)} data points for {symbol}.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol} from FMP: {e}")
        return None
    except Exception as e:
        print(f"An error occurred processing data for {symbol}: {e}")
        return None 