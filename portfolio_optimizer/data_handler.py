import pandas as pd
import yfinance as yf
import numpy as np
import streamlit as st

class DataHandler:
    """
    Handles fetching and processing of financial data for portfolio optimization.
    """

    # --- Benchmark Data Loading ---
    def fetch_nifty50_composition(self) -> pd.DataFrame:
        """
        Loads the NIFTY 50 composition and weights from a local CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing 'Symbol' and 'Weightage' for NIFTY 50 stocks.
        """
        try:
            df = pd.read_csv('data/nifty50.csv')
            # Ensure correct columns and normalization
            df['Weightage'] = df['Weightage'].astype(float)
            if df['Weightage'].max() > 1.0:
                df['Weightage'] = df['Weightage'] / 100.0
            nifty50 = df[['Symbol', 'Weightage']].copy()
            print("Successfully loaded benchmark data from CSV.")
            return nifty50
        except Exception as e:
            print(f"Error loading benchmark data from CSV: {e}")
            return pd.DataFrame()

    # --- Historical Price Fetching ---
    def fetch_historical_prices(self, tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical closing prices for a list of tickers from Yahoo Finance.

        Args:
            tickers (list): List of stock tickers.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame of historical closing prices.
        """
        print(f"Fetching historical prices for {len(tickers)} tickers...")
        prices = yf.download(tickers, start=start_date, end=end_date)['Close']
        prices.dropna(axis=1, how='all', inplace=True)
        print("Successfully fetched price data.")
        return prices

    # --- Statistics Calculation ---
    def calculate_statistics(self, prices: pd.DataFrame):
        """
        Calculates annualized expected returns and the covariance matrix.

        Args:
            prices (pd.DataFrame): DataFrame of historical prices.

        Returns:
            tuple: A tuple containing:
                - pd.Series: Annualized expected returns (mu).
                - pd.DataFrame: Annualized covariance matrix (cov).
        """
        returns = prices.pct_change().dropna()
        # Annualize returns (assuming 252 trading days)
        mu = returns.mean() * 252
        # Annualize covariance
        cov = returns.cov() * 252
        # Ensure the covariance matrix is positive semi-definite
        cov = (cov + cov.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(cov))
        if min_eig < 0:
            cov -= 1.001 * min_eig * np.eye(len(cov))
        print("Calculated statistical metrics (mu, cov).")
        return mu, cov

# Streamlit integration example (to be placed in the Streamlit app file, not here)
data_handler = DataHandler()
tickers = ["AAPL", "MSFT", "GOOGL"]
start_date = "2020-01-01"
end_date = "2023-01-01"
prices = data_handler.fetch_historical_prices(tickers, start_date, end_date)
st.write("Historical Closing Prices:")
st.dataframe(prices)
