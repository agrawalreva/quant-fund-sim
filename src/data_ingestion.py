"""
Data ingestion module for quant fund simulation.

Handles data collection from Yahoo Finance, FRED, and other sources.
Includes data cleaning, validation, and storage functionality.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Main class for data ingestion and cleaning operations."""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize data ingestion class.
        
        Args:
            fred_api_key: FRED API key for economic data (optional)
        """
        self.fred_api_key = fred_api_key
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        
    def get_equity_data(self, 
                       symbols: List[str], 
                       start_date: str, 
                       end_date: str,
                       interval: str = '1d') -> pd.DataFrame:
        """
        Fetch equity data from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data for all symbols
        """
        logger.info(f"Fetching equity data for {len(symbols)} symbols")
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if not hist.empty:
                    # Rename columns to include symbol
                    hist.columns = [f"{symbol}_{col}" for col in hist.columns]
                    data[symbol] = hist
                    logger.info(f"Successfully fetched data for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                
        if not data:
            raise ValueError("No data could be fetched for any symbols")
            
        # Combine all data
        combined_data = pd.concat(data.values(), axis=1)
        combined_data.index.name = 'date'
        
        return combined_data
    
    def get_macro_data(self, 
                      series_ids: List[str], 
                      start_date: str, 
                      end_date: str) -> pd.DataFrame:
        """
        Fetch macroeconomic data from FRED.
        
        Args:
            series_ids: List of FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with macro data
        """
        if not self.fred:
            logger.warning("FRED API key not provided, skipping macro data")
            return pd.DataFrame()
            
        logger.info(f"Fetching macro data for {len(series_ids)} series")
        
        data = {}
        for series_id in series_ids:
            try:
                series_data = self.fred.get_series(
                    series_id, 
                    start=start_date, 
                    end=end_date
                )
                if not series_data.empty:
                    data[series_id] = series_data
                    logger.info(f"Successfully fetched {series_id}")
                else:
                    logger.warning(f"No data found for {series_id}")
                    
            except Exception as e:
                logger.error(f"Error fetching {series_id}: {str(e)}")
                
        if not data:
            logger.warning("No macro data could be fetched")
            return pd.DataFrame()
            
        return pd.DataFrame(data)
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate price data.
        
        Args:
            df: Raw price data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning price data")
        
        # Remove rows with all NaN values
        df_clean = df.dropna(how='all')
        
        # Forward fill missing values (common for stock data)
        df_clean = df_clean.fillna(method='ffill')
        
        # Remove any remaining NaN values
        df_clean = df_clean.dropna()
        
        # Remove duplicate dates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def calculate_returns(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            df: Price data DataFrame
            price_cols: List of price column names
            
        Returns:
            DataFrame with returns
        """
        logger.info("Calculating returns")
        
        returns_data = {}
        for col in price_cols:
            if col in df.columns:
                # Calculate simple returns
                returns_data[f"{col}_return"] = df[col].pct_change()
                # Calculate log returns
                returns_data[f"{col}_log_return"] = np.log(df[col] / df[col].shift(1))
        
        returns_df = pd.DataFrame(returns_data, index=df.index)
        returns_df = returns_df.dropna()
        
        logger.info(f"Returns data shape: {returns_df.shape}")
        return returns_df
    
    def get_benchmark_data(self, 
                          benchmark_symbol: str = '^GSPC',
                          start_date: str = '2020-01-01',
                          end_date: str = None) -> pd.DataFrame:
        """
        Get benchmark data (default: S&P 500).
        
        Args:
            benchmark_symbol: Benchmark symbol (default: S&P 500)
            start_date: Start date
            end_date: End date (default: today)
            
        Returns:
            DataFrame with benchmark data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching benchmark data for {benchmark_symbol}")
        
        benchmark_data = self.get_equity_data(
            [benchmark_symbol], 
            start_date, 
            end_date
        )
        
        return benchmark_data
    
    def create_features(self, df: pd.DataFrame, lookback_periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create technical features from price data.
        
        Args:
            df: Price data DataFrame
            lookback_periods: List of lookback periods for moving averages
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Creating technical features")
        
        features_df = df.copy()
        
        # Get price columns (assuming Close prices)
        price_cols = [col for col in df.columns if 'Close' in col]
        
        for col in price_cols:
            symbol = col.split('_')[0]
            
            # Moving averages
            for period in lookback_periods:
                features_df[f"{symbol}_MA_{period}"] = df[col].rolling(period).mean()
                features_df[f"{symbol}_MA_ratio_{period}"] = df[col] / features_df[f"{symbol}_MA_{period}"]
            
            # Volatility (rolling standard deviation)
            features_df[f"{symbol}_volatility_20"] = df[col].pct_change().rolling(20).std()
            
            # RSI (simplified)
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features_df[f"{symbol}_RSI"] = 100 - (100 / (1 + rs))
            
            # Price momentum
            features_df[f"{symbol}_momentum_5"] = df[col] / df[col].shift(5) - 1
            features_df[f"{symbol}_momentum_20"] = df[col] / df[col].shift(20) - 1
        
        # Remove rows with NaN values from feature creation
        features_df = features_df.dropna()
        
        logger.info(f"Features data shape: {features_df.shape}")
        return features_df


def main():
    """Example usage of DataIngestion class."""
    
    # Initialize data ingestion
    data_ingestion = DataIngestion()
    
    # Define symbols to fetch
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Fetch equity data
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    try:
        # Get equity data
        equity_data = data_ingestion.get_equity_data(symbols, start_date, end_date)
        
        # Clean the data
        clean_data = data_ingestion.clean_price_data(equity_data)
        
        # Create features
        features_data = data_ingestion.create_features(clean_data)
        
        # Calculate returns
        price_cols = [col for col in clean_data.columns if 'Close' in col]
        returns_data = data_ingestion.calculate_returns(clean_data, price_cols)
        
        # Get benchmark data
        benchmark_data = data_ingestion.get_benchmark_data(start_date=start_date, end_date=end_date)
        
        print("Data ingestion completed successfully!")
        print(f"Equity data shape: {equity_data.shape}")
        print(f"Features data shape: {features_data.shape}")
        print(f"Returns data shape: {returns_data.shape}")
        print(f"Benchmark data shape: {benchmark_data.shape}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
