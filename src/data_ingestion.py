import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles data fetching, cleaning, and preparation for multi-asset analysis
    """
    
    def __init__(
        self,
        tickers: List[str],
        start_date: str = "2018-01-01",
        end_date: str = "2025-01-01"
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None
        self.clean_data = None
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data for all tickers using yfinance
        Returns long-format DataFrame
        """
        logger.info(f"Fetching data for {len(self.tickers)} tickers...")
        
        try:
            # Fetch all tickers in one API call
            # This returns a wide DataFrame with MultiIndex columns
            df_wide = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            if df_wide.empty:
                raise ValueError("No data returned from yfinance.")

            df_long = df_wide.stack()

            # Name the new index level for clarity
            df_long.index.names = ['Date', 'ticker']
            
            # Reset the index to turn 'Date' and 'ticker' into columns
            df_long = df_long.reset_index()

            # Rename 'Date' to 'date' to match the rest of the code
            df_long = df_long.rename(columns={'Date': 'date'})
            
            # Filter out any rows where all price/volume data is missing
            crit_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            cols_to_check = [col for col in crit_cols if col in df_long.columns]
            df_long = df_long.dropna(subset=cols_to_check, how='all')

            self.raw_data = df_long
            logger.info(f"Raw data shape: {self.raw_data.shape}")
            logger.info(f"Raw data columns: {list(self.raw_data.columns)}")
            return self.raw_data

        except Exception as e:
            logger.error(f"✗ Failed to fetch data: {str(e)}")
            raise e
    
    def clean_and_align(self) -> pd.DataFrame:
        """
        Clean data and align trading days across all tickers
        """
        if self.raw_data is None:
            raise ValueError("Must fetch data first")
        
        df = self.raw_data.copy()
        
        # Standardize column names
        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        # Convert Date to datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop_duplicates(subset=['date', 'ticker'], keep='first')
        
        # Filter out weekends and ensure valid trading days
        df = df[df['date'].dt.dayofweek < 5]
        
        # Handle missing values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # Forward fill within each ticker group
                df[col] = df.groupby('ticker')[col].fillna(method='ffill')
        
        # Remove rows with remaining NaN in critical columns
        df = df.dropna(subset=['close', 'volume'])
        
        # Ensure volume is non-negative
        df['volume'] = df['volume'].abs()
        
        # Create MultiIndex
        df = df.set_index(['date', 'ticker']).sort_index()
        
        self.clean_data = df
        logger.info(f"Clean data shape: {df.shape}")
        logger.info(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
        logger.info(f"Tickers: {df.index.get_level_values('ticker').nunique()}")
        
        return self.clean_data
    
    def compute_returns(self) -> pd.DataFrame:
        """
        Compute daily returns and log returns
        """
        if self.clean_data is None:
            raise ValueError("Must clean data first")
        
        df = self.clean_data.copy()
        
        # Compute returns per ticker
        df['return'] = df.groupby(level='ticker')['close'].pct_change()
        df['log_return'] = np.log(1 + df['return'])
        
        df['target'] = df.groupby(level='ticker')['return'].shift(-1)
        
        # Drop NaNs created by pct_change() and shift()
        df = df.dropna(subset=['return', 'log_return', 'target'])
        
        logger.info(f"Returns computed. Final shape: {df.shape}")
        logger.info(f"Return stats:\n{df['return'].describe()}")
        
        self.clean_data = df
        return df
    
    def get_panel_data(self) -> pd.DataFrame:
        """
        Complete pipeline: fetch → clean → compute returns
        """
        self.fetch_data()
        self.clean_and_align()
        self.compute_returns()
        return self.clean_data


def get_default_tickers() -> List[str]:
    """
    Returns a diversified list of large-cap tickers across sectors
    """
    return [
        # Technology
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AMD', 'INTC', 'CRM', 'ADBE',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO',
        # Consumer
        'AMZN', 'WMT', 'HD', 'NKE', 'COST',
        # Energy & Industrial
        'XOM', 'CVX', 'BA', 'CAT', 'GE',
        # Communication & Media
        'DIS', 'NFLX', 'CMCSA', 'T', 'VZ'
    ]


if __name__ == "__main__":
    tickers = get_default_tickers()
    
    ingestion = DataIngestion(
        tickers=tickers,
        start_date="2018-01-01",
        end_date="2025-01-01"
    )
    
    panel_data = ingestion.get_panel_data()
    
    print(f"Shape: {panel_data.shape}")
    print(f"\nColumns: {list(panel_data.columns)}")
    print(f"\nFirst few rows:\n{panel_data.head(10)}")
    print(f"\nData types:\n{panel_data.dtypes}")