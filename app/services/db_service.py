from sqlalchemy.orm import Session
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from app.models.stock_data import StockData
from app.services import data_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_or_fetch_stock_data(db: Session, symbol: str, api_key: str, max_age_days: int = 1) -> Optional[pd.DataFrame]:
    """
    Get stock data from the database if available and recent,
    otherwise fetch from API and store in the database.
    
    Args:
        db: Database session
        symbol: Stock symbol
        api_key: API key for FMP
        max_age_days: Maximum age of data in days before refetching
        
    Returns:
        DataFrame with stock data or None if not available
    """
    # Check if we have recent data in the database
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    
    # Count recent records for this symbol
    recent_count = db.query(StockData).filter(
        StockData.symbol == symbol,
        StockData.updated_at >= cutoff_date
    ).count()
    
    # If we have recent data, return it from the database
    if recent_count > 0:
        logger.info(f"Found recent data for {symbol} in database, using cached data")
        stock_data = db.query(StockData).filter(
            StockData.symbol == symbol
        ).order_by(StockData.date).all()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'Date': entry.date,
            'Open': entry.open,
            'High': entry.high,
            'Low': entry.low,
            'Close': entry.close,
            'Volume': entry.volume
        } for entry in stock_data])
        
        # Set Date as index for time series analysis
        if not df.empty:
            df = df.set_index('Date')
            return df
    
    # If no recent data, fetch from API
    logger.info(f"No recent data found for {symbol}, fetching from API")
    df = data_service.fetch_data(symbol, api_key)
    
    # If API fetch successful, store in database
    if df is not None and not df.empty:
        save_stock_data_to_db(db, symbol, df)
        return df
    
    return None

def save_stock_data_to_db(db: Session, symbol: str, df: pd.DataFrame) -> None:
    """
    Save stock data DataFrame to the database.
    
    Args:
        db: Database session
        symbol: Stock symbol
        df: DataFrame with stock data (with Date as index)
    """
    # Reset index to get Date as a column
    df_to_save = df.reset_index()
    
    try:
        # Begin transaction
        for _, row in df_to_save.iterrows():
            # Check if this record already exists
            existing = db.query(StockData).filter(
                StockData.symbol == symbol,
                StockData.date == row['Date']
            ).first()
            
            if existing:
                # Update existing record
                existing.open = row.get('Open')
                existing.high = row.get('High')
                existing.low = row.get('Low')
                existing.close = row.get('Close')
                existing.volume = row.get('Volume')
                existing.updated_at = datetime.now()
            else:
                # Create new record
                stock_data = StockData(
                    symbol=symbol,
                    date=row['Date'],
                    open=row.get('Open'),
                    high=row.get('High'),
                    low=row.get('Low'),
                    close=row.get('Close'),
                    volume=row.get('Volume')
                )
                db.add(stock_data)
        
        # Commit all changes
        db.commit()
        logger.info(f"Successfully saved {len(df_to_save)} records for {symbol} to database")
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving stock data to database: {e}")
        raise 