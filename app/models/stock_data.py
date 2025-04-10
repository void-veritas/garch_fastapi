from sqlalchemy import Column, Integer, Float, String, Date, UniqueConstraint
from sqlalchemy.sql import func
from datetime import datetime

from app.models.base import Base

class StockData(Base):
    """Model for storing historical stock price data."""
    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Integer)
    created_at = Column(Date, default=datetime.now)
    updated_at = Column(Date, default=datetime.now, onupdate=datetime.now)

    # Ensure each symbol has unique dates
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uix_symbol_date'),
    )

    def __repr__(self):
        return f"<StockData(symbol='{self.symbol}', date='{self.date}', close={self.close})>" 