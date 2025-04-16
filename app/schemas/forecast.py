from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator, model_validator
from datetime import datetime, date
import pandas as pd
import numpy as np

class DateRange(BaseModel):
    """Date range for data selection."""
    start: str
    end: str

class ModelParams(BaseModel):
    """GARCH model parameters."""
    p: int = Field(1, ge=1, le=3, description="ARCH parameter")
    q: int = Field(1, ge=1, le=3, description="GARCH parameter") 
    vol_model: Literal["Garch", "EGARCH", "GJR"] = Field("Garch", description="Volatility model type")
    dist: Literal["Normal", "StudentsT", "SkewStudent"] = Field("Normal", description="Error distribution")
    auto_select: bool = Field(False, description="Auto-select the best model")
    
    @validator('p', 'q')
    def validate_params(cls, v, values):
        if v < 1 or v > 3:
            return 1
        return v

class ForecastDataPoint(BaseModel):
    """Individual forecast data point."""
    Date: str
    Forecasted_Annualized_Volatility: float
    CI_Lower: float
    CI_Upper: float
    
    @validator('Forecasted_Annualized_Volatility', 'CI_Lower', 'CI_Upper', pre=True)
    def handle_nan(cls, v):
        """Handle NaN/None values."""
        if pd.isna(v) or v is None:
            return 0.0
        return round(float(v), 2)

class RiskMetrics(BaseModel):
    """Risk metrics for the forecast."""
    VaR_95: Optional[float] = None
    ES_95: Optional[float] = None
    RMSE: Optional[float] = None
    
    @model_validator(mode='before')
    @classmethod
    def handle_nan_values(cls, data):
        """Handle NaN values in risk metrics."""
        if isinstance(data, dict):
            for key, value in data.items():
                if pd.isna(value) or value is None:
                    data[key] = None
                elif isinstance(value, (int, float)):
                    data[key] = round(value, 4)
        return data

class ForecastResponse(BaseModel):
    """Response model for the GARCH forecast."""
    symbol: str
    forecast_horizon: int = Field(..., ge=1, le=30)
    model: str
    date_range: DateRange
    forecast: List[ForecastDataPoint]
    risk_metrics: RiskMetrics

class ForecastParams(BaseModel):
    """Parameters for the forecast endpoint."""
    horizon: int = Field(10, ge=1, le=30, description="Forecast horizon in days (1-30)")
    p: Optional[str] = Field("1", description="ARCH parameter (p)")
    q: Optional[str] = Field("1", description="GARCH parameter (q)")
    vol_model: str = Field("Garch", description="Volatility model (Garch, EGARCH, GJR)")
    distribution: str = Field("Normal", description="Error distribution (Normal, StudentsT, SkewStudent)")
    auto_select: bool = Field(False, description="Auto-select the best model specification")
    start_date: Optional[str] = Field(None, description="Start date for the data (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for the data (YYYY-MM-DD)")
    
    @validator('p', 'q')
    def validate_params(cls, v):
        """Validate and convert p and q parameters, but keep as string."""
        if v is None:
            return "1"
        try:
            # Remove any commas, preserving the resulting string
            v_clean = v.replace(',', '')
            return v_clean
        except (ValueError, AttributeError):
            return "1"
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        """Validate date format."""
        if v is None:
            return v
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Date must be in format YYYY-MM-DD")

# Models for backtest

class BacktestDataPoint(BaseModel):
    """Individual backtest data point."""
    Date: str
    Forecasted_Volatility: float
    Realized_Volatility: float
    Error: float
    VaR: Optional[float] = None
    ES: Optional[float] = None
    
    @validator('Forecasted_Volatility', 'Realized_Volatility', 'Error', 'VaR', 'ES', pre=True)
    def handle_nan(cls, v):
        """Handle NaN/None values."""
        if pd.isna(v) or v is None:
            return 0.0
        return round(float(v), 4)

class BacktestMetrics(BaseModel):
    """Metrics for the backtest."""
    RMSE: float
    MAE: float
    VaR_Hit_Rate: float
    VaR_Accuracy: float
    
    @validator('RMSE', 'MAE', 'VaR_Hit_Rate', 'VaR_Accuracy', pre=True)
    def handle_nan(cls, v):
        """Handle NaN/None values."""
        if pd.isna(v) or v is None:
            return 0.0
        return round(float(v), 4)

class BacktestResponse(BaseModel):
    """Response model for the GARCH backtest."""
    symbol: str
    model: str
    date_range: DateRange
    metrics: BacktestMetrics
    backtest_data: List[BacktestDataPoint]

# Models for model comparison

class ModelData(BaseModel):
    """Data for an individual model in comparison."""
    label: str
    aic: float
    bic: float
    is_best: int
    
    @validator('aic', 'bic', pre=True)
    def handle_nan(cls, v):
        """Handle NaN/None values."""
        if pd.isna(v) or v is None:
            return 0.0
        return round(float(v), 2)

class ForecastComparisonData(BaseModel):
    """Data for forecast comparison."""
    model_label: str
    values: List[float]
    is_best: int
    
    @validator('values', pre=True)
    def handle_nan_list(cls, v):
        """Handle NaN/None values in lists."""
        if v is None:
            return []
        return [0.0 if pd.isna(x) else round(float(x), 2) for x in v]

class ModelComparisonData(BaseModel):
    """Data for model comparison."""
    dates: List[str]
    forecasts: List[ForecastComparisonData]

class ModelComparisonResponse(BaseModel):
    """Response model for model comparison."""
    symbol: str
    horizon: int
    training_days: int
    date_range: str
    best_model: Dict
    models: List[ModelData]
    models_data: str  # JSON string for chart data
    forecast_data: str  # JSON string for forecast data 