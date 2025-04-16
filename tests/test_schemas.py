import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pydantic import ValidationError
from app.schemas.forecast import (
    DateRange,
    ModelParams,
    ForecastDataPoint,
    RiskMetrics,
    ForecastResponse,
    ForecastParams,
)

def test_forecast_data_point():
    """Test ForecastDataPoint handles NaN values properly"""
    # Test with valid data
    point = ForecastDataPoint(
        Date="2023-01-01",
        Forecasted_Annualized_Volatility=15.5,
        CI_Lower=10.2,
        CI_Upper=20.8
    )
    assert point.Forecasted_Annualized_Volatility == 15.5
    
    # Test with NaN values
    point = ForecastDataPoint(
        Date="2023-01-01",
        Forecasted_Annualized_Volatility=np.nan,
        CI_Lower=np.nan,
        CI_Upper=np.nan
    )
    assert point.Forecasted_Annualized_Volatility == 0.0
    assert point.CI_Lower == 0.0
    assert point.CI_Upper == 0.0
    
    # Test with None values
    point = ForecastDataPoint(
        Date="2023-01-01",
        Forecasted_Annualized_Volatility=None,
        CI_Lower=None,
        CI_Upper=None
    )
    assert point.Forecasted_Annualized_Volatility == 0.0

def test_risk_metrics():
    """Test RiskMetrics handles NaN and None values"""
    # Test with valid data
    metrics = RiskMetrics(
        VaR_95=5.2,
        ES_95=7.8,
        RMSE=0.05
    )
    assert metrics.VaR_95 == 5.2
    
    # Test with NaN values
    metrics = RiskMetrics(
        VaR_95=np.nan,
        ES_95=np.nan,
        RMSE=np.nan
    )
    assert metrics.VaR_95 is None
    assert metrics.ES_95 is None
    assert metrics.RMSE is None
    
    # Test with None values
    metrics = RiskMetrics(
        VaR_95=None,
        ES_95=None,
        RMSE=None
    )
    assert metrics.VaR_95 is None

def test_forecast_params():
    """Test ForecastParams validation"""
    # Test default values
    params = ForecastParams()
    assert params.horizon == 10
    assert params.p == "1"
    
    # Test parameter cleaning - our validator converts "2,1" to "21"
    params = ForecastParams(p="2,1")
    assert params.p == "21"
    
    # Test date validation
    params = ForecastParams(start_date="2023-01-01")
    assert params.start_date == "2023-01-01"
    
    # Test invalid date
    with pytest.raises(ValidationError):
        ForecastParams(start_date="01-01-2023")

def test_date_range():
    """Test DateRange model"""
    date_range = DateRange(start="2023-01-01", end="2023-12-31")
    assert date_range.start == "2023-01-01"
    assert date_range.end == "2023-12-31"

def test_model_params():
    """Test ModelParams validation"""
    # Test default values
    params = ModelParams()
    assert params.p == 1
    assert params.q == 1
    assert params.vol_model == "Garch"
    
    # Test valid values
    params = ModelParams(p=2, q=3, vol_model="EGARCH")
    assert params.p == 2
    assert params.q == 3
    assert params.vol_model == "EGARCH"
    
    # Test that values out of range raise validation errors
    with pytest.raises(ValidationError):
        ModelParams(p=5)  # p must be <= 3

def test_full_forecast_response():
    """Test complete ForecastResponse"""
    # Create components
    date_range = DateRange(start="2023-01-01", end="2023-12-31")
    risk_metrics = RiskMetrics(VaR_95=5.2, ES_95=7.8, RMSE=0.05)
    forecast_points = [
        ForecastDataPoint(
            Date="2023-01-01",
            Forecasted_Annualized_Volatility=15.5,
            CI_Lower=10.2,
            CI_Upper=20.8
        ),
        ForecastDataPoint(
            Date="2023-01-02",
            Forecasted_Annualized_Volatility=16.2,
            CI_Lower=11.0,
            CI_Upper=21.4
        )
    ]
    
    # Create full response
    response = ForecastResponse(
        symbol="AAPL",
        forecast_horizon=10,
        model="Garch(1,1) with Normal distribution",
        date_range=date_range,
        forecast=forecast_points,
        risk_metrics=risk_metrics
    )
    
    # Test values
    assert response.symbol == "AAPL"
    assert response.forecast_horizon == 10
    assert len(response.forecast) == 2
    assert response.forecast[0].Forecasted_Annualized_Volatility == 15.5
    assert response.risk_metrics.VaR_95 == 5.2


# Add a test for JSON serialization/deserialization
def test_json_serialization():
    """Test that models can be properly serialized to and from JSON"""
    # Create a forecast response
    date_range = DateRange(start="2023-01-01", end="2023-12-31")
    risk_metrics = RiskMetrics(VaR_95=5.2, ES_95=7.8, RMSE=0.05)
    forecast_points = [
        ForecastDataPoint(
            Date="2023-01-01",
            Forecasted_Annualized_Volatility=15.5,
            CI_Lower=10.2,
            CI_Upper=20.8
        )
    ]
    
    response = ForecastResponse(
        symbol="AAPL",
        forecast_horizon=10,
        model="Garch(1,1) with Normal distribution",
        date_range=date_range,
        forecast=forecast_points,
        risk_metrics=risk_metrics
    )
    
    # Serialize to JSON
    json_data = response.model_dump_json()
    
    # Deserialize from JSON
    reconstructed = ForecastResponse.model_validate_json(json_data)
    
    # Verify the data survived the round trip
    assert reconstructed.symbol == "AAPL"
    assert reconstructed.forecast_horizon == 10
    assert reconstructed.forecast[0].Forecasted_Annualized_Volatility == 15.5
    assert reconstructed.risk_metrics.VaR_95 == 5.2 