# Data Schemas

This directory contains Pydantic models used for data validation, serialization, and documentation.

## Overview

Pydantic models provide several benefits in our API:

1. **Input Validation**: Automatically validates and sanitizes request parameters
2. **Type Safety**: Ensures data conforms to expected types and constraints
3. **Documentation**: Automatically generates OpenAPI/Swagger documentation
4. **Serialization**: Handles conversion between Python objects and JSON
5. **Error Handling**: Produces clear error messages for invalid data

## Available Models

### Core Models

- `DateRange`: Simple start/end date structure
- `ModelParams`: GARCH model parameters with validation
- `ForecastDataPoint`: Individual forecast point with NaN handling
- `RiskMetrics`: Risk measurements (VaR, ES, RMSE)

### Response Models

- `ForecastResponse`: Complete forecast with metadata
- `BacktestResponse`: Backtest results with metrics
- `ModelComparisonResponse`: Model comparison results

### Request Models

- `ForecastParams`: Parameters for forecast endpoint with validation

## Key Features

### Robust Validation

All models include validation:

- Date format validation
- Range constraints (e.g., p, q parameters between 1-3)
- Type validation (e.g., ensuring numeric inputs)

### NaN Handling

Special handling for NaN/None values:

```python
@validator('values', pre=True)
def handle_nan_list(cls, v):
    """Handle NaN/None values in lists."""
    if v is None:
        return []
    return [0.0 if pd.isna(x) else round(float(x), 2) for x in v]
```

### Model Nesting

Models are composed hierarchically:

```python
class ForecastResponse(BaseModel):
    """Response model for the GARCH forecast."""
    symbol: str
    forecast_horizon: int = Field(..., ge=1, le=30)
    model: str
    date_range: DateRange
    forecast: List[ForecastDataPoint]
    risk_metrics: RiskMetrics
```

## Testing

Schema models have comprehensive tests in `/tests/test_schemas.py` covering:

1. Default value handling
2. NaN/None value handling
3. Validation rules
4. JSON serialization/deserialization 