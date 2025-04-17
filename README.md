# GARCH Volatility Forecasting API

A FastAPI-based web service for GARCH model forecasting, backtesting, and comparison.

## Features

- **Volatility Forecasting**: Generate multi-horizon GARCH forecasts with confidence intervals
- **Model Selection**: Auto-select optimal model specifications or specify custom parameters
- **Backtesting**: Walk-forward validation of forecasting performance with comprehensive metrics
- **Model Comparison**: Compare multiple GARCH specifications side-by-side
- **Interactive Visualizations**: Chart.js powered visualizations of forecasts, backtests and historical volatility
- **JSON & HTML Endpoints**: Get data in machine-readable format or as interactive charts

## Model Types Supported

- GARCH: Standard symmetric model
- EGARCH: Exponential GARCH for leverage effects
- GJR-GARCH: Glosten-Jagannathan-Runkle GARCH for asymmetric effects

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd garch_fast_api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   FMP_API_KEY=your_financial_modeling_prep_api_key
   ```

## Running the Application

Start the server with:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Key Endpoints

- `/forecast/{symbol}`: GET - JSON forecast for a stock symbol
- `/forecast/chart/{symbol}`: GET - HTML chart of the forecast
- `/forecast/backtest/{symbol}`: GET - Backtest results with comprehensive metrics
- `/forecast/backtest/chart/{symbol}`: GET - Visual representation of backtest results
- `/forecast/compare-models/{symbol}`: GET - Compare different model specifications
- `/forecast/horizon-analysis/{symbol}`: GET - Multi-horizon forecasting analysis

## Architecture

### Data Schemas
The application uses Pydantic for data validation and serialization:

- `ForecastResponse`: Complete forecast output with confidence intervals
- `ForecastDataPoint`: Individual forecast data points
- `RiskMetrics`: VaR, ES, and RMSE calculations
- `ModelParams`: GARCH model parameters with validation
- `BacktestResponse`: Complete backtest results with accuracy metrics
- `DateRange`: Date range validation for queries

See the detailed schema documentation in `app/schemas/README.md`.

### Components

- **API Endpoints**: FastAPI routing in `app/routers/`
- **GARCH Models**: Volatility modeling in `app/services/garch_service.py`
- **Data Access**: Market data retrieval in `app/services/db_service.py`
- **HTML Templates**: Jinja2 templates in `templates/`
- **Database**: SQLite with SQLAlchemy ORM in `app/models/`

## Backtesting

The backtesting feature performs walk-forward testing of the GARCH model:

- Trains models on rolling windows of historical data
- Forecasts volatility for specified horizons
- Evaluates forecasts against realized volatility
- Calculates performance metrics:
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error) 
  - VaR Hit Rate
  - VaR Accuracy

## Testing

Run tests with:

```bash
python -m pytest
```

Test coverage report:

```bash
python -m pytest --cov=app
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 