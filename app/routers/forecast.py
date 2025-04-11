from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from app.services import garch_service, data_service, db_service
# Import Settings, get_settings, and templates from config
from app.config import Settings, get_settings, templates, get_db
import pandas as pd
import json
import numpy as np # Need numpy for sqrt
from typing import Optional, List
from datetime import datetime, timedelta

# Removed sys path modification and import from main
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent.parent))
# from main import templates

router = APIRouter(prefix="/forecast", tags=["Forecast", "Chart", "Backtest"])

# Pydantic model for request body if needed later
# class ForecastRequest(BaseModel):
#     symbol: str
#     horizon: int = 1

# Enhanced parameters with proper model configuration
@router.get("/{symbol}", 
            response_model=None, 
            summary="Get GARCH Forecast (JSON)") 
def get_garch_forecast_json(
    symbol: str,
    horizon: int = Query(default=10, ge=1, le=30, description="Forecast horizon in days (1-30)"),
    p: int = Query(default=1, ge=1, le=3, description="ARCH parameter (p)"),
    q: int = Query(default=1, ge=1, le=3, description="GARCH parameter (q)"),
    vol_model: str = Query(default="Garch", description="Volatility model (Garch, EGARCH, GJR)"),
    distribution: str = Query(default="Normal", description="Error distribution (Normal, StudentsT, SkewStudent)"),
    auto_select: bool = Query(default=False, description="Auto-select the best model specification"),
    start_date: Optional[str] = Query(None, description="Start date for the data (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for the data (YYYY-MM-DD)"),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db)
):
    """Endpoint to get GARCH volatility forecast for a symbol (returns JSON)."""
    try:
        # 1. Fetch data
        print(f"Fetching data for {symbol} (JSON endpoint)...")
        df_prices = db_service.get_or_fetch_stock_data(db, symbol, settings.fmp_api_key)
        if df_prices is None or df_prices.empty:
            raise HTTPException(status_code=404,
                                detail=f"Could not fetch valid historical data for {symbol} from FMP.")
        if 'Close' not in df_prices.columns:
             raise HTTPException(status_code=500, detail="'Close' column missing in fetched data.")

        # Filter by date range if provided
        if start_date:
            df_prices = df_prices.loc[start_date:]
        if end_date:
            df_prices = df_prices.loc[:end_date]
            
        if df_prices.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol} in the specified date range.")

        # 2. Calculate Returns
        print(f"Calculating returns for {symbol}...")
        returns = garch_service.calculate_returns(df_prices['Close'])
        if returns.empty:
             raise HTTPException(status_code=500, detail="Failed to calculate returns. Insufficient data?")

        # 3. Fit GARCH model - either auto-select or use specified parameters
        if auto_select:
            print(f"Auto-selecting best GARCH model for {symbol}...")
            model_result = garch_service.model_selection(returns, max_p=2, max_q=2,
                                                     vol_models=['Garch', 'GJR', 'EGARCH'],
                                                     distributions=['Normal', 'StudentsT'])
            model_fit = model_result['model']
            model_params = model_result['params']
            model_description = f"{model_params['vol_model']}({model_params['p']},{model_params['q']}) with {model_params['dist']} distribution"
        else:
            print(f"Fitting {vol_model}({p},{q}) model with {distribution} distribution for {symbol}...")
            model_fit = garch_service.fit_garch_model(returns, p=p, q=q, vol_model=vol_model, dist=distribution)
            model_description = f"{vol_model}({p},{q}) with {distribution} distribution"
            
        if model_fit is None:
             raise HTTPException(status_code=500, detail=f"Failed to fit GARCH model for {symbol}.")

        # 4. Forecast volatility with confidence intervals
        print(f"Forecasting volatility for {symbol} (horizon={horizon})...")
        forecast_df = garch_service.forecast_volatility(model_fit, horizon=horizon, alpha=0.05)
        if forecast_df is None or forecast_df.empty:
            raise HTTPException(status_code=500, detail=f"Failed to generate volatility forecast for {symbol}.")

        # 5. Calculate risk metrics for the forecast
        risk_metrics = garch_service.calculate_risk_metrics(
            forecast_df['Forecasted_Annualized_Volatility'],
            returns.iloc[-252:],  # Use last year of returns for metrics
            confidence_level=0.95
        )

        # 6. Format and return the forecast as JSON
        forecast_list = forecast_df.reset_index().to_dict(orient='records')
        for record in forecast_list:
            record['Date'] = record['Date'].strftime('%Y-%m-%d')
            
        print(f"Successfully generated JSON forecast for {symbol}.")
        return {
            "symbol": symbol, 
            "forecast_horizon": horizon, 
            "model": model_description,
            "date_range": {
                "start": start_date or df_prices.index[0].strftime('%Y-%m-%d'),
                "end": end_date or df_prices.index[-1].strftime('%Y-%m-%d')
            },
            "forecast": forecast_list,
            "risk_metrics": {
                "VaR_95": float(risk_metrics['VaR'].iloc[0]),
                "ES_95": float(risk_metrics['ES'].iloc[0]),
                "RMSE": float(risk_metrics['RMSE'])
            }
        }

    except HTTPException as http_exc:
        print(f"HTTP Exception for {symbol} (JSON): {http_exc.detail}")
        raise http_exc
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred for {symbol} (JSON): {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.get("/chart/{symbol}", 
            response_class=HTMLResponse, 
            summary="Display GARCH Forecast Chart (HTML)")
async def get_garch_forecast_chart(
    request: Request,
    symbol: str,
    horizon: int = Query(default=10, ge=1, le=30, description="Forecast horizon in days (1-30)"),
    p: int = Query(default=1, ge=1, le=3, description="ARCH parameter (p)"),
    q: int = Query(default=1, ge=1, le=3, description="GARCH parameter (q)"),
    vol_model: str = Query(default="Garch", description="Volatility model (Garch, EGARCH, GJR)"),
    distribution: str = Query(default="Normal", description="Error distribution (Normal, StudentsT, SkewStudent)"),
    auto_select: bool = Query(default=False, description="Auto-select the best model specification"),
    start_date: Optional[str] = Query(None, description="Start date for the data (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for the data (YYYY-MM-DD)"),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db)
):
    """Endpoint to display a chart of the GARCH volatility forecast with confidence intervals."""
    try:
        # 1. Fetch data
        print(f"Fetching data for {symbol} (Chart endpoint)...")
        df_prices = db_service.get_or_fetch_stock_data(db, symbol, settings.fmp_api_key)
        if df_prices is None or df_prices.empty:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"Could not fetch data for {symbol}"}, status_code=404)
        if 'Close' not in df_prices.columns:
             return templates.TemplateResponse("error.html", {"request": request, "detail": "'Close' column missing"}, status_code=500)

        # Filter by date range if provided
        if start_date:
            df_prices = df_prices.loc[start_date:]
        if end_date:
            df_prices = df_prices.loc[:end_date]
            
        if df_prices.empty:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"No data available for {symbol} in the specified date range"}, status_code=404)

        # 2. Calculate Returns
        print(f"Calculating returns for {symbol}...")
        returns = garch_service.calculate_returns(df_prices['Close'])
        if returns.empty:
            return templates.TemplateResponse("error.html", {"request": request, "detail": "Failed to calculate returns"}, status_code=500)

        # 3. Fit GARCH model - either auto-select or use specified parameters
        if auto_select:
            print(f"Auto-selecting best GARCH model for {symbol}...")
            model_result = garch_service.model_selection(returns, max_p=2, max_q=2,
                                                     vol_models=['Garch', 'GJR', 'EGARCH'],
                                                     distributions=['Normal', 'StudentsT'])
            model_fit = model_result['model']
            model_params = model_result['params']
            model_description = f"{model_params['vol_model']}({model_params['p']},{model_params['q']}) with {model_params['dist']} distribution"
            p = model_params['p']  # For subsequent volatility calculation
            q = model_params['q']
            vol_model = model_params['vol_model']
            distribution = model_params['dist']
        else:
            print(f"Fitting {vol_model}({p},{q}) model with {distribution} distribution for {symbol}...")
            model_fit = garch_service.fit_garch_model(returns, p=p, q=q, vol_model=vol_model, dist=distribution)
            model_description = f"{vol_model}({p},{q}) with {distribution} distribution"
            
        if model_fit is None:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"Failed to fit GARCH model for {symbol}"}, status_code=500)

        # 4. Forecast volatility with confidence intervals
        print(f"Forecasting volatility for {symbol} (horizon={horizon})...")
        forecast_df = garch_service.forecast_volatility(model_fit, horizon=horizon, alpha=0.05)
        if forecast_df is None or forecast_df.empty:
             return templates.TemplateResponse("error.html", {"request": request, "detail": f"Failed to generate forecast for {symbol}"}, status_code=500)

        # 5. Calculate risk metrics
        risk_metrics = garch_service.calculate_risk_metrics(
            forecast_df['Forecasted_Annualized_Volatility'],
            returns.iloc[-252:],
            confidence_level=0.95
        )

        # 6. Prepare data for the template
        
        # 6a. Get historical conditional volatility (annualized)
        hist_periods = 100 # Number of historical periods to show
        hist_daily_vol = model_fit.conditional_volatility.iloc[-hist_periods:]
        # Annualize by multiplying daily std deviation by sqrt(252)
        hist_vol_series = hist_daily_vol * np.sqrt(252)
        
        # Format historical data
        hist_df = pd.DataFrame({'Historical_Annualized_Volatility': hist_vol_series})
        hist_list = hist_df.reset_index().to_dict(orient='records')
        for record in hist_list:
            record['Date'] = record['Date'].strftime('%Y-%m-%d') 
        
        # 6b. Format forecast data with confidence intervals
        forecast_list = forecast_df.reset_index().to_dict(orient='records')
        for record in forecast_list:
            record['Date'] = record['Date'].strftime('%Y-%m-%d')
            # Round values for display
            record['Forecasted_Annualized_Volatility'] = round(record['Forecasted_Annualized_Volatility'], 2)
            record['CI_Lower'] = round(record['CI_Lower'], 2)
            record['CI_Upper'] = round(record['CI_Upper'], 2)
        
        # 6c. Format risk metrics
        var_value = round(float(risk_metrics['VaR'].iloc[0]), 4)
        es_value = round(float(risk_metrics['ES'].iloc[0]), 4)
        
        # 6d. Get date range info
        date_range_str = ""
        if start_date or end_date:
            start_str = start_date or df_prices.index[0].strftime('%Y-%m-%d')
            end_str = end_date or df_prices.index[-1].strftime('%Y-%m-%d')
            date_range_str = f"{start_str} to {end_str}"
        
        # 6e. Convert data to JSON strings for embedding
        hist_data_json = json.dumps(hist_list)
        forecast_data_json = json.dumps(forecast_list)

        print(f"Rendering chart for {symbol} with history and {model_description}.")
        # 7. Render the HTML template with all data
        return templates.TemplateResponse(
            "chart.html",
            {
                "request": request, 
                "symbol": symbol,
                "horizon": horizon,
                "model_description": model_description,
                "date_range": date_range_str,
                "hist_data": hist_data_json,
                "forecast_data": forecast_data_json,
                "var_95": var_value,
                "es_95": es_value
            }
        )

    except Exception as e:
        import traceback
        print(f"An unexpected error occurred for {symbol} (Chart): {e}")
        traceback.print_exc()
        try:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"An unexpected server error occurred: {str(e)}"}, status_code=500)
        except: 
             raise HTTPException(status_code=500, detail=f"An unexpected server error occurred and failed to load error page: {str(e)}") 

@router.get("/backtest/{symbol}", 
            response_class=HTMLResponse, 
            summary="Display GARCH Backtest Results Chart (HTML)")
async def get_garch_backtest_chart(
    request: Request,
    symbol: str,
    window: int = Query(default=252, ge=50, description="Rolling window size for backtest (min 50)"),
    p: int = Query(default=1, ge=1, le=3, description="ARCH parameter (p)"),
    q: int = Query(default=1, ge=1, le=3, description="GARCH parameter (q)"),
    vol_model: str = Query(default="Garch", description="Volatility model (Garch, EGARCH, GJR)"),
    distribution: str = Query(default="Normal", description="Error distribution (Normal, StudentsT, SkewStudent)"),
    auto_select: bool = Query(default=False, description="Auto-select model at each step"),
    start_date: Optional[str] = Query(None, description="Start date for the data (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for the data (YYYY-MM-DD)"),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db)
):
    """Endpoint to perform and display walk-forward GARCH backtest results with performance metrics."""
    try:
        # 1. Fetch data
        print(f"Fetching data for {symbol} (Backtest endpoint)...")
        df_prices = db_service.get_or_fetch_stock_data(db, symbol, settings.fmp_api_key)
        if df_prices is None or df_prices.empty:
             return templates.TemplateResponse("error.html", {"request": request, "detail": f"Could not fetch data for {symbol}"}, status_code=404)
        if 'Close' not in df_prices.columns:
             return templates.TemplateResponse("error.html", {"request": request, "detail": "'Close' column missing"}, status_code=500)

        # Filter by date range if provided
        if start_date:
            df_prices = df_prices.loc[start_date:]
        if end_date:
            df_prices = df_prices.loc[:end_date]
            
        if df_prices.empty:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"No data available for {symbol} in the specified date range"}, status_code=404)

        # 2. Run Backtest Service with enhanced parameters
        print(f"Running backtest for {symbol} with window {window} and {'auto-select' if auto_select else f'{vol_model}({p},{q})'} model...")
        backtest_results_df = garch_service.backtest_garch_forecast(
            df_prices['Close'], 
            window_size=window,
            p=p, 
            q=q,
            vol_model=vol_model,
            dist=distribution,
            auto_select=auto_select
        )

        if backtest_results_df is None or backtest_results_df.empty:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"Failed to generate backtest results for {symbol}. Not enough data or other error."}, status_code=500)
        
        # 3. Calculate performance metrics
        mse = np.mean(backtest_results_df['Error']**2)
        mae = np.mean(np.abs(backtest_results_df['Error']))
        rmse = np.sqrt(mse)
        
        # VaR performance - theoretical expectation is 5% breaches for 95% VaR
        if 'VaR_95' in backtest_results_df.columns:
            returns = garch_service.calculate_returns(df_prices['Close'])
            returns_for_backtesting = returns.loc[backtest_results_df.index]
            var_hits = (returns_for_backtesting < -backtest_results_df['VaR_95']).mean()
            var_accuracy = 1.0 - abs(var_hits - 0.05) / 0.05  # How close to expected 5% 
        else:
            var_hits = None
            var_accuracy = None
            
        # Model counts if auto-select was used
        if auto_select and 'Model_Used' in backtest_results_df.columns:
            model_counts = backtest_results_df['Model_Used'].value_counts().to_dict()
            top_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        else:
            top_models = [(f"{vol_model}({p},{q})-{distribution}", len(backtest_results_df))]
        
        # 4. Prepare data for template
        # Limit the data sent to the template if it's very long (e.g., last 2 years = ~500 points)
        backtest_results_df = backtest_results_df.iloc[-500:]

        backtest_list = backtest_results_df.reset_index().to_dict(orient='records')
        for record in backtest_list:
            record['Date'] = record['Date'].strftime('%Y-%m-%d')
            # Round numbers for display
            record['Forecasted_Volatility'] = round(record['Forecasted_Volatility'], 2)
            record['Realized_Volatility'] = round(record['Realized_Volatility'], 2)
            record['Error'] = round(record['Error'], 2)
            if 'VaR_95' in record:
                record['VaR_95'] = round(record['VaR_95'], 4)
            if 'ES_95' in record:
                record['ES_95'] = round(record['ES_95'], 4)
        
        # Get date range info
        date_range_str = ""
        if start_date or end_date:
            start_str = start_date or df_prices.index[0].strftime('%Y-%m-%d')
            end_str = end_date or df_prices.index[-1].strftime('%Y-%m-%d')
            date_range_str = f"{start_str} to {end_str}"
            
        # Convert to JSON for template
        backtest_data_json = json.dumps(backtest_list)
        model_info = ', '.join([f"{model}: {count} days" for model, count in top_models])
        
        # 5. Render the HTML template with backtest results and metrics
        return templates.TemplateResponse(
            "backtest.html",
            {
                "request": request,
                "symbol": symbol,
                "window_size": window,
                "auto_select": auto_select,
                "date_range": date_range_str,
                "model_info": model_info,
                "backtest_data": backtest_data_json,
                "metrics": {
                    "mse": round(mse, 4),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "var_hits": round(var_hits * 100, 2) if var_hits is not None else None,
                    "var_accuracy": round(var_accuracy * 100, 2) if var_accuracy is not None else None
                }
            }
        )
    
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred for {symbol} (Backtest): {e}")
        traceback.print_exc()
        try:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"An unexpected server error occurred: {str(e)}"}, status_code=500)
        except:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.get("/horizon-analysis/{symbol}",
           response_class=HTMLResponse,
           summary="Multi-horizon GARCH analysis with averaged forecasts")
async def get_multi_horizon_analysis(
    request: Request,
    symbol: str,
    horizon: int = Query(default=5, ge=1, le=30, description="Forecast horizon length (days)"),
    lookback_days: int = Query(default=30, ge=5, le=365, description="Number of days to analyze"),
    p: int = Query(default=1, ge=1, le=3, description="ARCH parameter (p)"),
    q: int = Query(default=1, ge=1, le=3, description="GARCH parameter (q)"),
    vol_model: str = Query(default="Garch", description="Volatility model (Garch, EGARCH, GJR)"),
    distribution: str = Query(default="Normal", description="Error distribution (Normal, StudentsT)"),
    auto_select: bool = Query(default=False, description="Auto-select the best model specification"),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db)
):
    """Endpoint to analyze multiple time points with 5-day forecasts, and compare averages with actuals."""
    try:
        # 1. Fetch data
        print(f"Fetching data for {symbol} (Horizon Analysis endpoint)...")
        df_prices = db_service.get_or_fetch_stock_data(db, symbol, settings.fmp_api_key)
        if df_prices is None or df_prices.empty:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"Could not fetch data for {symbol}"}, status_code=404)
        if 'Close' not in df_prices.columns:
             return templates.TemplateResponse("error.html", {"request": request, "detail": "'Close' column missing"}, status_code=500)

        # Calculate the start date - we need enough history plus our forecast period + lookback period
        min_required_data = 252 + lookback_days + horizon  # 252 days for model training baseline
        if len(df_prices) < min_required_data:
            return templates.TemplateResponse("error.html", 
                                              {"request": request, 
                                               "detail": f"Not enough data for {symbol} to perform horizon analysis. Need at least {min_required_data} days."}, 
                                              status_code=400)
        
        # Select dates for analysis - starting from the last date and going backward
        end_idx = len(df_prices) - 1
        start_idx = end_idx - lookback_days
        
        # Analysis results storage
        forecasts = []
        forecast_dates = []
        actual_volatilities = []
        averaged_forecasts = []
        horizon_errors = []
        
        # Calculate returns for the whole dataset once
        returns = garch_service.calculate_returns(df_prices['Close'])
        
        # For each day in our lookback period
        print(f"Running {lookback_days} horizon-based forecasts...")
        for i in range(lookback_days - horizon):
            # Current date index
            current_idx = start_idx + i
            current_date = df_prices.index[current_idx]
            
            # For each date, we'll train on data up to that point
            train_returns = returns.iloc[:current_idx]
            
            # Fit model (or select automatically)
            if auto_select:
                model_result = garch_service.model_selection(train_returns, max_p=2, max_q=2)
                model_fit = model_result['model']
            else:
                model_fit = garch_service.fit_garch_model(train_returns, p=p, q=q, 
                                                         vol_model=vol_model, dist=distribution)
            
            if model_fit:
                # Generate forecast
                forecast_df = garch_service.forecast_volatility(model_fit, horizon=horizon)
                
                if forecast_df is not None and not forecast_df.empty:
                    # Average the forecast over the horizon
                    avg_forecast = forecast_df['Forecasted_Annualized_Volatility'].mean()
                    
                    # Calculate actual volatility over the next 'horizon' days
                    if current_idx + horizon < len(returns):
                        actual_returns = returns.iloc[current_idx:current_idx + horizon]
                        actual_vol = np.std(actual_returns) * np.sqrt(252)  # Annualized
                        
                        # Calculate error
                        error = avg_forecast - actual_vol
                        
                        # Store results
                        forecast_dates.append(current_date)
                        forecasts.append(forecast_df.to_dict('records'))
                        actual_volatilities.append(actual_vol)
                        averaged_forecasts.append(avg_forecast)
                        horizon_errors.append(error)
        
        if not forecast_dates:
            return templates.TemplateResponse("error.html", {"request": request, "detail": "Failed to generate any valid forecasts."}, status_code=500)
        
        # Create summary metrics
        mse = np.mean(np.square(horizon_errors))
        mae = np.mean(np.abs(horizon_errors))
        rmse = np.sqrt(mse)
        bias = np.mean(horizon_errors)  # Positive means over-forecasting
        
        # Format dates
        dates_str = [date.strftime('%Y-%m-%d') for date in forecast_dates]
        
        # Prepare data for chart
        horizon_analysis_data = {
            "dates": dates_str,
            "avg_forecasts": [round(x, 2) for x in averaged_forecasts],
            "actual_vols": [round(x, 2) for x in actual_volatilities],
            "errors": [round(x, 2) for x in horizon_errors]
        }
        
        # Convert to JSON for template
        horizon_analysis_json = json.dumps(horizon_analysis_data)
        
        # Create a nice title for the chart
        model_description = "Auto-selected" if auto_select else f"{vol_model}({p},{q}) with {distribution} distribution"
        
        # Render the template
        return templates.TemplateResponse(
            "horizon_analysis.html",
            {
                "request": request,
                "symbol": symbol,
                "horizon": horizon,
                "lookback_days": lookback_days,
                "model_description": model_description,
                "horizon_data": horizon_analysis_json,
                "metrics": {
                    "mse": round(mse, 4),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "bias": round(bias, 4)
                }
            }
        )
    
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred for {symbol} (Horizon Analysis): {e}")
        traceback.print_exc()
        try:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"An unexpected server error occurred: {str(e)}"}, status_code=500)
        except:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}") 