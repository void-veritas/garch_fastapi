from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from app.services import garch_service, data_service, db_service
# Import Settings, get_settings, and templates from config
from app.config import Settings, get_settings, templates, get_db
import pandas as pd
import json
import numpy as np # Need numpy for sqrt

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

@router.get("/{symbol}", 
            response_model=None, # Adjust if you create a Pydantic model later
            summary="Get GARCH Forecast (JSON)") 
def get_garch_forecast_json(symbol: str,
                       horizon: int = Query(default=10, ge=1, le=30, description="Forecast horizon in days (1-30)"),
                       settings: Settings = Depends(get_settings),
                       db: Session = Depends(get_db)):
    """Endpoint to get GARCH(1,1) annualized volatility forecast for a symbol (returns JSON)."""
    try:
        # 1. Fetch data
        print(f"Fetching data for {symbol} (JSON endpoint)...")
        df_prices = db_service.get_or_fetch_stock_data(db, symbol, settings.fmp_api_key)
        if df_prices is None or df_prices.empty:
            raise HTTPException(status_code=404,
                                detail=f"Could not fetch valid historical data for {symbol} from FMP.")
        if 'Close' not in df_prices.columns:
             raise HTTPException(status_code=500, detail="'Close' column missing in fetched data.")

        # 2. Calculate Returns
        print(f"Calculating returns for {symbol}...")
        returns = garch_service.calculate_returns(df_prices['Close'])
        if returns.empty:
             raise HTTPException(status_code=500, detail="Failed to calculate returns. Insufficient data?")

        # 3. Fit GARCH model
        print(f"Fitting GARCH(1,1) model for {symbol}...")
        model_fit = garch_service.fit_garch_model(returns, p=1, q=1)
        if model_fit is None:
             raise HTTPException(status_code=500, detail=f"Failed to fit GARCH model for {symbol}.")

        # 4. Forecast volatility
        print(f"Forecasting volatility for {symbol} (horizon={horizon})...")
        forecast_df = garch_service.forecast_volatility(model_fit, horizon=horizon)
        if forecast_df is None or forecast_df.empty:
            raise HTTPException(status_code=500, detail=f"Failed to generate volatility forecast for {symbol}.")

        # 5. Format and return the forecast as JSON
        forecast_list = forecast_df.reset_index().to_dict(orient='records')
        for record in forecast_list:
            record['Date'] = record['Date'].strftime('%Y-%m-%d')
        print(f"Successfully generated JSON forecast for {symbol}.")
        return {"symbol": symbol, "forecast_horizon": horizon, "forecast": forecast_list}

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
async def get_garch_forecast_chart(request: Request, # Need Request for templates
                                 symbol: str,
                                 horizon: int = Query(default=10, ge=1, le=30, description="Forecast horizon in days (1-30)"),
                                 settings: Settings = Depends(get_settings),
                                 db: Session = Depends(get_db)):
    """Endpoint to display a chart of the GARCH(1,1) volatility forecast, including recent history."""
    try:
        # 1. Fetch data
        print(f"Fetching data for {symbol} (Chart endpoint)...")
        df_prices = db_service.get_or_fetch_stock_data(db, symbol, settings.fmp_api_key)
        if df_prices is None or df_prices.empty:
            # Return an HTML error page or message
             return templates.TemplateResponse("error.html", {"request": request, "detail": f"Could not fetch data for {symbol}"}, status_code=404)
        if 'Close' not in df_prices.columns:
             return templates.TemplateResponse("error.html", {"request": request, "detail": "'Close' column missing"}, status_code=500)

        # 2. Calculate Returns
        print(f"Calculating returns for {symbol}...")
        returns = garch_service.calculate_returns(df_prices['Close'])
        if returns.empty:
            return templates.TemplateResponse("error.html", {"request": request, "detail": "Failed to calculate returns"}, status_code=500)

        # 3. Fit GARCH model
        print(f"Fitting GARCH(1,1) model for {symbol}...")
        model_fit = garch_service.fit_garch_model(returns, p=1, q=1)
        if model_fit is None:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"Failed to fit GARCH model for {symbol}"}, status_code=500)

        # 4. Forecast volatility
        print(f"Forecasting volatility for {symbol} (horizon={horizon})...")
        forecast_df = garch_service.forecast_volatility(model_fit, horizon=horizon)
        if forecast_df is None or forecast_df.empty:
             return templates.TemplateResponse("error.html", {"request": request, "detail": f"Failed to generate forecast for {symbol}"}, status_code=500)

        # 5. Prepare data for the template
        
        # 5a. Get historical conditional volatility (annualized)
        hist_periods = 100 # Number of historical periods to show
        # model_fit.conditional_volatility contains the conditional standard deviation
        # Take the last 'hist_periods' points
        # hist_variance = model_fit.conditional_volatility**2 # Incorrect: This is already std dev
        hist_daily_vol = model_fit.conditional_volatility.iloc[-hist_periods:]
        # Annualize by multiplying daily std deviation by sqrt(252)
        hist_vol_series = hist_daily_vol * np.sqrt(252)
        
        # Format historical data
        hist_df = pd.DataFrame({'Historical_Annualized_Volatility': hist_vol_series})
        hist_list = hist_df.reset_index().to_dict(orient='records')
        for record in hist_list:
            # Assuming the index name from garch service is 'Date'
            record['Date'] = record['Date'].strftime('%Y-%m-%d') 
        
        # 5b. Format forecast data (already done)
        forecast_list = forecast_df.reset_index().to_dict(orient='records')
        for record in forecast_list:
            record['Date'] = record['Date'].strftime('%Y-%m-%d')
        
        # 5c. Convert data to JSON strings for embedding
        hist_data_json = json.dumps(hist_list)
        forecast_data_json = json.dumps(forecast_list)

        print(f"Rendering chart for {symbol} with history.")
        # Render the HTML template, passing both historical and forecast data
        return templates.TemplateResponse(
            "chart.html",
            {
                "request": request, 
                "symbol": symbol,
                "horizon": horizon,
                "hist_data": hist_data_json,     # Pass historical data
                "forecast_data": forecast_data_json # Pass forecast data
            }
        )

    except Exception as e:
        # Catch any other unexpected errors and show an error page
        import traceback
        print(f"An unexpected error occurred for {symbol} (Chart): {e}")
        traceback.print_exc()
        # Use a generic error template if available, otherwise raise HTTP exception
        # Assuming we create an error.html template later
        try:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"An unexpected server error occurred: {str(e)}"}, status_code=500)
        except: # If error template itself fails
             raise HTTPException(status_code=500, detail=f"An unexpected server error occurred and failed to load error page: {str(e)}") 

@router.get("/backtest/{symbol}", 
            response_class=HTMLResponse, 
            summary="Display GARCH Backtest Results Chart (HTML)")
async def get_garch_backtest_chart(request: Request,
                                  symbol: str,
                                  window: int = Query(default=252, ge=50, description="Rolling window size for backtest (min 50)"),
                                  settings: Settings = Depends(get_settings),
                                  db: Session = Depends(get_db)):
    """Endpoint to perform and display walk-forward GARCH backtest results."""
    try:
        # 1. Fetch data
        print(f"Fetching data for {symbol} (Backtest endpoint)...")
        df_prices = db_service.get_or_fetch_stock_data(db, symbol, settings.fmp_api_key)
        if df_prices is None or df_prices.empty:
             return templates.TemplateResponse("error.html", {"request": request, "detail": f"Could not fetch data for {symbol}"}, status_code=404)
        if 'Close' not in df_prices.columns:
             return templates.TemplateResponse("error.html", {"request": request, "detail": "'Close' column missing"}, status_code=500)

        # 2. Run Backtest Service
        print(f"Running backtest for {symbol} with window {window}...")
        backtest_results_df = garch_service.backtest_garch_forecast(df_prices['Close'], window_size=window)

        if backtest_results_df is None or backtest_results_df.empty:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"Failed to generate backtest results for {symbol}. Not enough data or other error."}, status_code=500)
        
        # 3. Prepare data for template
        # Limit the data sent to the template if it's very long (e.g., last 2 years = ~500 points)
        backtest_results_df = backtest_results_df.iloc[-500:]

        backtest_list = backtest_results_df.reset_index().to_dict(orient='records')
        for record in backtest_list:
            record['Date'] = record['Date'].strftime('%Y-%m-%d')
            # Round numbers for display
            record['Forecasted_Volatility'] = round(record['Forecasted_Volatility'], 2)
            record['Realized_Volatility'] = round(record['Realized_Volatility'], 2)
            record['Error'] = round(record['Error'], 2)

        backtest_data_json = json.dumps(backtest_list)

        # Calculate some summary statistics (e.g., Mean Absolute Error)
        mae = round(backtest_results_df['Error'].abs().mean(), 2)
        rmse = round(np.sqrt((backtest_results_df['Error']**2).mean()), 2)
        stats = {
            "mae": mae,
            "rmse": rmse
        }

        print(f"Rendering backtest chart for {symbol}.")
        return templates.TemplateResponse(
            "backtest.html",
            {
                "request": request,
                "symbol": symbol,
                "window": window,
                "backtest_data": backtest_data_json,
                "stats": stats
            }
        )

    except Exception as e:
        import traceback
        print(f"An unexpected error occurred for {symbol} (Backtest): {e}")
        traceback.print_exc()
        try:
            return templates.TemplateResponse("error.html", {"request": request, "detail": f"An unexpected server error occurred: {str(e)}"}, status_code=500)
        except: 
             raise HTTPException(status_code=500, detail=f"An unexpected server error occurred and failed to load error page: {str(e)}") 