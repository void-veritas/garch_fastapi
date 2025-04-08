from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import HTMLResponse
from app.services import garch_service, data_service
from app.config import Settings, get_settings
import pandas as pd
import json

# Import templates configured in main.py
# This assumes main.py is in the parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from main import templates

router = APIRouter(prefix="/forecast", tags=["Forecast", "Chart"])

# Pydantic model for request body if needed later
# class ForecastRequest(BaseModel):
#     symbol: str
#     horizon: int = 1

@router.get("/{symbol}", 
            response_model=None, # Adjust if you create a Pydantic model later
            summary="Get GARCH Forecast (JSON)") 
def get_garch_forecast_json(symbol: str,
                       horizon: int = Query(default=10, ge=1, le=30, description="Forecast horizon in days (1-30)"),
                       settings: Settings = Depends(get_settings)):
    """Endpoint to get GARCH(1,1) annualized volatility forecast for a symbol (returns JSON)."""
    try:
        # 1. Fetch data
        print(f"Fetching data for {symbol} (JSON endpoint)...")
        df_prices = data_service.fetch_data(symbol, settings.fmp_api_key)
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
                                 settings: Settings = Depends(get_settings)):
    """Endpoint to display a chart of the GARCH(1,1) volatility forecast."""
    try:
        # 1. Fetch data
        print(f"Fetching data for {symbol} (Chart endpoint)...")
        df_prices = data_service.fetch_data(symbol, settings.fmp_api_key)
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
        forecast_list = forecast_df.reset_index().to_dict(orient='records')
        for record in forecast_list:
            record['Date'] = record['Date'].strftime('%Y-%m-%d') # Format date for JS
        
        # Convert list of dicts to JSON string to safely embed in HTML/JS
        forecast_data_json = json.dumps(forecast_list)

        print(f"Rendering chart for {symbol}.")
        # Render the HTML template, passing the data to it
        return templates.TemplateResponse(
            "chart.html",
            {
                "request": request, # Required by Jinja2Templates
                "symbol": symbol,
                "horizon": horizon,
                "forecast_data": forecast_data_json # Pass JSON string
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