# Placeholder for GARCH modeling logic
import pandas as pd
import numpy as np
from arch import arch_model
from typing import Union, Tuple, Dict, List, Optional
from tqdm import tqdm # Import tqdm for progress bar
import itertools
from scipy import stats

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculates percentage returns from a price series."""
    # Using log returns is common for financial modeling
    return np.log(prices / prices.shift(1)).dropna() * 100
    # Alternative: simple returns:
    # return prices.pct_change().dropna() * 100

def fit_garch_model(returns: pd.Series, p: int = 1, q: int = 1, 
                    vol_model: str = 'Garch', dist: str = 'Normal') -> Union['arch.univariate.ARCHModelResult', None]:
    """Fits a GARCH(p, q) model to the given returns series.

    Args:
        returns: A pandas Series of percentage returns.
        p: The order of the ARCH terms.
        q: The order of the GARCH terms.
        vol_model: Volatility model to use ('Garch', 'EGARCH', 'GJR', 'FIGARCH', etc.)
        dist: Distribution to use ('Normal', 'StudentsT', 'SkewStudent', etc.)

    Returns:
        The fitted model result object, or None if fitting fails.
    """
    if returns.empty or len(returns) < p + q + 1: # Need enough data
        print(f"Cannot fit GARCH model: Insufficient data points ({len(returns)}). Need at least {p+q+1}.")
        return None
    try:
        # Define the GARCH model
        model = arch_model(returns, vol=vol_model, p=p, q=q, dist=dist)

        # Fit the model
        # Use robustness settings for better convergence
        model_fit = model.fit(disp='off', update_freq=0, cov_type='robust')
        print(f"GARCH model fitting successful: {vol_model}({p},{q}) with {dist} distribution.")
        return model_fit
    except Exception as e:
        print(f"Error fitting GARCH({p},{q}) model: {e}")
        return None

def model_selection(returns: pd.Series, max_p: int = 2, max_q: int = 2,
                   vol_models: List[str] = ['Garch', 'GJR', 'EGARCH'],
                   distributions: List[str] = ['Normal', 'StudentsT']) -> Dict:
    """Perform model selection to find the best GARCH specification.
    
    Args:
        returns: Returns series
        max_p: Maximum ARCH order to consider
        max_q: Maximum GARCH order to consider
        vol_models: List of volatility models to try
        distributions: List of error distributions to try
        
    Returns:
        Dictionary with best model parameters and fitted model
    """
    p_range = range(1, max_p + 1)
    q_range = range(1, max_q + 1)
    
    best_aic = np.inf
    best_bic = np.inf
    best_model = None
    best_params = {}
    
    print("Performing GARCH model selection...")
    # Generate all combinations
    all_combos = list(itertools.product(p_range, q_range, vol_models, distributions))
    
    for p, q, vol_model, dist in tqdm(all_combos, desc="Testing GARCH models"):
        try:
            model_fit = fit_garch_model(returns, p=p, q=q, vol_model=vol_model, dist=dist)
            if model_fit is not None:
                # Check information criteria
                current_aic = model_fit.aic
                current_bic = model_fit.bic
                
                # Update best model based on BIC (more conservative for forecasting)
                if current_bic < best_bic:
                    best_bic = current_bic
                    best_aic = current_aic
                    best_model = model_fit
                    best_params = {
                        'p': p, 'q': q, 
                        'vol_model': vol_model, 
                        'dist': dist,
                        'aic': current_aic,
                        'bic': current_bic
                    }
        except Exception as e:
            # Skip failed models
            continue
    
    if best_model is None:
        print("Model selection failed, defaulting to GARCH(1,1)")
        best_model = fit_garch_model(returns)
        best_params = {'p': 1, 'q': 1, 'vol_model': 'Garch', 'dist': 'Normal'}
    else:
        print(f"Best model: {best_params['vol_model']}({best_params['p']},{best_params['q']}) "
              f"with {best_params['dist']} distribution (BIC: {best_params['bic']:.2f})")
        
    return {'model': best_model, 'params': best_params}

def forecast_volatility(model_fit: 'arch.univariate.ARCHModelResult', 
                       horizon: int = 1, 
                       alpha: float = 0.05) -> Union[pd.DataFrame, None]:
    """Forecasts conditional volatility using the fitted GARCH model.

    Args:
        model_fit: The fitted ARCHModelResult object.
        horizon: The number of steps ahead to forecast.
        alpha: Significance level for confidence intervals (e.g., 0.05 for 95% CI)

    Returns:
        A pandas DataFrame containing the dates and forecasted variance/
        volatility with confidence intervals, or None if forecasting fails.
    """
    if model_fit is None:
        print("Cannot forecast: Invalid model fit object.")
        return None
    try:
        # Forecast conditional variance
        forecast = model_fit.forecast(horizon=horizon, reindex=False)
        
        # The forecast object contains variance forecasts
        variance_forecast = forecast.variance
        
        # Convert daily variance to annualized volatility (sqrt(variance * 252))
        # 252 is the typical number of trading days in a year
        if isinstance(variance_forecast, pd.DataFrame):
            # If we have a DataFrame with multiple columns (simulation method)
            mean_variance = variance_forecast.mean(axis=1)
            annualized_volatility_forecast = np.sqrt(mean_variance * 252)
        else:
            # If we have a single-column Series (mean forecasts)
            annualized_volatility_forecast = np.sqrt(variance_forecast * 252)
        
        # Create a DataFrame for the forecast
        # Generate future dates starting from the day after the last date in the original data
        last_date = model_fit.resid.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        
        # Ensure all arrays have the same length
        if len(annualized_volatility_forecast) != len(future_dates):
            print(f"Warning: Forecast length mismatch. Resizing forecast to match {len(future_dates)} dates.")
            # If the forecast length doesn't match horizon, adjust it
            if len(annualized_volatility_forecast) > len(future_dates):
                annualized_volatility_forecast = annualized_volatility_forecast[:len(future_dates)]
            else:
                # Pad with the last value
                last_value = annualized_volatility_forecast.iloc[-1] if hasattr(annualized_volatility_forecast, 'iloc') else annualized_volatility_forecast[-1]
                padding = [last_value] * (len(future_dates) - len(annualized_volatility_forecast))
                if isinstance(annualized_volatility_forecast, pd.Series):
                    # For pandas Series
                    padding_idx = pd.date_range(
                        start=annualized_volatility_forecast.index[-1] + pd.Timedelta(days=1), 
                        periods=len(padding)
                    )
                    padding_series = pd.Series(padding, index=padding_idx)
                    annualized_volatility_forecast = pd.concat([annualized_volatility_forecast, padding_series])
                else:
                    # For numpy arrays
                    annualized_volatility_forecast = np.append(annualized_volatility_forecast, padding)
        
        # Simple approximation for confidence intervals based on historical volatility
        # Use standard deviation of returns to estimate volatility of volatility
        returns_std = np.std(model_fit.resid)
        z_score_lower = stats.norm.ppf(alpha / 2)
        z_score_upper = stats.norm.ppf(1 - alpha / 2)
        
        # Create confidence intervals
        if isinstance(annualized_volatility_forecast, pd.Series):
            ci_lower = annualized_volatility_forecast + z_score_lower * returns_std * np.sqrt(252)
            ci_upper = annualized_volatility_forecast + z_score_upper * returns_std * np.sqrt(252)
        else:
            ci_lower = annualized_volatility_forecast + z_score_lower * returns_std * np.sqrt(252)
            ci_upper = annualized_volatility_forecast + z_score_upper * returns_std * np.sqrt(252)
        
        # Ensure CI doesn't go below zero
        ci_lower = np.maximum(ci_lower, 0)
        
        # Create DataFrame with aligned lengths
        forecast_df = pd.DataFrame(index=future_dates)
        forecast_df['Forecasted_Annualized_Volatility'] = annualized_volatility_forecast
        forecast_df['CI_Lower'] = ci_lower
        forecast_df['CI_Upper'] = ci_upper

        print(f"Generated volatility forecast for {horizon} steps with {(1-alpha)*100}% confidence intervals.")
        return forecast_df

    except Exception as e:
        print(f"Error forecasting volatility: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_risk_metrics(volatility: pd.Series, returns: pd.Series, confidence_level: float = 0.95) -> Dict:
    """Calculate risk metrics based on volatility and returns.
    
    Args:
        volatility: Annualized volatility series
        returns: Daily returns series
        confidence_level: Confidence level for VaR and ES calculations
    
    Returns:
        Dictionary of risk metrics
    """
    # Daily volatility = annualized volatility / sqrt(252)
    daily_vol = volatility / np.sqrt(252)
    
    # Value at Risk (VaR) calculation
    # Using the parametric approach assuming normal distribution
    z_score = stats.norm.ppf(1 - confidence_level)
    var = z_score * daily_vol
    
    # Expected Shortfall (ES) / Conditional VaR
    # Expected loss beyond VaR
    z_es = stats.norm.pdf(z_score) / (1 - confidence_level)
    es = z_es * daily_vol
    
    # Calculate realized volatility
    realized_vol = returns.rolling(window=21).std() * np.sqrt(252)
    
    # RMSE between forecasted and realized volatility (where available)
    rmse = np.sqrt(np.mean((volatility.dropna() - realized_vol.dropna())**2))
    
    return {
        'VaR': var,
        'ES': es,
        'RMSE': rmse
    }

def backtest_garch_forecast(prices: pd.Series, 
                          window_size: int = 252, 
                          p: int = 1, 
                          q: int = 1,
                          vol_model: str = 'Garch',
                          dist: str = 'Normal',
                          auto_select: bool = False) -> pd.DataFrame | None:
    """Performs walk-forward backtesting of 1-day GARCH volatility forecasts.

    Args:
        prices: Series of historical prices with a DatetimeIndex.
        window_size: The size of the rolling window for fitting the GARCH model.
        p: The order of the ARCH terms.
        q: The order of the GARCH terms.
        vol_model: Volatility model specification
        dist: Error distribution
        auto_select: Whether to perform model selection at each step

    Returns:
        DataFrame with columns: 'Date', 'Forecasted_Volatility', 'Realized_Volatility', 'Error', 'VaR', 'ES'.
        Forecasted and Realized are annualized.
        Returns None if backtesting fails.
    """
    if prices.empty or len(prices) <= window_size:
        print(f"Insufficient data for backtest. Need more than {window_size} price points.")
        return None

    returns = calculate_returns(prices)
    if returns.empty or len(returns) < window_size:
         print(f"Insufficient returns data for backtest window size {window_size}.")
         return None

    forecasts = []
    realized_variances = [] 
    dates = []
    var_values = []
    es_values = []
    models_used = []

    print(f"Starting GARCH backtest for {len(returns) - window_size} steps...")
    # Iterate from the end of the first window up to the end of the returns series
    for t in tqdm(range(window_size, len(returns)), desc="GARCH Backtest Progress"):
        # Data up to t-1 is used for fitting
        current_window_returns = returns.iloc[t - window_size : t]
        
        # Model selection or fixed model
        if auto_select:
            # Only do model selection every 20 steps to save computation
            if (t - window_size) % 20 == 0:
                result = model_selection(current_window_returns, max_p=2, max_q=2)
                model_fit = result['model']
                models_used.append(f"{result['params']['vol_model']}({result['params']['p']},{result['params']['q']})-{result['params']['dist']}")
            else:
                # Use last successful parameters
                if models_used:
                    last_model = models_used[-1]
                    model_parts = last_model.replace(')', '').replace('(', '-').split('-')
                    vol_model_t = model_parts[0]
                    p_t = int(model_parts[1])
                    q_t = int(model_parts[2])
                    dist_t = model_parts[3]
                    model_fit = fit_garch_model(current_window_returns, p=p_t, q=q_t, 
                                              vol_model=vol_model_t, dist=dist_t)
                    models_used.append(last_model)
                else:
                    model_fit = fit_garch_model(current_window_returns, p=p, q=q, 
                                             vol_model=vol_model, dist=dist)
                    models_used.append(f"{vol_model}({p},{q})-{dist}")
        else:
            # Fixed model specification
            model_fit = fit_garch_model(current_window_returns, p=p, q=q, 
                                      vol_model=vol_model, dist=dist)
            models_used.append(f"{vol_model}({p},{q})-{dist}")
        
        if model_fit:
            try:
                # Forecast variance for the next step (day t)
                forecast_result = model_fit.forecast(horizon=1, reindex=False)
                forecasted_variance_t = forecast_result.variance.iloc[0, 0] # Get the h.1 value
                
                # Store the daily forecasted variance
                forecasts.append(forecasted_variance_t)
                
                # Get actual return for day t (which happened after the forecast was made)
                actual_return_t = returns.iloc[t] 
                # Use squared return as proxy for realized variance
                realized_variances.append(actual_return_t**2)
                
                # Store the date for which the forecast was made (day t)
                dates.append(returns.index[t])
                
                # Calculate VaR and ES
                daily_vol = np.sqrt(forecasted_variance_t)
                annualized_vol = daily_vol * np.sqrt(252)
                
                # 95% VaR (using parametric approach)
                z_score = stats.norm.ppf(0.05)  # 5% significance
                var_t = -z_score * daily_vol
                var_values.append(var_t)
                
                # Expected Shortfall (ES)
                z_es = stats.norm.pdf(z_score) / 0.05
                es_t = z_es * daily_vol
                es_values.append(es_t)
                
            except Exception as e:
                continue
        else:
            continue

    if not dates:
        print("Backtest did not produce any valid forecast points.")
        return None
    
    print("Backtest loop finished. Processing results...")

    # Create DataFrame from results
    backtest_results = pd.DataFrame({
        'Date': dates,
        # Annualize forecasted volatility: sqrt(variance * 252)
        'Forecasted_Volatility': np.sqrt(np.array(forecasts) * 252),
        # Annualize realized volatility proxy: sqrt(squared_return * 252)
        'Realized_Volatility': np.sqrt(np.array(realized_variances) * 252),
        'VaR_95': np.array(var_values),
        'ES_95': np.array(es_values),
        'Model_Used': models_used
    })
    backtest_results = backtest_results.set_index('Date')

    # Calculate Error
    backtest_results['Error'] = backtest_results['Forecasted_Volatility'] - backtest_results['Realized_Volatility']
    
    # Calculate performance metrics
    mse = np.mean(backtest_results['Error']**2)
    mae = np.mean(np.abs(backtest_results['Error']))
    
    # Calculate hit rate for VaR (percentage of times return exceeds VaR)
    returns_for_backtesting = returns.loc[backtest_results.index]
    var_hits = (returns_for_backtesting < -backtest_results['VaR_95']).mean()
    
    print(f"Backtest complete. Generated {len(backtest_results)} results.")
    print(f"Performance metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, VaR hit rate: {var_hits:.2%}")
    
    return backtest_results 

def compare_garch_models(returns: pd.Series, 
                        models_to_compare: List[Dict] = None,
                        horizon: int = 10,
                        training_days: int = 252) -> Dict:
    """Compares multiple GARCH model specifications and forecasts.
    
    Args:
        returns: Series of historical returns with DatetimeIndex
        models_to_compare: List of model specifications to compare, each containing:
                           {'vol_model': str, 'p': int, 'q': int, 'dist': str}
                           If None, a default set of common models will be used
        horizon: Number of days to forecast for all models
        training_days: Number of training days to use (most recent)
        
    Returns:
        Dictionary containing:
        - 'models': List of fitted model details with metrics
        - 'best_model': The best model based on BIC
        - 'forecasts': Dictionary with forecast data for all models
    """
    # Default set of models if none provided
    if models_to_compare is None:
        models_to_compare = [
            {'vol_model': 'Garch', 'p': 1, 'q': 1, 'dist': 'Normal'},
            {'vol_model': 'Garch', 'p': 1, 'q': 1, 'dist': 'StudentsT'},
            {'vol_model': 'GJR', 'p': 1, 'q': 1, 'dist': 'Normal'},
            {'vol_model': 'GJR', 'p': 1, 'q': 1, 'dist': 'StudentsT'},
            {'vol_model': 'EGARCH', 'p': 1, 'q': 1, 'dist': 'Normal'},
            {'vol_model': 'EGARCH', 'p': 1, 'q': 1, 'dist': 'StudentsT'}
        ]
    
    # Use the most recent training_days of returns
    if len(returns) > training_days:
        train_returns = returns.iloc[-training_days:]
    else:
        train_returns = returns
        
    print(f"Comparing {len(models_to_compare)} GARCH models using {len(train_returns)} days of returns")
    
    fitted_models = []
    best_model = None
    best_bic = np.inf
    
    # Fit each model and collect metrics
    for i, model_spec in enumerate(models_to_compare):
        model_label = f"{model_spec['vol_model']}({model_spec['p']},{model_spec['q']})-{model_spec['dist']}"
        print(f"Fitting model {i+1}/{len(models_to_compare)}: {model_label}")
        
        try:
            model_fit = fit_garch_model(
                train_returns, 
                p=model_spec['p'], 
                q=model_spec['q'], 
                vol_model=model_spec['vol_model'],
                dist=model_spec['dist']
            )
            
            if model_fit is not None:
                # Extract model information
                model_info = {
                    'description': model_label,
                    'aic': model_fit.aic,
                    'bic': model_fit.bic,
                    'loglikelihood': model_fit.loglikelihood,
                    'params': str(model_fit.params),
                    'model_fit': model_fit,  # Store the actual model fit object
                    'spec': model_spec
                }
                
                # Check if this is the best model so far (based on BIC)
                if model_fit.bic < best_bic:
                    best_bic = model_fit.bic
                    best_model = model_info.copy()
                
                fitted_models.append(model_info)
                
        except Exception as e:
            print(f"Error fitting model {model_label}: {e}")
            continue
    
    if not fitted_models:
        print("No models were successfully fitted.")
        return None
    
    # Mark the best model
    for model in fitted_models:
        model['is_best'] = 1 if (model['bic'] == best_bic) else 0
    
    # Generate forecasts for all successfully fitted models
    forecasts_data = {
        'dates': [],
        'forecasts': []
    }
    
    # Create forecast dates (for x-axis)
    last_date = train_returns.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    forecasts_data['dates'] = [date.strftime('%Y-%m-%d') for date in forecast_dates]
    
    # Generate forecasts for each model
    for model in fitted_models:
        try:
            forecast_df = forecast_volatility(model['model_fit'], horizon=horizon)
            
            if forecast_df is not None and not forecast_df.empty:
                forecast_values = forecast_df['Forecasted_Annualized_Volatility'].tolist()
                
                forecasts_data['forecasts'].append({
                    'model_label': model['description'],
                    'values': [round(x, 2) for x in forecast_values],
                    'is_best': 1 if model['is_best'] else 0
                })
                
        except Exception as e:
            print(f"Error forecasting with model {model['description']}: {e}")
            continue
    
    # Prepare final result (removing model_fit objects which aren't JSON serializable)
    result_models = []
    for model in fitted_models:
        model_copy = model.copy()
        if 'model_fit' in model_copy:
            del model_copy['model_fit']
        result_models.append(model_copy)
    
    best_model_copy = None
    if best_model:
        best_model_copy = best_model.copy()
        if 'model_fit' in best_model_copy:
            del best_model_copy['model_fit']
    
    return {
        'models': result_models,
        'best_model': best_model_copy,
        'forecasts': forecasts_data
    } 