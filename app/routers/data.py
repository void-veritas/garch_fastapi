from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.services import data_service, db_service
from app.config import Settings, get_settings, get_db

router = APIRouter(prefix="/data", tags=["Data"])

@router.get("/{symbol}")
def get_historical_data(symbol: str, 
                        settings: Settings = Depends(get_settings),
                        db: Session = Depends(get_db)):
    """Endpoint to fetch historical data for a symbol. Uses database cache if available."""
    try:
        data = db_service.get_or_fetch_stock_data(db, symbol, settings.fmp_api_key)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Failed to fetch data for {symbol}")
        return {"symbol": symbol, "data": data.reset_index().to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 