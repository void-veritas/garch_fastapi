from fastapi import FastAPI
from app.routers import data, forecast
from app.db_init import init_db

# Initialize the database
init_db()

app = FastAPI(title="GARCH Volatility Forecast API")


@app.get("/", tags=["General"])
def read_root():
    """Root endpoint for the API."""
    return {"message": "Welcome to the GARCH Volatility Forecast API"}

# Include routers
app.include_router(data.router)
app.include_router(forecast.router)

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
