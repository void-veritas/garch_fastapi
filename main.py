from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from app.routers import data, forecast

app = FastAPI(title="GARCH Volatility Forecast API")

# Configure Jinja2Templates
templates = Jinja2Templates(directory="templates")


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
