from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

class Settings(BaseSettings):
    fmp_api_key: str = "YOUR_FMP_API_KEY"
    database_url: str = "sqlite:///./garch_app.db"
    
    model_config = SettingsConfigDict(env_file='.env')

# Define templates instance here
templates = Jinja2Templates(directory="templates")

@lru_cache()
def get_settings():
    return Settings()

# Database setup
settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 