from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from fastapi.templating import Jinja2Templates

class Settings(BaseSettings):
    fmp_api_key: str = "YOUR_FMP_API_KEY"

    model_config = SettingsConfigDict(env_file='.env')

# Define templates instance here
templates = Jinja2Templates(directory="templates")

@lru_cache()
def get_settings():
    return Settings() 