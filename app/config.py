from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    fmp_api_key: str = "YOUR_FMP_API_KEY"

    model_config = SettingsConfigDict(env_file='.env')


@lru_cache()
def get_settings():
    return Settings() 