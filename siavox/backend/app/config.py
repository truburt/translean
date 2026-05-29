import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Google OAuth client ID
    GOOGLE_CLIENT_ID: str = "your-google-client-id.apps.googleusercontent.com"
    
    # Ollama instance details
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma4:e4b"
    
    # SQLite Database URL
    DATABASE_URL: str = "sqlite:///./siavox.db"
    
    # FastAPI session secret key
    SECRET_KEY: str = "dev-secret-key-1234567890-change-in-production"
    
    # Session Cookie Name
    SESSION_COOKIE_NAME: str = "siavox_session"

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
