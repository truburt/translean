"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

Application configuration powered by environment variables.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Centralized configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=(".env", "../.env", "../../.env"), env_prefix="", case_sensitive=False, extra="ignore")

    app_name: str = "translean"
    app_base_url: str = "http://localhost:8000"
    database_url: str = "sqlite+aiosqlite:///./translean.db"

    whisper_base_url: str = "http://localhost:8000"
    whisper_model: str = "Systran/faster-whisper-large-v3"

    ollama_base_url: str = "http://localhost:11434"
    llm_model_translation: str = "translategemma:12b"

    oidc_issuer_url: str = "https://example-issuer"
    oidc_client_id: str = "example-client"
    oidc_client_secret: str = "change-me"
    oidc_redirect_uri: str = "http://localhost:5173/auth/callback"
    oidc_scope: str = "openid email profile"
    allowed_origins: str = "*"
    forwarded_allow_ips: str = "*"

    # Security: use secure cookies in production (HTTPS)
    use_secure_cookies: bool = True
    dev_mode: bool = False

    log_level: str = "warning"
    log_file: Optional[str] = None

settings = Settings()
