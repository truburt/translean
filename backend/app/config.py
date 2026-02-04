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
    whisper_keep_alive_seconds: int = 900

    ollama_base_url: str = "http://localhost:11434"
    llm_model_translation: str = "translategemma:12b"
    ollama_keep_alive_seconds: int = 900

    commit_timeout_seconds: float = 6.0
    silence_finalize_seconds: float = 1.4
    min_preview_buffer_seconds: float = 0.5
    stable_window_seconds: float = 5.0
    no_speech_prob_skip: float = 0.85
    no_speech_prob_logprob_skip: float = 0.6
    avg_logprob_skip: float = -1.0
    compression_ratio_skip: float = 2.4

    oidc_issuer_url: str = "https://example-issuer"
    oidc_client_id: str = "example-client"
    oidc_client_secret: str = "change-me"
    oidc_redirect_uri: str = "http://localhost:5173/auth/callback"
    oidc_scope: str = "openid email profile"
    allowed_origins: str = "*"
    forwarded_allow_ips: str = "*"
    admin_email_whitelist: str = ""

    # Security: use secure cookies in production (HTTPS)
    use_secure_cookies: bool = True
    dev_mode: bool = False

    log_level: str = "warning"
    log_file: Optional[str] = None

settings = Settings()
