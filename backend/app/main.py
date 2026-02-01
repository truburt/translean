"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

FastAPI application entrypoint wiring routers and middleware.
"""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from .api import router as api_router
from .config import settings
from .db import engine, Base
from . import models  # noqa: F401

# Configure logging handlers
handlers = [logging.StreamHandler()]
if settings.log_file:
    handlers.append(logging.FileHandler(settings.log_file))

logging.basicConfig(
    level=logging.INFO,
    handlers=handlers,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Set the application's specific log level from settings
logging.getLogger("app").setLevel(settings.log_level.upper())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log configuration on startup."""
    logger.info("Starting Translean Backend")
    logger.info("Loaded Configuration:")
    for key, value in settings.model_dump().items():
        if any(secret in key.lower() for secret in ["key", "secret", "token", "password"]):
            value = "***"
        logger.info("  %s: %s", key, value)
    
    # Initialize Database
    if settings.dev_mode:
        async with engine.begin() as conn:
            logger.info("Initializing Database")
            await conn.run_sync(Base.metadata.create_all)
    
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

origins = [origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trust forwarded headers from Nginx (or other proxies)
# This is crucial for correct scheme detection (http vs https) and Secure cookies
app.add_middleware(
    ProxyHeadersMiddleware,
    trusted_hosts=[ip.strip() for ip in settings.forwarded_allow_ips.split(",")] if settings.forwarded_allow_ips != "*" else "*",
)

app.include_router(api_router)
