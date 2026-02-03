"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

OIDC token verification helpers with JWKS support.
"""
import datetime as dt
import re
from functools import lru_cache
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwk, jwt

from .config import settings
from .schemas import UserInfo

import logging
logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _jwks_cache():
    """Cache container for JWKS and expiry metadata."""

    return {
        "keys": [],
        "expires_at": dt.datetime.min.replace(tzinfo=dt.timezone.utc),
        "authorization_endpoint": "",
        "token_endpoint": "",
    }


def _refresh_jwks() -> list[dict]:
    cache = _jwks_cache()
    now = dt.datetime.now(dt.timezone.utc)
    # If we have keys and endpoints and cache is fresh, return keys
    # We check authorization_endpoint to ensure we have fully populated cache
    if cache["keys"] and cache.get("authorization_endpoint") and cache["expires_at"] > now:
        return cache["keys"]

    config_url = f"{settings.oidc_issuer_url.rstrip('/')}/.well-known/openid-configuration"
    try:
        # 1. Fetch OIDC discovery document
        provider = httpx.get(config_url, timeout=5.0)
        provider.raise_for_status()
        config = provider.json()
        
        jwks_uri = config.get("jwks_uri")
        auth_endpoint = config.get("authorization_endpoint")
        token_endpoint = config.get("token_endpoint")

        # 2. Fetch JWKS if URI is present
        keys = []
        if jwks_uri:
            resp = httpx.get(jwks_uri, timeout=5.0)
            resp.raise_for_status()
            keys = resp.json().get("keys", [])
            
        # 3. Update cache
        cache.update({
            "keys": keys,
            "expires_at": now + dt.timedelta(minutes=15),
            "authorization_endpoint": auth_endpoint,
            "token_endpoint": token_endpoint,
        })
        return keys
    except Exception:
        # On failure, return existing keys if any, or empty list
        return cache.get("keys", [])


def get_authorization_endpoint() -> str:
    """Retrieve the OIDC authorization endpoint, refreshing cache if needed."""
    _refresh_jwks()
    cache = _jwks_cache()
    # Fallback to manual construction if discovery failed
    return cache.get("authorization_endpoint") or f"{settings.oidc_issuer_url.rstrip('/')}/authorize"


def get_token_endpoint() -> str:
    """Retrieve the OIDC token endpoint, refreshing cache if needed."""
    _refresh_jwks()
    cache = _jwks_cache()
    # Fallback to manual construction if discovery failed
    return cache.get("token_endpoint") or f"{settings.oidc_issuer_url.rstrip('/')}/token"


def decode_token(token: str) -> Optional[UserInfo]:
    """Decode a JWT using JWKS if available, otherwise HS256 fallback."""
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        payload = None
        if kid:
            jwks_keys = _refresh_jwks()
            key_data = next((k for k in jwks_keys if k.get("kid") == kid), None)
            if key_data:
                key = jwk.construct(key_data)
                public_key = key.to_pem().decode()
                payload = jwt.decode(
                    token,
                    public_key,
                    algorithms=[key_data.get("alg", "RS256")],
                    audience=settings.oidc_client_id,
                    issuer=settings.oidc_issuer_url,
                    options={"verify_aud": True},
                )

        if payload is None:
            # Fallback to HS256 (e.g. dev token or no matching key found)
            payload = jwt.decode(
                token,
                settings.oidc_client_secret,
                algorithms=["HS256"],
                audience=settings.oidc_client_id,
                options={"verify_aud": True},
            )
    except JWTError as e:
        logger.error(f"OIDC Token Validation Failed: {e}")
        return None

    subject = payload.get("sub")
    email = payload.get("email")
    name = payload.get("name") or payload.get("preferred_username")
    if not subject or not email:
        logger.error("Invalid token: missing subject or email")
        return None
    return UserInfo(sub=subject, email=email, name=name)


def is_admin_email(email: str) -> bool:
    """Return True if the email is whitelisted as an admin."""
    if not email:
        return False
    raw_list = settings.admin_email_whitelist or ""
    candidates = {item.strip().lower() for item in re.split(r"[,\s]+", raw_list.replace(";", ",")) if item.strip()}
    return email.strip().lower() in candidates


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> UserInfo:
    """FastAPI dependency returning user info or raising 401."""

    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")

    user_info = decode_token(credentials.credentials)
    if user_info is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return user_info


async def exchange_code_for_token(code: str) -> dict:
    """Exchange an authorization code for tokens at the provider's token endpoint."""

    token_url = get_token_endpoint()
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": settings.oidc_redirect_uri,
        "client_id": settings.oidc_client_id,
        "client_secret": settings.oidc_client_secret,
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(token_url, data=data)
        response.raise_for_status()
        return response.json()


async def get_current_user_websocket(websocket: WebSocket) -> Optional[UserInfo]:
    """Authenticate a WebSocket connection using token query param or header."""

    token = websocket.query_params.get("token")
    if token:
        token = token.strip().strip('"').strip("'") 
        if token.lower().startswith("bearer "):
            token = token[7:]

    if not token:
        auth_header = websocket.headers.get("Authorization") or websocket.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1]

    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None

    user_info = decode_token(token)
    if user_info is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None
    return user_info
