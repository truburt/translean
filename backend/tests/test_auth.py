"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
"""
import pytest
from jose import jwt

from app import auth
from app.config import settings


@pytest.mark.asyncio
async def test_decode_token_hs256_roundtrip(monkeypatch):
    monkeypatch.setattr(settings, "oidc_client_secret", "super-secret")
    monkeypatch.setattr(settings, "oidc_client_id", "my-client")

    token = jwt.encode(
        {"sub": "abc", "email": "user@example.com", "aud": "my-client"},
        settings.oidc_client_secret,
        algorithm="HS256"
    )
    user = auth.decode_token(token)
    assert user is not None
    assert user.sub == "abc"


@pytest.mark.asyncio
async def test_decode_token_invalid(monkeypatch):
    monkeypatch.setattr(settings, "oidc_client_secret", "super-secret")
    token = jwt.encode({"foo": "bar"}, settings.oidc_client_secret, algorithm="HS256")
    assert auth.decode_token(token) is None
