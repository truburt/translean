
import pytest
from httpx import AsyncClient, ASGITransport
from app.config import settings
from app.main import app 

@pytest.mark.asyncio
async def test_dev_auth_bypass_enabled(monkeypatch):
    monkeypatch.setattr(settings, "dev_mode", True)
    monkeypatch.setattr(settings, "oidc_client_secret", "test-secret")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/auth/login")
        
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "localStorage.setItem('access_token'" in response.text
    assert "Dev User" in response.text

@pytest.mark.asyncio
async def test_dev_auth_bypass_disabled(monkeypatch):
    monkeypatch.setattr(settings, "dev_mode", False)
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/auth/login")
        
    assert response.status_code == 307 # RedirectResponse default status code
    assert "authorize" in response.headers["location"]
