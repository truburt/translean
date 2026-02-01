"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
"""
import pytest
from unittest.mock import MagicMock, patch
from app import auth
from app.config import settings

@pytest.mark.asyncio
async def test_oidc_discovery_sucess(monkeypatch):
    """Test that authorization and token endpoints are discovered correctly."""
    monkeypatch.setattr(settings, "oidc_issuer_url", "https://issuer.com")
    
    mock_config = {
        "jwks_uri": "https://issuer.com/jwks",
        "authorization_endpoint": "https://issuer.com/auth/custom",
        "token_endpoint": "https://issuer.com/token/custom"
    }
    
    mock_jwks = {"keys": []}

    with patch("httpx.get") as mock_get:
        # Define side effects for httpx.get
        def side_effect(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            if url.endswith("/.well-known/openid-configuration"):
                mock_resp.json.return_value = mock_config
            elif url == "https://issuer.com/jwks":
                mock_resp.json.return_value = mock_jwks
            else:
                mock_resp.json.return_value = {}
            return mock_resp

        mock_get.side_effect = side_effect
        
        # Clear cache before test
        auth._jwks_cache.cache_clear()
        
        # Act
        auth_endpoint = auth.get_authorization_endpoint()
        token_endpoint = auth.get_token_endpoint()
        
        # Assert
        assert auth_endpoint == "https://issuer.com/auth/custom"
        assert token_endpoint == "https://issuer.com/token/custom"
        
        # Verify call count - should cache after first refresh
        assert mock_get.call_count >= 1

@pytest.mark.asyncio
async def test_oidc_discovery_failure_fallback(monkeypatch):
    """Test fallback to constructed URLs when discovery fails."""
    monkeypatch.setattr(settings, "oidc_issuer_url", "https://issuer.com")

    with patch("httpx.get") as mock_get:
        mock_get.side_effect = Exception("Discovery failed")
        
        # Clear cache
        auth._jwks_cache.cache_clear()
        
        # Act
        auth_endpoint = auth.get_authorization_endpoint()
        token_endpoint = auth.get_token_endpoint()
        
        # Assert fallback logic
        assert auth_endpoint == "https://issuer.com/authorize"
        assert token_endpoint == "https://issuer.com/token" 
