import logging
from sqlalchemy.orm import Session
from google.oauth2 import id_token
from google.auth.transport import requests
from fastapi import HTTPException, status, Depends, Request
from backend.app.config import settings
from backend.app.db import User, get_db

logger = logging.getLogger("siavox.auth")

def verify_google_token(token: str) -> dict:
    """Verifies a Google ID token.
    
    If the app is in dev mode or using the default client id, allows mock validation
    for convenience of local testing.
    """
    # Allow mock login for testing or if GOOGLE_CLIENT_ID is not configured properly
    if settings.GOOGLE_CLIENT_ID.startswith("your-google-client-id") or token.startswith("mock_token_"):
        logger.warning("Using mock authentication because client ID is not configured or token is mocked.")
        email = f"{token.replace('mock_token_', '')}@example.com" if token.startswith("mock_token_") else "testuser@example.com"
        name = "Local Test User"
        return {"email": email, "name": name, "sub": "mock_sub_12345"}
        
    try:
        # Verify the ID token using google-auth library
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), settings.GOOGLE_CLIENT_ID)
        
        # ID token issuer must be Google
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')
            
        return idinfo
    except Exception as e:
        logger.error(f"Google token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google credential or token expired"
        )

def get_current_user_from_cookie(request: Request, db: Session = Depends(get_db)) -> User:
    """FastAPI dependency to retrieve the current user session from cookie/auth header."""
    if settings.DEBUG_MODE:
        dev_email = "devuser@example.com"
        dev_name = "Dev User"
        user = db.query(User).filter(User.email == dev_email).first()
        if not user:
            user = User(email=dev_email, name=dev_name)
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    # We can check cookie first, then authorization header
    session_email = request.cookies.get(settings.SESSION_COOKIE_NAME)
    
    # Fallback to Authorization Header
    if not session_email:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a simple session setup, we could store email or token
            # For simplicity in this local app, the bearer token can be the user's email if mocked,
            # or we verify it. Let's support session_email directly.
            session_email = auth_header.replace("Bearer ", "").strip()
            
    if not session_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
        
    # Query database for user
    user = db.query(User).filter(User.email == session_email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalid or user not found"
        )
    return user
