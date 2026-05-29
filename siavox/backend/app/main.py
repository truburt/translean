import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Response, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import uvicorn

from backend.app.config import settings
from backend.app.db import get_db, init_db, User, Interaction
from backend.app.auth import verify_google_token, get_current_user_from_cookie
from backend.app.ollama_client import OllamaClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to initialize database."""
    init_db()
    yield

# Setup FastAPI App
app = FastAPI(title="Siavox API", version="1.0.0", lifespan=lifespan)

# Setup CORS to allow local cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama HTTP Client
ollama_client = OllamaClient()

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/api/health")
async def health_check():
    """Health check endpoint testing connectivity."""
    ollama_ok = await ollama_client.check_connection()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ollama_connected": ollama_ok,
        "database": "connected"
    }

@app.post("/api/auth/google")
async def google_auth(request: Request, response: Response, payload: dict, db: Session = Depends(get_db)):
    """Google authentication endpoint.
    
    Verifies JWT token, creates user if they don't exist, and establishes session.
    """
    token = payload.get("credential")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Google identity credential token"
        )
        
    # Verify token with helper
    id_info = verify_google_token(token)
    email = id_info.get("email")
    name = id_info.get("name")
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google token did not contain email address"
        )
        
    # Check if user exists, else create
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email, name=name)
        db.add(user)
        db.commit()
        db.refresh(user)
        
    # Set HTTP-Only Session Cookie
    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=email,
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True in production with HTTPS
        max_age=86400 * 30  # 30 days session
    )
    
    return {
        "success": True,
        "user": {
            "email": user.email,
            "name": user.name
        }
    }

@app.post("/api/auth/logout")
async def logout(response: Response):
    """Logs out the user and clears the session cookie."""
    response.delete_cookie(settings.SESSION_COOKIE_NAME)
    return {"success": True, "message": "Successfully logged out"}

@app.get("/api/auth/me")
async def check_session(request: Request, db: Session = Depends(get_db)):
    """Returns current user profile if authenticated."""
    try:
        user = get_current_user_from_cookie(request, db)
        return {
            "authenticated": True,
            "user": {
                "email": user.email,
                "name": user.name
            }
        }
    except HTTPException:
        return {"authenticated": False}

@app.get("/api/auth/config")
async def get_auth_config():
    """Exposes public client configurations such as Google Client ID."""
    return {
        "google_client_id": settings.GOOGLE_CLIENT_ID
    }

@app.get("/api/history")
async def get_history(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_from_cookie)
):
    """Retrieves user's historical speech interaction notes/messages."""
    interactions = (
        db.query(Interaction)
        .filter(Interaction.user_id == current_user.id)
        .order_by(Interaction.created_at.desc())
        .all()
    )
    
    return [
        {
            "id": item.id,
            "audio_path": item.audio_path,
            "raw_transcript": item.raw_transcript,
            "transformed_text": item.transformed_text,
            "format": item.format,
            "source_language": item.source_language,
            "target_language": item.target_language,
            "created_at": item.created_at.isoformat()
        }
        for item in interactions
    ]

@app.post("/api/process")
async def process_audio(
    request: Request,
    file: UploadFile = File(...),
    format: str = Form("note"),  # "note" or "message"
    target_language: str = Form("auto"),
    custom_instructions: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_from_cookie)
):
    """Processes uploaded 16kHz mono WAV file, sends it to Ollama, and persists it."""
    # Validate format parameter
    if format not in ["note", "message"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid format selected. Must be 'note' or 'message'"
        )

    # Save audio file locally
    filename = f"{current_user.id}_{int(datetime.now(timezone.utc).timestamp())}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save audio file locally: {e}"
        )

    # Read audio bytes for Ollama
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read audio file: {e}"
        )

    # Dispatch to Ollama client
    try:
        raw_transcript, source_language, detected_target_language, transformed_text = await ollama_client.process_audio(
            audio_bytes=audio_bytes,
            output_format=format,
            target_language=target_language,
            custom_instructions=custom_instructions
        )
    except Exception as e:
        # Cleanup file if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)
            
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ollama integration error: {e}"
        )

    # Save interaction to DB
    interaction = Interaction(
        user_id=current_user.id,
        audio_path=f"/static/uploads/{filename}",  # URL path
        raw_transcript=raw_transcript,
        transformed_text=transformed_text,
        format=format,
        source_language=source_language,
        target_language=detected_target_language
    )
    
    db.add(interaction)
    db.commit()
    db.refresh(interaction)

    return {
        "id": interaction.id,
        "raw_transcript": interaction.raw_transcript,
        "transformed_text": interaction.transformed_text,
        "format": interaction.format,
        "source_language": interaction.source_language,
        "target_language": interaction.target_language,
        "created_at": interaction.created_at.isoformat(),
        "audio_path": interaction.audio_path
    }

# Serve uploaded static audio files
app.mount("/static/uploads", StaticFiles(directory="uploads"), name="uploads")

# Serve static frontend web files if the frontend folder exists (useful for local development without Docker/Nginx proxy)
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
