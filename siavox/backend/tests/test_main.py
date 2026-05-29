import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.config import settings
from backend.app.db import Base, get_db
from backend.app.main import app

# Setup Test Database (in-memory SQLite)
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override get_db dependency
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def run_around_tests():
    # Setup test tables
    Base.metadata.create_all(bind=engine)
    yield
    # Teardown test tables
    Base.metadata.drop_all(bind=engine)

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ollama_connected" in data

def test_auth_config():
    response = client.get("/api/auth/config")
    assert response.status_code == 200
    data = response.json()
    assert "google_client_id" in data

def test_unauthenticated_me():
    response = client.get("/api/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is False

def test_google_mock_auth_and_session():
    # Attempt mock login
    login_payload = {"credential": "mock_token_test_dev_user"}
    response = client.post("/api/auth/google", json=login_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user"]["email"] == "test_dev_user@example.com"
    assert data["user"]["name"] == "Local Test User"
    
    # Store session cookies directly on the client instance to avoid deprecation warnings
    client.cookies = response.cookies
    
    try:
        # Check session validation with cookie
        me_response = client.get("/api/auth/me")
        assert me_response.status_code == 200
        me_data = me_response.json()
        assert me_data["authenticated"] is True
        assert me_data["user"]["email"] == "test_dev_user@example.com"
        
        # Check history fetch
        history_response = client.get("/api/history")
        assert history_response.status_code == 200
        assert isinstance(history_response.json(), list)
        assert len(history_response.json()) == 0
        
        # Logout check
        logout_response = client.post("/api/auth/logout")
        assert logout_response.status_code == 200
        assert logout_response.json()["success"] is True
    finally:
        client.cookies.clear()

def test_process_audio_success(monkeypatch):
    from unittest.mock import AsyncMock
    import io
    import os
    
    # Mock OllamaClient.process_audio method to return the new 4-tuple
    mock_process = AsyncMock(return_value=(
        "Hello space", 
        "English", 
        "Spanish", 
        "Hola espacio"
    ))
    monkeypatch.setattr("backend.app.main.ollama_client.process_audio", mock_process)
    
    # Authenticate client
    login_payload = {"credential": "mock_token_test_dev_user"}
    response = client.post("/api/auth/google", json=login_payload)
    assert response.status_code == 200
    client.cookies = response.cookies
    
    try:
        # Load sample audio file from res folder
        sample_path = os.path.join(os.path.dirname(__file__), "..", "..", "res", "sample-30s.wav")
        with open(sample_path, "rb") as f:
            audio_bytes = f.read()
        audio_file = io.BytesIO(audio_bytes)
        files = {
            "file": ("sample-30s.wav", audio_file, "audio/wav")
        }
        data = {
            "format": "note",
            "target_language": "auto",
            "custom_instructions": "translate to spanish"
        }
        
        process_response = client.post("/api/process", files=files, data=data)
        assert process_response.status_code == 200
        result = process_response.json()
        assert result["raw_transcript"] == "Hello space"
        assert result["source_language"] == "English"
        assert result["target_language"] == "Spanish"
        assert result["transformed_text"] == "Hola espacio"
        
        # Verify it was saved to history
        history_response = client.get("/api/history")
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert len(history_data) == 1
        assert history_data[0]["target_language"] == "Spanish"
    finally:
        client.cookies.clear()
