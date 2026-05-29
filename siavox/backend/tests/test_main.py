import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.config import settings
# Force DEBUG_MODE to False for standard tests
settings.DEBUG_MODE = False
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

def test_debug_mode_bypass(monkeypatch):
    # Enable debug mode using monkeypatch
    monkeypatch.setattr(settings, "DEBUG_MODE", True)
    
    # Verify that requesting me endpoint without session cookies/headers works
    # and returns the dev user
    response = client.get("/api/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is True
    assert data["user"]["email"] == "devuser@example.com"
    assert data["user"]["name"] == "Dev User"

    # Verify that another protected endpoint (like /api/history) is also bypassed
    history_response = client.get("/api/history")
    assert history_response.status_code == 200
    assert isinstance(history_response.json(), list)

@pytest.mark.anyio
async def test_ollama_client_is_model_loaded_connection_fails(monkeypatch):
    import httpx2 as httpx
    from backend.app.ollama_client import OllamaClient
    
    async def mock_get(self, url, **kwargs):
        raise httpx.RequestError("Connection failed")
        
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    
    client = OllamaClient()
    loaded = await client.is_model_loaded()
    assert loaded is False

@pytest.mark.anyio
async def test_ollama_client_is_model_loaded_success(monkeypatch):
    import httpx2 as httpx
    from backend.app.ollama_client import OllamaClient
    
    async def mock_get(self, url, **kwargs):
        class MockResponse:
            status_code = 200
            def json(self):
                return {
                    "models": [
                        {
                            "name": "gemma4:e4b",
                            "model": "gemma4:e4b"
                        }
                    ]
                }
        return MockResponse()
        
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    
    client = OllamaClient(model="gemma4:e4b")
    loaded = await client.is_model_loaded()
    assert loaded is True

@pytest.mark.anyio
async def test_ollama_client_is_model_loaded_different_model(monkeypatch):
    import httpx2 as httpx
    from backend.app.ollama_client import OllamaClient
    
    async def mock_get(self, url, **kwargs):
        class MockResponse:
            status_code = 200
            def json(self):
                return {
                    "models": [
                        {
                            "name": "llama3:latest",
                            "model": "llama3:latest"
                        }
                    ]
                }
        return MockResponse()
        
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    
    client = OllamaClient(model="gemma4:e4b")
    loaded = await client.is_model_loaded()
    assert loaded is False

def test_api_health_endpoint_integration(monkeypatch):
    from unittest.mock import AsyncMock
    
    # Mock OllamaClient check_connection and is_model_loaded
    mock_conn = AsyncMock(return_value=True)
    mock_loaded = AsyncMock(return_value=True)
    
    monkeypatch.setattr("backend.app.main.ollama_client.check_connection", mock_conn)
    monkeypatch.setattr("backend.app.main.ollama_client.is_model_loaded", mock_loaded)
    
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ollama_connected"] is True
    assert data["ollama_model_loaded"] is True

def test_transcribe_and_refine_endpoints(monkeypatch):
    from unittest.mock import AsyncMock
    import io
    import os
    
    # Mock OllamaClient transcribe_audio and refine_transcript methods
    mock_transcribe = AsyncMock(return_value="verbatim speech segment")
    mock_refine = AsyncMock(return_value=("Russian", "Russian", "окончательный текст"))
    
    monkeypatch.setattr("backend.app.main.ollama_client.transcribe_audio", mock_transcribe)
    monkeypatch.setattr("backend.app.main.ollama_client.refine_transcript", mock_refine)
    
    # Authenticate client
    login_payload = {"credential": "mock_token_test_dev_user"}
    response = client.post("/api/auth/google", json=login_payload)
    assert response.status_code == 200
    client.cookies = response.cookies
    
    try:
        # Test /api/transcribe
        sample_path = os.path.join(os.path.dirname(__file__), "..", "..", "res", "sample-30s.wav")
        with open(sample_path, "rb") as f:
            audio_bytes = f.read()
        
        files = {
            "file": ("sample-30s.wav", io.BytesIO(audio_bytes), "audio/wav")
        }
        
        transcribe_res = client.post("/api/transcribe", files=files)
        assert transcribe_res.status_code == 200
        transcribe_data = transcribe_res.json()
        assert transcribe_data["raw_transcript"] == "verbatim speech segment"
        assert "audio_path" in transcribe_data
        
        # Test /api/refine
        refine_payload = {
            "raw_transcript": "verbatim speech segment",
            "audio_path": transcribe_data["audio_path"],
            "format": "note",
            "target_language": "Russian"
        }
        
        refine_res = client.post("/api/refine", data=refine_payload)
        assert refine_res.status_code == 200
        refine_data = refine_res.json()
        assert refine_data["raw_transcript"] == "verbatim speech segment"
        assert refine_data["source_language"] == "Russian"
        assert refine_data["target_language"] == "Russian"
        assert refine_data["transformed_text"] == "окончательный текст"
        
        # Verify it was saved to history
        history_response = client.get("/api/history")
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert len(history_data) == 1
        assert history_data[0]["transformed_text"] == "окончательный текст"
    finally:
        client.cookies.clear()


def test_transcribe_short_chunk(monkeypatch):
    """Verify /api/transcribe handles short WAV chunks (≤7 s) produced by streaming recording."""
    from unittest.mock import AsyncMock
    import io
    import struct
    import wave as wave_module

    # Build a minimal valid 2-second silent WAV (16kHz, mono, 16-bit PCM)
    sample_rate = 16000
    duration_s = 2
    n_samples = sample_rate * duration_s
    pcm_data = b'\x00\x00' * n_samples  # silence

    wav_buf = io.BytesIO()
    with wave_module.open(wav_buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    wav_bytes = wav_buf.getvalue()

    mock_transcribe = AsyncMock(return_value="short chunk text")
    monkeypatch.setattr("backend.app.main.ollama_client.transcribe_audio", mock_transcribe)

    # Authenticate
    login_payload = {"credential": "mock_token_test_dev_user"}
    resp = client.post("/api/auth/google", json=login_payload)
    assert resp.status_code == 200
    client.cookies = resp.cookies

    try:
        files = {"file": ("chunk_1.wav", io.BytesIO(wav_bytes), "audio/wav")}
        res = client.post("/api/transcribe", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["raw_transcript"] == "short chunk text"
        assert "audio_path" in data
        # Audio path should reflect the uploaded chunk filename
        assert "chunk_1.wav" in data["audio_path"]
    finally:
        client.cookies.clear()


def test_transcribe_chunk_filenames_are_unique(monkeypatch):
    """Verify that two rapid chunk uploads for the same user produce unique filenames.

    Before the UUID fix, two requests within the same second would generate the same
    filename (user_id + timestamp + original_filename), causing the second upload to
    silently overwrite the first. The UUID suffix ensures this cannot happen.
    """
    from unittest.mock import AsyncMock
    import io
    import wave as wave_module

    # Build a minimal 2-second WAV
    sample_rate = 16000
    pcm_data = b'\x00\x00' * (sample_rate * 2)
    def make_wav():
        buf = io.BytesIO()
        with wave_module.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    mock_transcribe = AsyncMock(return_value="some text")
    monkeypatch.setattr("backend.app.main.ollama_client.transcribe_audio", mock_transcribe)

    # Authenticate
    login_payload = {"credential": "mock_token_test_dev_user"}
    resp = client.post("/api/auth/google", json=login_payload)
    assert resp.status_code == 200
    client.cookies = resp.cookies

    try:
        # Upload the same filename twice in rapid succession
        common_name = "chunk_rapid.wav"
        wav_bytes = make_wav()

        res1 = client.post("/api/transcribe", files={"file": (common_name, io.BytesIO(wav_bytes), "audio/wav")})
        res2 = client.post("/api/transcribe", files={"file": (common_name, io.BytesIO(wav_bytes), "audio/wav")})

        assert res1.status_code == 200
        assert res2.status_code == 200

        path1 = res1.json()["audio_path"]
        path2 = res2.json()["audio_path"]

        # Paths must be different (UUID suffix ensures uniqueness)
        assert path1 != path2, (
            f"Filename collision detected: both uploads returned '{path1}'. "
            "Ensure the UUID suffix is applied in /api/transcribe."
        )
    finally:
        client.cookies.clear()
