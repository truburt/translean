import os
import pytest
import io
import httpx2 as httpx
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
    print(f"\n[FIXTURE] Creating all tables on {engine.url}")
    Base.metadata.create_all(bind=engine)
    print(f"[FIXTURE] Tables registered on Base.metadata: {list(Base.metadata.tables.keys())}")
    yield
    print(f"\n[FIXTURE] Dropping all tables on {engine.url}")
    Base.metadata.drop_all(bind=engine)

client = TestClient(app)

# Check Ollama service availability before running e2e tests
@pytest.fixture(scope="session", autouse=True)
def check_ollama_service():
    try:
        response = httpx.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5.0)
        if response.status_code != 200:
            pytest.skip(f"Ollama server returned {response.status_code}")
        models = [m.get("name") for m in response.json().get("models", [])]
        required_model = settings.OLLAMA_MODEL
        if required_model not in models and f"{required_model}:latest" not in models:
            pytest.skip(f"Required Ollama model '{required_model}' not found in local models: {models}")
    except Exception as e:
        pytest.skip(f"Ollama server is not reachable at {settings.OLLAMA_HOST}: {e}")

@pytest.mark.e2e
def test_e2e_full_flow(run_around_tests, monkeypatch):
    import json
    
    mock_responses = [
        # Call 1: Boxing Single-chunk Flow (both verbatim transcript and transformed text in 1 response)
        {
            "message": {
                "content": '{"raw_transcript": "The history of boxing blazes with names like Sullivan, Leonard, Johnson, Ross, and other great champions. But in boxing’s roll of honor, none stands above those of Jack Dempsey and Joe Louis. The date: July 4th, 1919. Jess Willard. Challenger: Jack Dempsey.", "source_language": "English", "target_language": "English", "transformed_text": "- Boxing history features champions like Sullivan, Leonard, Johnson, and Ross.\\n- Jack Dempsey and Joe Louis stand above all others.\\n- On July 4, 1919, challenger Jack Dempsey faced heavyweight champion Jess Willard."}'
            }
        },
        # Call 2: NATO Chunk 1 (verbatim transcript of first 30s)
        {
            "message": {
                "content": '{"raw_transcript": "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango", "source_language": "English"}'
            }
        },
        # Call 3: NATO Chunk 2 (verbatim transcript of remaining 23.5s)
        {
            "message": {
                "content": '{"raw_transcript": "uniform victor whiskey xray yankee zulu", "source_language": "English"}'
            }
        },
        # Call 4: NATO Refinement (combines transcripts and formats)
        {
            "message": {
                "content": '{"source_language": "English", "target_language": "English", "transformed_text": "- NATO Phonetic Alphabet items detected: alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india, juliet, kilo, lima, mike, november, oscar, papa, quebec, romeo, sierra, tango, uniform, victor, whiskey, xray, yankee, zulu."}'
            }
        }
    ]
    
    call_index = 0
    original_post = httpx.AsyncClient.post
    
    async def mock_post(client_self, url, **kwargs):
        nonlocal call_index
        if "/api/chat" in str(url):
            if call_index >= len(mock_responses):
                raise ValueError("Unexpected extra call to Ollama API")
            res_data = mock_responses[call_index]
            call_index += 1
            
            class MockResponse:
                def __init__(self, data):
                    self.status_code = 200
                    self._data = data
                    self.text = json.dumps(data)
                def json(self):
                    return self._data
                    
            return MockResponse(res_data)
        return await original_post(client_self, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # 1. Start unauthenticated: /api/auth/me should return authenticated = False
    response = client.get("/api/auth/me")
    assert response.status_code == 200
    assert response.json()["authenticated"] is False

    # 2. Login via Google OAuth Mock Sandbox
    login_payload = {"credential": "mock_token_test_dev_user"}
    response = client.post("/api/auth/google", json=login_payload)
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["user"]["email"] == "test_dev_user@example.com"
    client.cookies = response.cookies

    try:
        # Verify authenticated session works
        response = client.get("/api/auth/me")
        assert response.status_code == 200
        assert response.json()["authenticated"] is True
        assert response.json()["user"]["email"] == "test_dev_user@example.com"

        # 3. Case 1: Upload sample-30s.wav (Duration: 30s - single chunk flow)
        sample_30s_path = os.path.join(os.path.dirname(__file__), "..", "..", "res", "sample-30s.wav")
        assert os.path.exists(sample_30s_path), f"File not found: {sample_30s_path}"
        
        with open(sample_30s_path, "rb") as f:
            audio_bytes_30s = f.read()
            
        files_30s = {
            "file": ("sample-30s.wav", io.BytesIO(audio_bytes_30s), "audio/wav")
        }
        data_30s = {
            "format": "note",
            "target_language": "auto",
            "custom_instructions": "transcribe verbatim"
        }
        
        print("\nProcessing sample-30s.wav via e2e API (single chunk)...")
        response_30s = client.post("/api/process", files=files_30s, data=data_30s)
        assert response_30s.status_code == 200
        result_30s = response_30s.json()
        assert "raw_transcript" in result_30s
        assert "transformed_text" in result_30s
        # Verify boxing transcript words are present
        assert "boxing" in result_30s["raw_transcript"].lower() or "dempsey" in result_30s["raw_transcript"].lower()

        # 4. Case 2: Upload sample-nato.wav (Duration: 53.5s - multi-chunk flow)
        sample_nato_path = os.path.join(os.path.dirname(__file__), "..", "..", "res", "sample-nato.wav")
        assert os.path.exists(sample_nato_path), f"File not found: {sample_nato_path}"
        
        with open(sample_nato_path, "rb") as f:
            audio_bytes_nato = f.read()
            
        files_nato = {
            "file": ("sample-nato.wav", io.BytesIO(audio_bytes_nato), "audio/wav")
        }
        data_nato = {
            "format": "note",
            "target_language": "auto",
            "custom_instructions": "transcribe verbatim"
        }
        
        print("Processing sample-nato.wav via e2e API (multi-chunk)...")
        response_nato = client.post("/api/process", files=files_nato, data=data_nato)
        assert response_nato.status_code == 200
        result_nato = response_nato.json()
        assert "raw_transcript" in result_nato
        assert "transformed_text" in result_nato
        
        # Verify NATO alphabet phonetic words are present in the transcript (proving chunking worked across segments)
        transcript_lower = result_nato["raw_transcript"].lower()
        # Look for a few phonetic spelling words from start/middle/end of the audio file (alpha, tango, zulu)
        assert "alpha" in transcript_lower or "bravo" in transcript_lower
        assert "tango" in transcript_lower or "romeo" in transcript_lower
        assert "zulu" in transcript_lower or "yankee" in transcript_lower

        # 5. History Retrieval: Verify both records exist in /api/history
        history_response = client.get("/api/history")
        assert history_response.status_code == 200
        history = history_response.json()
        assert len(history) == 2
        # Most recent first
        assert history[0]["audio_path"].endswith("sample-nato.wav")
        assert history[1]["audio_path"].endswith("sample-30s.wav")

        # 6. Logout
        logout_response = client.post("/api/auth/logout")
        assert logout_response.status_code == 200
        assert logout_response.json()["success"] is True

        # Verify unauthenticated again
        me_response = client.get("/api/auth/me")
        assert me_response.status_code == 200
        assert me_response.json()["authenticated"] is False

        # Assert call_index == 4 (verifies 1 Ollama call for sample-30s.wav and 3 calls for sample-nato.wav, confirming chunking worked)
        assert call_index == 4

    finally:
        client.cookies.clear()
        # Clean up any files created in uploads directory during testing
        if os.path.exists("uploads"):
            for f in os.listdir("uploads"):
                if f.startswith("1_"): # User ID is 1
                    try:
                        os.remove(os.path.join("uploads", f))
                    except Exception:
                        pass
