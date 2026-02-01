
import asyncio
import pytest
from app import whisper_client as wc
from app.config import settings

class MockResponse:
    def __init__(self, payload: dict):
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload

class MockClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    async def post(self, url, files=None, data=None):
        wav_data = files["file"][1]
        # WAV header is 44 bytes. Payload is len - 44.
        pcm_len = len(wav_data) - 44
        self.calls.append(pcm_len)
        
        # Always return unstable text to prevent stabilization (which would trim buffer correctly)
        payload = self.payloads[0] if self.payloads else {"text": "", "language": "en", "segments": []}
        return MockResponse(payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

class DummyTranscoder:
    def __init__(self, pcm_buffer, **kwargs):
        self.pcm_buffer = pcm_buffer
        self.on_pcm = kwargs.get("on_pcm")

    async def start(self):
        pass
        
    async def write(self, data):
        self.pcm_buffer.extend(data)
        if self.on_pcm:
            self.on_pcm()
            
    async def close(self):
        pass

@pytest.mark.asyncio
async def test_buffer_trimming_data_loss(monkeypatch):
    """
    Reproduces the issue where buffer is trimmed to STABLE_WINDOW_SECONDS (5.0s)
    even when no text is committed.
    """
    monkeypatch.setattr(settings, "whisper_base_url", "http://test")
    monkeypatch.setattr(wc, "ContainerToPCMTranscoder", DummyTranscoder)
    
    # Disable throttling
    #monkeypatch.setattr(wc, "STABLE_THROTTLE_SECONDS", 0.0)
    #monkeypatch.setattr(wc, "PREVIEW_THROTTLE_SECONDS", 1000.0)
    
    chunk_size = 32000 # 1 second
    # Send enough chunks to exceed 5.0 seconds
    # 7 chunks = 7 seconds
    chunks = [b"\x01" * chunk_size for _ in range(7)]
    
    header = b"\x1a\x45\xdf\xa3"
    
    async def audio_gen():
        yield header
        for i, c in enumerate(chunks):
            print(f"Yielding chunk {i} out of {len(chunks)}")
            yield c
            # Minimal yield to let the consumer loop run
            await asyncio.sleep(0.001)
        yield {"type": "stop_recording"}

    # Mock Whisper response: Always unstable
    long_segment = {
        "text": "unstable text...",
        "language": "en",
        "segments": [{
            "start": 0.0,
            "end": 6.0,
            "text": "unstable text..."
        }]
    }
    
    mock_client = MockClient([long_segment])
    monkeypatch.setattr(wc.httpx, "AsyncClient", lambda **kwargs: mock_client)

    results = []
    async for item in wc.stream_transcription(audio_gen(), "en"):
        results.append(item)
        
    print(f"Mock calls (PCM bytes): {mock_client.calls}")
    
    # We expect the payload size to increase: 32000, 64000, 96000, 128000, 160000 (5s).
    # Then for 6th second (192000), if blindly trimmed to 5s BEFORE this call? No, it trims AFTER.
    # So we might see 192000.
    # But then for 7th second, if trimmed to 5s previously, we would see 5s + 1s = 6s (192000).
    # If NOT trimmed, we would see 7s (224000).
    
    max_payload = max(mock_client.calls)
    
    # 7 seconds = 7 * 32000 = 224000 bytes
    # If the bug is present, the buffer never grows much beyond 5s + 1 chunk.
    # So max might be around 192000 (6s) or slightly more, but definitely not 224000 (7s).
    
    # We assert that we DID reach full 7s size at least once (likely the last flush).
    # If buffer was trimmed, the flush will be small (5s).
    
    # Expectation: 
    # With bug: Max payload ~ 192000 (6s) or 160000 (5s) (if trimming happened aggressively)
    # Fixed: Max payload should reach 224000 (7s)
    
    assert max_payload >= 220000, f"Buffer never reached 7s! Max saw: {max_payload}"
