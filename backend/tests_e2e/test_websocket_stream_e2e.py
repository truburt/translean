import asyncio
import logging

import contextlib
import difflib
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

# Ensure backend/app is importable when running from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import pytest
import websockets
from jose import jwt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAMPLE_WEBM = Path(__file__).resolve().parents[1] / "tests" / "sample-30s.webm"
SAMPLE_TEXT = Path(__file__).resolve().parents[1] / "tests" / "sample-30s.txt"
NATO_WEBM = Path(__file__).resolve().parents[1] / "tests" / "sample-nato.webm"
NATO_TIMECODES = Path(__file__).resolve().parents[1] / "tests" / "sample-nato-timecodes.txt"
PCM_BYTES_PER_SECOND = 32000.0


def _assert_transcript_quality(
    expected_text: str,
    combined_text: str,
    *,
    token_threshold: float = 0.95,
    similarity_threshold: float = 0.80,
    required_tokens: list[str] | None = None,
) -> None:
    """Compare transcript quality using token overlap and fuzzy ratio."""
    def normalize(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    norm_expected = normalize(expected_text)
    norm_actual = normalize(combined_text)
    expected_tokens = tokenize(expected_text)
    actual_tokens = tokenize(combined_text)

    logger.info(f"Normalized Expected: {norm_expected}")
    logger.info(f"Normalized Actual:   {norm_actual}")

    token_overlap = len(expected_tokens & actual_tokens) / max(len(expected_tokens), 1)
    logger.info(f"Token Overlap: {token_overlap:.4f}")

    matcher = difflib.SequenceMatcher(None, norm_expected, norm_actual)
    similarity = matcher.ratio()
    logger.info(f"Similarity: {similarity:.4f}")

    if required_tokens:
        missing = [t for t in required_tokens if t not in actual_tokens]
        assert not missing, f"Missing required tokens in transcript: {missing}"

    assert token_overlap >= token_threshold, f"Expected token overlap >= {token_threshold:.2f}, got {token_overlap:.4f}"
    assert similarity >= similarity_threshold, f"Expected similarity >= {similarity_threshold:.2f}, got {similarity:.4f}"


def _probe_duration_seconds(media_path: Path) -> float:
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            stderr=subprocess.STDOUT,
        )
        return float(output.decode().strip())
    except Exception as exc:
        logger.warning("ffprobe duration probe failed: %s", exc)
        return 0.0


def _estimate_bytes_per_second(webm_bytes: bytes, media_path: Path = SAMPLE_WEBM) -> float:
    duration = _probe_duration_seconds(media_path)
    if duration > 0:
        return len(webm_bytes) / duration
    return PCM_BYTES_PER_SECOND


def _load_timecoded_words(path: Path) -> list[tuple[float, float, str]]:
    """Parse timecoded word entries for targeted transcription checks."""
    entries: list[tuple[float, float, str]] = []
    raw = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw_line in enumerate(raw, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid timecode line {line_no}: {raw_line}")
        start_s, end_s = float(parts[0]), float(parts[1])
        word = parts[2].lower()
        entries.append((start_s, end_s, word))
    if not entries:
        raise ValueError(f"No timecoded entries found in {path}")
    return entries


def _extract_silence_markers(entries: list[tuple[float, float, str]], min_gap_s: float = 0.9) -> list[float]:
    """Detect intended long gaps between words for silence injection."""
    markers: list[float] = []
    for (prev_start, prev_end, _), (next_start, _, _) in zip(entries, entries[1:]):
        gap = next_start - prev_end
        if gap >= min_gap_s:
            markers.append(prev_end)
    return markers


def _collect_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    normalized: set[str] = set()
    aliases = {"juliett": "juliet", "whisky": "whiskey", "xray": "xray"}
    for token in tokens:
        normalized.add(aliases.get(token, token))
    for first, second in zip(tokens, tokens[1:]):
        merged = first + second
        normalized.add(aliases.get(merged, merged))
    return normalized


def _assert_transcript_covers_words(expected_words: list[str], combined_text: str) -> None:
    """Assert that every expected word appears in the transcript output."""
    actual_tokens = _collect_tokens(combined_text)
    missing = [word for word in expected_words if word not in actual_tokens]
    assert not missing, f"Missing tokens in transcript: {missing}"


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def backend_server() -> Generator[str, None, None]:
    """Launch the backend server in a subprocess on a free port."""
    port = get_free_port()
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"
    env = os.environ.copy()
    
    # Configure environment for test server
    env["DEV_MODE"] = "true"
    env["API_TOKEN"] = "test-token"
    env["OIDC_CLIENT_SECRET"] = "test-secret"
    env["OIDC_CLIENT_ID"] = "test-client"
    env["DATABASE_URL"] = "sqlite+aiosqlite:///./test_e2e_ws.db"
    
    # Pass through Whisper configuration or set default for the subprocess
    if "WHISPER_BASE_URL" not in env:
        env["WHISPER_BASE_URL"] = os.getenv("E2E_WHISPER_BASE_URL", "http://localhost:7860")
    if "WHISPER_MODEL" not in env:
        env["WHISPER_MODEL"] = os.getenv("E2E_WHISPER_MODEL", "Systran/faster-whisper-large-v3")

    # Command to run uvicorn
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    logger.info(f"Starting backend on {base_url}...")
    # Start the process
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr, # Pipe stderr to see errors if it fails
    )

    # Health check loop
    ready = False
    start = time.time()
    while time.time() - start < 30:  # 30s timeout
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    ready = True
                    break
        except (httpx.ConnectError, httpx.ReadTimeout):
            time.sleep(0.5)
            # Check if process died
            if proc.poll() is not None:
                raise RuntimeError(f"Backend process died prematurely with code {proc.returncode}")

    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError(f"Backend failed to start on {base_url} within 30s")

    yield base_url

    # Teardown
    logger.info("Stopping backend...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    
    # Clean up temp db
    if os.path.exists("./test_e2e_ws.db"):
        with contextlib.suppress(OSError):
            os.remove("./test_e2e_ws.db")


@pytest.fixture(scope="module")
def auth_token() -> str:
    """Generate a valid JWT for the test server."""
    payload = {
        "sub": "test-user-e2e",
        "email": "e2e@test.local",
        "name": "E2E User",
        "iat": datetime.now(timezone.utc),
    }
    # Secret must match OIDC_CLIENT_SECRET env var passed to server
    return jwt.encode(payload, "test-secret", algorithm="HS256")


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_websocket_streams_sample_audio(backend_server: str, auth_token: str):
    """Stream sample audio like the web client and assert final transcript output."""
    ws_url = backend_server.replace("http://", "ws://").replace("https://", "wss://") + "/ws/stream"
    timeout = float(os.getenv("E2E_WS_TIMEOUT", "120"))
    warmup_timeout = float(os.getenv("E2E_WS_WARMUP_TIMEOUT", "200"))
    chunk_size = int(os.getenv("E2E_WS_CHUNK_SIZE", "4096"))
    chunk_delay = float(os.getenv("E2E_WS_CHUNK_DELAY", "0.01"))

    # We assume the backend subprocess talks to a Whisper service (local or remote)
    # The backend handles its own warmup upon WebSocket connection.

    audio_data = SAMPLE_WEBM.read_bytes()
    expected_text = SAMPLE_TEXT.read_text(encoding="utf-8")

    final_paragraphs: dict[str | int, str] = {}
    final_order: list[str | int] = []
    start_time = time.monotonic()

    logger.info(f"Connecting to {ws_url}...")
    async with websockets.connect(
        ws_url,
        additional_headers={"Authorization": f"Bearer {auth_token}"},
        max_size=2**23,
    ) as websocket:
        # 1. Send Initialization
        await websocket.send(
            json.dumps(
                {
                    "source_language": "en",
                    "target_language": "en",
                    "title": "E2E WebSocket Sample",
                }
            )
        )

        # 2. Wait for Backend Warmup/Ready
        logger.info("Waiting for backend warmup...")
        while True:
            # Enforce a sub-timeout for startup so we don't hang forever if broken
            init_msg = await asyncio.wait_for(websocket.recv(), timeout=warmup_timeout)
            if isinstance(init_msg, bytes):
                continue
            
            data = json.loads(init_msg)
            status = data.get("status")
            
            if status == "ready":
                logger.info("Backend ready. Starting stream.")
                break
            elif status == "llm_ready":
                logger.info("Backend fully ready (llm_ready). Starting stream.")
                break
            elif status == "error":
                pytest.fail(f"Backend reported error during warmup: {data.get('error')}")
            elif status == "warming_up":
                logger.info("Backend warming up...")
            else:
                # Other messages like conversation_id info; ignore or log
                logger.info(f"Received pre-ready message: {data}")

        # 3. Stream Audio
        logger.info(f"Streaming {len(audio_data)} bytes...")
        for offset in range(0, len(audio_data), chunk_size):
            await websocket.send(audio_data[offset:offset + chunk_size])
            if chunk_delay:
                await asyncio.sleep(chunk_delay)

        await websocket.send(json.dumps({"type": "stop_recording"}))

        # 4. Wait for Processing Complete
        logger.info("Waiting for processing completion...")
        while True:
            remaining = timeout - (time.monotonic() - start_time)
            if remaining <= 0:
                logger.warning("Timeout waiting for completion.")
                break

            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
                
            if isinstance(message, bytes):
                continue

            payload = json.loads(message)
            if payload.get("status") == "processing_complete":
                break

            if payload.get("is_final") and payload.get("source"):
                par_key: str | int = payload.get("paragraph_index")
                if par_key is None:
                    par_key = payload.get("paragraph_id") or len(final_paragraphs)
                if par_key not in final_order:
                    final_order.append(par_key)
                final_paragraphs[par_key] = payload["source"]

    assert final_paragraphs
    final_sources = [final_paragraphs[k] for k in final_order]
    combined_text = " ".join(final_sources)
    logger.info(f"Combined Text: {combined_text}")

    _assert_transcript_quality(expected_text, combined_text)


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_websocket_streams_sample_audio_with_gaps(backend_server: str, auth_token: str):
    """Stream sample audio in ~0.5s randomized chunks with intentional silence gaps."""
    ws_url = backend_server.replace("http://", "ws://").replace("https://", "wss://") + "/ws/stream"
    timeout = float(os.getenv("E2E_WS_TIMEOUT", "120"))
    warmup_timeout = float(os.getenv("E2E_WS_WARMUP_TIMEOUT", "200"))

    audio_data = SAMPLE_WEBM.read_bytes()
    expected_text = SAMPLE_TEXT.read_text(encoding="utf-8")

    bytes_per_sec = _estimate_bytes_per_second(audio_data)
    rng = random.Random(1234)
    gap_markers = [10.0, 20.0]  # seconds into the stream to pause for silence

    final_paragraphs: dict[str | int, str] = {}
    final_order: list[str | int] = []
    start_time = time.monotonic()
    playback_time = 0.0

    logger.info(f"Connecting to {ws_url} (bps={bytes_per_sec:.1f})...")
    async with websockets.connect(
        ws_url,
        additional_headers={"Authorization": f"Bearer {auth_token}"},
        max_size=2**23,
        ping_interval=None,
        ping_timeout=None,
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "source_language": "en",
                    "target_language": "en",
                    "title": "E2E WebSocket Sample (Gaps)",
                }
            )
        )

        # Wait for backend readiness
        while True:
            init_msg = await asyncio.wait_for(websocket.recv(), timeout=warmup_timeout)
            if isinstance(init_msg, bytes):
                continue
            data = json.loads(init_msg)
            status = data.get("status")
            if status == "ready":
                break
            if status == "llm_ready":
                logger.info("Backend fully ready (llm_ready). Starting stream.")
                break
            if status == "error":
                pytest.fail(f"Backend reported error during warmup: {data.get('error')}")

        # Stream with ~0.5s chunks and two 1.5s silence gaps
        offset = 0
        base_chunk_seconds = 1.0  # keep chunks short but with enough context for stability
        min_chunk_len = max(2048, int(0.7 * bytes_per_sec))
        while offset < len(audio_data):
            target_chunk = bytes_per_sec * base_chunk_seconds * rng.uniform(0.8, 1.2)
            chunk_len = max(min_chunk_len, int(target_chunk))
            chunk = audio_data[offset : offset + chunk_len]
            if not chunk:
                break

            await websocket.send(chunk)
            chunk_duration = len(chunk) / bytes_per_sec
            playback_time += chunk_duration
            await asyncio.sleep(chunk_duration)

            if gap_markers and playback_time >= gap_markers[0]:
                logger.info("Injecting silence gap (1.5s)")
                await asyncio.sleep(1.5)
                gap_markers.pop(0)

            offset += chunk_len

        await websocket.send(json.dumps({"type": "stop_recording"}))

        while True:
            remaining = timeout - (time.monotonic() - start_time)
            if remaining <= 0:
                break

            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break

            if isinstance(message, bytes):
                continue

            payload = json.loads(message)
            if payload.get("status") == "processing_complete":
                break

            if payload.get("is_final") and payload.get("source"):
                par_key: str | int = payload.get("paragraph_index")
                if par_key is None:
                    par_key = payload.get("paragraph_id") or len(final_paragraphs)
                if par_key not in final_order:
                    final_order.append(par_key)
                final_paragraphs[par_key] = payload["source"]

    assert final_paragraphs
    final_sources = [final_paragraphs[k] for k in final_order]
    combined_text = " ".join(final_sources)
    logger.info(f"Combined Text (gaps): {combined_text}")

    _assert_transcript_quality(
        expected_text,
        combined_text,
        token_threshold=0.9,
        similarity_threshold=0.8,
        required_tokens=["boxing", "july"],
    )


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_websocket_streams_nato_with_timecoded_gaps(backend_server: str, auth_token: str):
    """Stream a timecoded NATO-word sample with randomized jitter and silence gaps."""
    if not NATO_WEBM.exists():
        pytest.skip("NATO sample audio not found (sample-nato.webm).")
    if not NATO_TIMECODES.exists():
        pytest.skip("NATO timecode file not found (sample-nato-timecodes.txt).")

    ws_url = backend_server.replace("http://", "ws://").replace("https://", "wss://") + "/ws/stream"
    timeout = float(os.getenv("E2E_WS_TIMEOUT", "120"))
    warmup_timeout = float(os.getenv("E2E_WS_WARMUP_TIMEOUT", "200"))

    entries = _load_timecoded_words(NATO_TIMECODES)
    expected_words = [word for _, _, word in entries]
    expected_text = " ".join(expected_words)

    audio_data = NATO_WEBM.read_bytes()
    bytes_per_sec = _estimate_bytes_per_second(audio_data, media_path=NATO_WEBM)

    rng = random.Random(int(os.getenv("E2E_WS_NATO_SEED", "2024")))
    gap_threshold = float(os.getenv("E2E_WS_NATO_GAP_THRESHOLD", "0.9"))
    gap_markers = _extract_silence_markers(entries, min_gap_s=gap_threshold)
    jitter_max = float(os.getenv("E2E_WS_NATO_JITTER_MAX", "0.25"))
    silence_min = float(os.getenv("E2E_WS_NATO_SILENCE_MIN", "1.0"))
    silence_max = float(os.getenv("E2E_WS_NATO_SILENCE_MAX", "1.8"))
    base_chunk_seconds = float(os.getenv("E2E_WS_NATO_CHUNK_SECONDS", "0.8"))

    final_paragraphs: dict[str | int, str] = {}
    final_order: list[str | int] = []
    start_time = time.monotonic()
    playback_time = 0.0

    logger.info(f"Connecting to {ws_url} (bps={bytes_per_sec:.1f})...")
    async with websockets.connect(
        ws_url,
        additional_headers={"Authorization": f"Bearer {auth_token}"},
        max_size=2**23,
        ping_interval=None,
        ping_timeout=None,
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "source_language": "en",
                    "target_language": "en",
                    "title": "E2E NATO Sample (Jitter)",
                }
            )
        )

        while True:
            init_msg = await asyncio.wait_for(websocket.recv(), timeout=warmup_timeout)
            if isinstance(init_msg, bytes):
                continue
            data = json.loads(init_msg)
            status = data.get("status")
            if status == "ready":
                break
            if status == "llm_ready":
                logger.info("Backend fully ready (llm_ready). Starting stream.")
                break
            if status == "error":
                pytest.fail(f"Backend reported error during warmup: {data.get('error')}")

        # Shared state for consumer
        processing_complete = asyncio.Event()

        async def consumer():
            while True:
                try:
                    # Use a short timeout to allow checking processing_complete
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    if processing_complete.is_set():
                        break
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

                if isinstance(message, bytes):
                    continue

                payload = json.loads(message)
                if payload.get("status") == "processing_complete":
                    processing_complete.set()
                    break

                if payload.get("is_final") and payload.get("source"):
                    par_key: str | int = payload.get("paragraph_index")
                    if par_key is None:
                        par_key = payload.get("paragraph_id") or len(final_paragraphs)
                    if par_key not in final_order:
                        final_order.append(par_key)
                    final_paragraphs[par_key] = payload["source"]

        consumer_task = asyncio.create_task(consumer())

        try:
            offset = 0
            min_chunk_len = max(2048, int(0.6 * bytes_per_sec))
            while offset < len(audio_data):
                target_chunk = bytes_per_sec * base_chunk_seconds * rng.uniform(0.75, 1.25)
                chunk_len = max(min_chunk_len, int(target_chunk))
                chunk = audio_data[offset : offset + chunk_len]
                if not chunk:
                    break

                await websocket.send(chunk)
                chunk_duration = len(chunk) / bytes_per_sec
                playback_time += chunk_duration

                # Emulate realtime send plus network jitter, with occasional longer silence gaps.
                await asyncio.sleep(chunk_duration + rng.uniform(0.0, jitter_max))
                if gap_markers and playback_time >= gap_markers[0]:
                    extra_pause = rng.uniform(silence_min, silence_max)
                    logger.info("Injecting silence gap (%.2fs)", extra_pause)
                    await asyncio.sleep(extra_pause)
                    gap_markers.pop(0)

                offset += chunk_len

            await websocket.send(json.dumps({"type": "stop_recording"}))

            # Wait for consumer to finish processing
            try:
                await asyncio.wait_for(processing_complete.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for processing completion.")
        
        finally:
            if not consumer_task.done():
                consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer_task

    assert final_paragraphs
    final_sources = [final_paragraphs[k] for k in final_order]
    combined_text = " ".join(final_sources)
    logger.info(f"Combined Text (nato): {combined_text}")

    _assert_transcript_quality(
        expected_text,
        combined_text,
        token_threshold=0.9,
        similarity_threshold=0.75,
    )
    _assert_transcript_covers_words(expected_words, combined_text)
