
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.api import router, run_warm_up
import asyncio

@pytest.mark.asyncio
async def test_run_warm_up_failure():
    mock_websocket = AsyncMock()
    mock_ready_event = asyncio.Event()

    with patch("app.api.warm_up_whisper", new_callable=AsyncMock) as mock_whisper:
        with patch("app.api.warm_up_llm", new_callable=AsyncMock) as mock_llm:
            # Simulate failure
            mock_whisper.return_value = False
            mock_llm.return_value = True

            mock_llm_ready = asyncio.Event()
            await run_warm_up(mock_websocket, "en", mock_ready_event, mock_llm_ready)

            # Check if error message was sent
            assert mock_websocket.send_json.called
            calls = mock_websocket.send_json.call_args_list
            
            # Should look for {"status": "error", ...}
            error_sent = False
            for call in calls:
                arg = call[0][0]
                if arg.get("status") == "error":
                    error_sent = True
                    break
            
            assert error_sent, "Error status should be sent to client"
            
            # ready_event should NOT be set (based on my implementation logic? Wait, let's check code)
            # In my implementation:
            # if not res_whisper or not res_llm:
            #    ... send error ...
            #    return
            # So ready_event.set() is skipped.
            assert not mock_ready_event.is_set(), "Ready event should not be set on failure"
            assert not mock_llm_ready.is_set(), "LLM Ready event should not be set on failure"

@pytest.mark.asyncio
async def test_run_warm_up_success():
    mock_websocket = AsyncMock()
    mock_ready_event = asyncio.Event()

    with patch("app.api.warm_up_whisper", new_callable=AsyncMock) as mock_whisper:
        with patch("app.api.warm_up_llm", new_callable=AsyncMock) as mock_llm:
            # Simulate success
            mock_whisper.return_value = True
            mock_llm.return_value = True

            mock_llm_ready = asyncio.Event()
            await run_warm_up(mock_websocket, "en", mock_ready_event, mock_llm_ready)

            # Check if ready message was sent
            assert mock_websocket.send_json.called
            calls = mock_websocket.send_json.call_args_list
            
            whisper_ready_sent = False
            llm_ready_sent = False
            for call in calls:
                arg = call[0][0]
                status = arg.get("status")
                if status == "whisper_ready":
                    whisper_ready_sent = True
                elif status == "llm_ready":
                    llm_ready_sent = True
            
            assert whisper_ready_sent, "Whisper ready status should be sent"
            assert llm_ready_sent, "LLM ready status should be sent"
            assert mock_ready_event.is_set(), "Ready event should be set on success"
            assert mock_llm_ready.is_set(), "LLM Ready event should be set on success"
