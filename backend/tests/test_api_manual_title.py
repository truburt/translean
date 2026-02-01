
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from app import api

@pytest.mark.asyncio
async def test_update_conversation_title_endpoint():
    conn_id = uuid4()
    mock_payload = MagicMock()
    mock_payload.title = "Manual Title"
    
    mock_session = AsyncMock()
    mock_user = MagicMock()
    
    mock_conversation = MagicMock()
    mock_conversation.id = conn_id
    mock_conversation.title = "Old Title"
    mock_conversation.is_title_manual = False
    
    with patch("app.api.get_or_create_user", AsyncMock(return_value=MagicMock())), \
         patch("app.api.get_conversation", AsyncMock(return_value=mock_conversation)):
         
        updated = await api.update_conversation_title_endpoint(conn_id, mock_payload, mock_session, mock_user)
        
        assert updated.title == "Manual Title"
        assert updated.is_title_manual is True
        mock_session.commit.assert_awaited_once()

@pytest.mark.asyncio
async def test_title_generation_skipped_if_manual():
    """
    Test that title generation is skipped when is_title_manual is True.
    """
    mock_websocket = AsyncMock()
    mock_websocket.receive_json.return_value = {
        "source_language": "en", 
        "target_language": "es",
        "title": None 
    }
    
    mock_user = MagicMock()
    mock_session = AsyncMock()
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    conversation_id = uuid4()
    mock_conversation = MagicMock()
    mock_conversation.id = conversation_id
    mock_conversation.title = "Manual Title"
    mock_conversation.is_title_manual = True # FLAG IS TRUE
    
    # Existing history sufficient to trigger title (>50 words)
    p1 = MagicMock()
    p1.translated_text = "word " * 60
    p1.type = "stable"
    mock_conversation.paragraphs = [p1]
    
    mock_generate_title = AsyncMock(return_value="Generated Title")

    async def mock_stream(*args, **kwargs):
        # streaming one chunk to trigger finalization logic
        yield {"text": "more words", "is_final": True}

    async def mock_warm_up(ws, lang, whisper_ready, llm_ready):
        whisper_ready.set()
        llm_ready.set()

    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.get_current_user_websocket", AsyncMock(return_value=mock_user)), \
         patch("app.api.get_or_create_user", AsyncMock()), \
         patch("app.api.create_conversation", AsyncMock(return_value=mock_conversation)), \
         patch("app.api.add_paragraph", AsyncMock()), \
         patch("app.api.update_paragraph_content", AsyncMock(return_value=MagicMock())), \
         patch("app.api.stream_transcription", side_effect=mock_stream), \
         patch("app.api.translate_text", AsyncMock(return_value="more words")), \
         patch("app.api.generate_title", mock_generate_title), \
         patch("app.api.run_warm_up", side_effect=mock_warm_up), \
         patch("app.api._gather_audio", AsyncMock()):
         
        await api.websocket_stream(mock_websocket)
        
        # Should NOT call generate_title because is_title_manual is True
        mock_generate_title.assert_not_called()
