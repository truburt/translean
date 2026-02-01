import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from app import api
from tests.utils import mock_warm_up

@pytest.mark.asyncio
async def test_resume_updates_target_language():
    """
    Test that resuming a conversation with a different target language in init
    updates the conversation in the database.
    """
    mock_websocket = AsyncMock()
    # Initial config with NEW target language
    mock_websocket.receive_json.return_value = {
        "source_language": "en", 
        "target_language": "es", # Changed from 'fr'
        "title": "Test Session"
    }
    
    mock_user = MagicMock()
    mock_user.email = "test@example.com"
    
    mock_session = AsyncMock()
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    conversation_id = uuid4()
    mock_conversation = MagicMock()
    mock_conversation.id = conversation_id
    mock_conversation.target_language = "fr" # Original language
    mock_conversation.paragraphs = []
    
    # Mock Repositories
    mock_get_conversation = AsyncMock(return_value=mock_conversation)
    mock_create_conversation = AsyncMock(return_value=mock_conversation)
    mock_add_paragraph = AsyncMock(return_value=MagicMock(type="active", paragraph_index=0, source_text="", translated_text=""))
    
    # Mock Whisper Stream to exit immediately
    async def mock_stream(*args, **kwargs):
        if False: yield # empty generator

    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.get_current_user_websocket", AsyncMock(return_value=mock_user)), \
         patch("app.api.get_or_create_user", AsyncMock()), \
         patch("app.api.get_conversation", mock_get_conversation), \
         patch("app.api.create_conversation", mock_create_conversation), \
         patch("app.api.add_paragraph", mock_add_paragraph), \
         patch("app.api.update_paragraph_content", AsyncMock()), \
         patch("app.api.stream_transcription", side_effect=mock_stream), \
         patch("app.api.translate_text", AsyncMock()), \
         patch("app.api.generate_title", AsyncMock()), \
         patch("app.api.run_warm_up", side_effect=mock_warm_up), \
         patch("app.api.delete_conversation", AsyncMock(return_value=True)), \
         patch("app.api._gather_audio", MagicMock()):
         
        # helper to inject ID
        await api.websocket_stream(mock_websocket, conversation_id=str(conversation_id))
        
        # Verify update
        assert mock_conversation.target_language == "Spanish"
        mock_session.commit.assert_called()


@pytest.mark.asyncio
async def test_dynamic_update_target_language():
    """
    Test that receiving a target_language update via WebSocket updates the DB.
    """
    mock_websocket = AsyncMock()
    mock_websocket.receive_json.return_value = {
        "source_language": "en", 
        "target_language": "en",
        "title": "Test Session"
    }
    
    mock_user = MagicMock()
    mock_session = AsyncMock()
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    mock_conversation = MagicMock()
    mock_conversation.id = uuid4()
    mock_conversation.target_language = "en"
    mock_conversation.paragraphs = []

    mock_create_conversation = AsyncMock(return_value=mock_conversation)
    mock_add_paragraph = AsyncMock(return_value=MagicMock(type="active", paragraph_index=0))

    # Mock Whisper Stream to yield a config update, then exit
    async def mock_stream(*args, **kwargs):
        # The gather_audio yields the config dict directly if it sees JSON
        # stream_transcription yields it if it sees it?
        # Ahem, stream_transcription yields whatever _gather_audio yields if it doesn't recognize it as audio? 
        # Actually _gather_audio yields "bytes" or parsed json.
        # stream_transcription takes audio_iter.
        # Looking at whisper_client.py (not shown fully but api.py calls it):
        # The api.py logic for chunks:
        # yield parser: if chunk is dict...
        # Wait. In api.py, _gather_audio yields bytes OR dicts (config).
        # stream_transcription in api.py (imported from whisper_client) probably consumes bytes.
        # If _gather_audio yields a dict, stream_transcription must pass it through or api.py handles it?
        # Let's check api.py around line 583:
        # `async for chunk in transcript_iter:`
        # So `stream_transcription` must be yielding the config dict too.
        # We will assume stream_transcription simply yields the config dict if it receives it or if we mock it to yield it.
        yield {"target_language": "de"} 
        
    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.get_current_user_websocket", AsyncMock(return_value=mock_user)), \
         patch("app.api.get_or_create_user", AsyncMock()), \
         patch("app.api.create_conversation", mock_create_conversation), \
         patch("app.api.add_paragraph", mock_add_paragraph), \
         patch("app.api.stream_transcription", side_effect=mock_stream), \
         patch("app.api.run_warm_up", side_effect=mock_warm_up), \
         patch("app.api.delete_conversation", AsyncMock(return_value=True)), \
         patch("app.api._gather_audio", MagicMock()): # _gather_audio mock is irrelevant if stream_transcription is mocked
         
        await api.websocket_stream(mock_websocket, conversation_id=None)
        
        # Verify update
        assert mock_conversation.target_language == "German"
        mock_session.commit.assert_called()
