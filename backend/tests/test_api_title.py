import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Mocking the imports in app.api before importing it if needed, 
# or using patch.context.

from app import api
from tests.utils import DummyLimiter, mock_warm_up

@pytest.mark.asyncio
async def test_websocket_stream_title_generation():
    """
    Test that title generation is triggered when total words >= 50,
    counting words from the entire conversation history.
    """
    mock_websocket = AsyncMock()
    mock_websocket.receive_json.return_value = {
        "source_language": "en", 
        "target_language": "es",
        "title": None
    }
    
    # Mock user
    mock_user = MagicMock()
    mock_user.email = "test@example.com"
    
    # Mock DB Session
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.flush = AsyncMock()
    
    # Context manager for SessionLocal
    mock_session_cls = MagicMock(return_value=mock_session)
    
    # Mock Conversation
    conversation_id = uuid4()
    mock_conversation = MagicMock()
    mock_conversation.id = conversation_id
    mock_conversation.title = "New Session"
    mock_conversation.is_title_manual = False
    mock_conversation.paragraphs = []
    
    # Setup session.execute to return the conversation
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_conversation
    mock_session.execute.return_value = mock_result
    
    # Mock Repositories
    mock_get_or_create_user = AsyncMock(return_value=MagicMock())
    mock_create_conversation = AsyncMock(return_value=mock_conversation)
    mock_add_paragraph = AsyncMock(return_value=MagicMock())
    
    # Mock update_paragraph_content to return a paragraph with accumulated text
    async def side_effect_update(session, pid, source_append=None, translation_append=None, source_override=None, translation_override=None, **kwargs):
        # We simulate that the paragraph now has this translation
        p = MagicMock()
        p.id = pid
        if source_override is not None:
            p.source_text = source_override
        else:
            p.source_text = source_append
        if translation_override is not None:
            p.translated_text = translation_override
        else:
            p.translated_text = translation_append
        p.type = "stable"
        
        # Also update the conversation paragraphs list to reflect "persistence"
        # We replace the paragraphs list with this one paragraph for simplicity,
        # ensuring the "select" sees proper text.
        mock_conversation.paragraphs = [p]
        return p
        
    mock_update_paragraph_content = AsyncMock(side_effect=side_effect_update)

    # Mock Whisper Stream
    # We want to yield enough text to trigger title generation (50 words).
    # "word " * 50
    long_text = "word " * 60
    
    async def mock_stream(*args, **kwargs):
        # Yield one chunk that is final
        yield {"text": long_text, "is_final": True}
        
    # Mock Generate Title
    mock_generate_title = AsyncMock(return_value="Generated Title")

    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.get_current_user_websocket", AsyncMock(return_value=mock_user)), \
         patch("app.api.get_or_create_user", mock_get_or_create_user), \
         patch("app.api.create_conversation", mock_create_conversation), \
         patch("app.api.add_paragraph", mock_add_paragraph), \
         patch("app.api.update_paragraph_content", mock_update_paragraph_content), \
         patch("app.api.stream_transcription", side_effect=mock_stream), \
         patch("app.api.translate_text", AsyncMock(return_value=long_text)), \
          patch("app.api.generate_title", mock_generate_title), \
          patch("app.api.run_warm_up", side_effect=mock_warm_up), \
          patch("app.api.translation_limiter", DummyLimiter()), \
          patch("app.api._gather_audio", MagicMock()):
         
        await api.websocket_stream(mock_websocket)
        
        assert mock_generate_title.called
        assert mock_conversation.title == "Generated Title"

@pytest.mark.asyncio
async def test_websocket_stream_title_generation_with_history():
    """
    Test that title generation is triggered when history + new text >= 50 words.
    """
    mock_websocket = AsyncMock()
    mock_websocket.receive_json.return_value = {
        "source_language": "en", 
        "target_language": "es",
        "title": None
    }
    
    mock_user = MagicMock()
    
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.flush = AsyncMock()
    
    mock_session_cls = MagicMock(return_value=mock_session)
    
    conversation_id = uuid4()
    mock_conversation = MagicMock()
    mock_conversation.id = conversation_id
    mock_conversation.title = "New Session"
    mock_conversation.is_title_manual = False
    
    # Pre-existing history: 40 words
    p1 = MagicMock()
    p1.translated_text = "word " * 40
    p1.type = "stable"
    mock_conversation.paragraphs = [p1]
    
    # Setup session.execute
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_conversation
    mock_session.execute.return_value = mock_result
    
    async def side_effect_update(session, pid, source_append=None, translation_append=None, source_override=None, translation_override=None, **kwargs):
        # New paragraph
        p2 = MagicMock()
        p2.id = pid
        p2.source_text = source_override or source_append
        if translation_override is not None:
            p2.translated_text = translation_override
        else:
            p2.translated_text = translation_append
        p2.type = "stable"
        # Update history
        mock_conversation.paragraphs = [p1, p2]
        return p2
        
    mock_update_paragraph_content = AsyncMock(side_effect=side_effect_update)
    
    # Stream: 20 words (Total 60)
    async def mock_stream(*args, **kwargs):
        yield {"text": "word " * 20, "is_final": True}
        
    mock_generate_title = AsyncMock(return_value="History Title")

    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.get_current_user_websocket", AsyncMock(return_value=mock_user)), \
         patch("app.api.get_or_create_user", AsyncMock()), \
         patch("app.api.create_conversation", AsyncMock(return_value=mock_conversation)), \
         patch("app.api.add_paragraph", AsyncMock()), \
         patch("app.api.update_paragraph_content", mock_update_paragraph_content), \
         patch("app.api.stream_transcription", side_effect=mock_stream), \
         patch("app.api.translate_text", AsyncMock(return_value="word " * 20)), \
          patch("app.api.generate_title", mock_generate_title), \
          patch("app.api.run_warm_up", side_effect=mock_warm_up), \
          patch("app.api.translation_limiter", DummyLimiter()), \
          patch("app.api._gather_audio", MagicMock()):
         
        await api.websocket_stream(mock_websocket)
        
        assert mock_generate_title.called
        assert mock_conversation.title == "History Title"
