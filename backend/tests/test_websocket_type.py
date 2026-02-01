import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from app import api
from tests.utils import mock_warm_up

@pytest.mark.asyncio
async def test_websocket_stream_sends_paragraph_type():
    """
    Test that WebSocket stream includes 'type' in both preview and final messages.
    """
    mock_websocket = AsyncMock()
    # Initial config
    mock_websocket.receive_json.return_value = {
        "source_language": "en", 
        "target_language": "fr",
        "title": "Test Session"
    }
    
    # Mock user
    mock_user = MagicMock()
    mock_user.email = "test@example.com"
    
    # Mock DB Session
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.commit = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.flush = AsyncMock()
    mock_session_cls = MagicMock(return_value=mock_session)
    
    # Mock Conversation
    conversation_id = uuid4()
    mock_conversation = MagicMock()
    mock_conversation.id = conversation_id
    mock_conversation.title = "Test Session"
    mock_conversation.paragraphs = []
    
    # Mock Paragraph
    mock_paragraph = MagicMock()
    mock_paragraph.id = 123
    mock_paragraph.paragraph_index = 0
    mock_paragraph.source_text = ""
    mock_paragraph.translated_text = ""
    mock_paragraph.detected_language = "en"
    mock_paragraph.type = "active"
    
    # Mock Repositories
    mock_create_conversation = AsyncMock(return_value=mock_conversation)
    # add_paragraph returns our mock paragraph
    mock_add_paragraph = AsyncMock(return_value=mock_paragraph)
    
    # update_paragraph_content returns the updated paragraph
    # We want to check that when it is final, type is preserved or updated.
    # The API code passes `paragraph_type="stable"` to update_paragraph_content.
    # Let's mock the return of update to reflect that change.
    async def side_effect_update(session, pid, source_append, translation_append, source_override=None, translation_override=None, **kwargs):
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
        p.detected_language = kwargs.get("detected_language")
        p.type = kwargs.get("paragraph_type", "active")
        return p
        
    mock_update_paragraph_content = AsyncMock(side_effect=side_effect_update)

    # Mock Whisper Stream
    # Yield one non-final chunk, then one final chunk
    async def mock_stream(*args, **kwargs):
        yield {"text": "Hello", "is_final": False}
        yield {"text": "Hello world", "is_final": True}
        
    # Mock Translate
    mock_translate = AsyncMock(return_value="Bonjour le monde")

    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.get_current_user_websocket", AsyncMock(return_value=mock_user)), \
         patch("app.api.get_or_create_user", AsyncMock()), \
         patch("app.api.create_conversation", mock_create_conversation), \
         patch("app.api.add_paragraph", mock_add_paragraph), \
         patch("app.api.update_paragraph_content", mock_update_paragraph_content), \
         patch("app.api.stream_transcription", side_effect=mock_stream), \
         patch("app.api.translate_text", mock_translate), \
         patch("app.api.generate_title", AsyncMock()), \
         patch("app.api.run_warm_up", side_effect=mock_warm_up), \
         patch("app.api._gather_audio", MagicMock()):
         
        await api.websocket_stream(mock_websocket)
        
        # Verify calls to send_json
        calls = [c[0][0] for c in mock_websocket.send_json.call_args_list]
        
        # Filter for message updates (ignoring initial title/ready/etc if any)
        # We expect a preview message and a final message
        
        # Check Preview Message
        previews = [c for c in calls if "is_final" in c and c["is_final"] is False]
        assert len(previews) > 0
        preview = previews[0]
        assert "type" in preview
        assert preview["type"] == "active"
        
        # Check Final Message
        finals = [c for c in calls if "is_final" in c and c["is_final"] is True]
        assert len(finals) > 0
        final = finals[0]
        assert "type" in final
        assert final["type"] == "active"
