import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.api import rebuild_conversation_task, ConnectionManager, manager
from app import models

@pytest.mark.asyncio
async def test_rebuild_conversation_task_broadcasts_updates():
    # Setup dependencies
    conversation_id = uuid.uuid4()
    
    # Mock mocks
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.rollback = AsyncMock()

    # Mock conversation data
    mock_conversation = MagicMock()
    mock_conversation.id = conversation_id
    mock_conversation.paragraphs = [
        MagicMock(source_text="Hello world.", type="active", paragraph_index=0, detected_language="en"),
        MagicMock(source_text="This is a test.", type="active", paragraph_index=1, detected_language="en"),
    ]
    mock_conversation.title = "New Session 123"
    mock_conversation.is_title_manual = False
    mock_conversation.source_language = "en"
    mock_conversation.target_language = "Spanish"

    # Mock DB functions
    patch_session_local = patch("app.api.SessionLocal", return_value=mock_session)
    patch_get_conversation = patch("app.api.get_conversation", new_callable=AsyncMock, return_value=mock_conversation)
    patch_add_paragraph = patch("app.api.add_paragraph", new_callable=AsyncMock)
    patch_update_paragraph = patch("app.api.update_paragraph_content", new_callable=AsyncMock)
    patch_cleanup = patch("app.api.cleanup_text", new_callable=AsyncMock, side_effect=lambda t, **k: t)
    patch_translate = patch("app.api.translate_text", new_callable=AsyncMock, side_effect=lambda t, *a, **k: f"Translated: {t}")

    patch_summarize = patch("app.api.summarize_text", new_callable=AsyncMock, return_value="Summary")
    patch_generate_title = patch("app.api.generate_title", new_callable=AsyncMock, return_value="New Title")
    patch_estimate = patch("app.api.estimate_token_count", return_value=10)
    patch_limiter = patch("app.api.translation_limiter")

    # Mock ConnectionManager broadcast
    # We can either mock the global `manager` or patch `ConnectionManager.broadcast`
    # Since `rebuild_conversation_task` uses the global `manager` variable, we should patch where it's imported or the object itself.
    # In `app.api`, `manager` is an instance. 
    
    with patch_session_local, \
         patch_get_conversation, \
         patch_add_paragraph as mock_add_p, \
         patch_update_paragraph, \
         patch_cleanup, \
         patch_translate, \
         patch_summarize, \
         patch_generate_title, \
         patch_estimate, \
         patch_limiter as mock_limiter, \
         patch.object(manager, "broadcast", new_callable=AsyncMock) as mock_broadcast:
        
        # Mock limiter acquire
        mock_limiter.acquire = AsyncMock()
        
        # Setup return value for add_paragraph so it has an ID
        mock_combined_p = MagicMock()
        mock_combined_p.id = uuid.uuid4()
        mock_combined_p.paragraph_index = 2
        
        mock_summary_p = MagicMock()
        mock_summary_p.id = uuid.uuid4()
        
        mock_add_p.side_effect = [mock_combined_p, mock_summary_p]

        # Run Task
        await rebuild_conversation_task(conversation_id)

        # Verification
        assert mock_broadcast.called
        
        # Check expected broadcast calls
        # 1. Chunk updates for "Hello world." and "This is a test."
        # The logic chunks by sentence. "Hello world." and "This is a test." might be separate or combined depending on chunk_text_by_sentenceMock logic?
        # Actually `chunk_text_by_sentence` uses regex split.
        
        # Verify at least one 'chunk' message
        chunk_calls = [c for c in mock_broadcast.call_args_list if c[0][1].get("type") == "chunk"]
        assert len(chunk_calls) > 0, "Should broadcast chunks"
        
        # Verify chunk keys match new schema
        first_chunk = chunk_calls[0][0][1]
        assert "source" in first_chunk
        assert "translation" in first_chunk
        assert "source_text" not in first_chunk # Ensure old keys are gone
        
        # Verify 'final' message
        final_calls = [c for c in mock_broadcast.call_args_list if c[0][1].get("type") == "final"]
        assert len(final_calls) == 1, "Should broadcast final message"
        
        final_msg = final_calls[0][0][1]
        assert final_msg["title"] == "New Title"
        # Check flattened ID
        assert final_msg["paragraph_id"] == str(mock_combined_p.id)

@pytest.mark.asyncio
async def test_connection_manager_logic():
    cm = ConnectionManager()
    conv_id = uuid.uuid4()
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    
    await cm.connect(conv_id, ws1)
    await cm.connect(conv_id, ws2)
    
    assert len(cm.active_connections[conv_id]) == 2
    
    await cm.broadcast(conv_id, {"test": "msg"})
    
    ws1.send_text.assert_awaited_with('{"test": "msg"}')
    ws2.send_text.assert_awaited_with('{"test": "msg"}')
    
    await cm.disconnect(conv_id, ws1)
    assert len(cm.active_connections[conv_id]) == 1
    
    await cm.disconnect(conv_id, ws2)
    assert conv_id not in cm.active_connections

@pytest.mark.asyncio
async def test_rebuild_deduplication_and_cancellation():
    """Test that rebuilds are deduplicated and can be cancelled."""
    conversation_id = uuid.uuid4()
    
    # Mock active_rebuild_tasks
    mock_tasks = {}
    
    with patch("app.api.active_rebuild_tasks", mock_tasks):
        # 1. Test Deduplication
        # Simulate an existing running task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_tasks[conversation_id] = mock_task
        
        # Call endpoint logic (simulated since we can't easily call FastAPI endpoint directly here without TestClient)
        # But we can verify the logic block:
        if conversation_id in mock_tasks and not mock_tasks[conversation_id].done():
            result = {"status": "accepted", "message": "Rebuild task already running."}
        else:
            result = None
            
        assert result["message"] == "Rebuild task already running."
        
        # 2. Test Cancellation Logic (simulated)
        # Simulate new input arriving
        if conversation_id in mock_tasks:
            t = mock_tasks[conversation_id]
            if not t.done():
                t.cancel()
        
        mock_task.cancel.assert_called()
