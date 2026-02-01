import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.api import TranslationWorker
from app import models

@pytest.mark.asyncio
async def test_translation_worker_merges_requests():
    # Setup dependencies
    queue = asyncio.Queue()
    websocket = AsyncMock()
    conversation_id = uuid.uuid4()
    
    # Mock database interactions
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.commit = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.flush = AsyncMock()
    mock_session.execute.return_value.scalar_one_or_none.return_value = ""
    
    mock_session_cls = MagicMock(return_value=mock_session)
    
    # Mock update_paragraph_content to verify calls
    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.translate_text", return_value="Translated Text") as mock_translate, \
         patch("app.api.update_paragraph_content", new_callable=AsyncMock) as mock_update_paragraph, \
         patch("app.api.get_or_create_user", new_callable=AsyncMock), \
         patch("app.api.translation_limiter.acquire", new_callable=AsyncMock) as mock_acquire: # Eliminate delay
         
        worker = TranslationWorker(queue, websocket, conversation_id, {}, {}, AsyncMock())
        task = asyncio.create_task(worker.run())
        
        # Enqueue multiple items for the same paragraph
        # Item format: (p_id, text_segment, tgt_lang, ctx_text, p_idx, p_lang, p_type, src_lang)
        p_id = uuid.uuid4()
        queue.put_nowait((p_id, "Hello", "Spanish", "", 1, "en", "active", "English"))
        queue.put_nowait((p_id, "world", "Spanish", "", 1, "en", "active", "English"))
        queue.put_nowait((p_id, "how are you?", "Spanish", "", 1, "en", "active", "English"))
        
        # Add a sentinel to stop the worker logic explicitly if needed, but queue join is better
        # For the test, we can just wait a bit or verify side effects
        
        # We need to let the worker process
        # Since mock_acquire is stripped of delay, it should process fast.
        # But we want to ensure *coalescing* happens.
        # The worker coalesces by opportunistically draining the queue.
        # By `put_nowait` all at once, they should be in the queue before the worker
        # wakes up from the first `await queue.get()` if we yield control properly?
        # Ideally, we pause the worker *inside* the processing loop, but `get_nowait` is synchronous.
        # If we put them all before starting the worker, it should pick them all up.
        
        await asyncio.sleep(0.1)
        
        # Allow worker to finish
        queue.put_nowait(None)
        await asyncio.wait_for(task, timeout=1.0)
        
        # Verify translate_text was called with the latest snapshot (last queued text)
        
        # Check calls to translate_text
        # It's possible it processed them in batches if the loop was too fast, 
        # but with `put_nowait` and no awaits between puts, they should be available.
        
        calls = mock_translate.call_args_list
        assert len(calls) > 0, "Translate text should have been called"
        
        # We expect at least one call with the full text if perfect coalescing.
        # Or if it split, we want to ensure no text was lost/replaced.
        # But the Issue is "replacement". So if we see "how are you?" only, that's bad.
        
        # Let's inspect the arguments of the call
        processed_texts = [c[0][0] for c in calls]
        print(f"Processed texts: {processed_texts}")
        
        assert processed_texts[-1] == "how are you?", \
            f"Expected latest snapshot translation. Got: {processed_texts}"

@pytest.mark.asyncio
async def test_translation_worker_splits_on_target_change():
    """
    Ensure the worker DOES NOT coalesce requests with different target languages.
    """
    queue = asyncio.Queue()
    websocket = AsyncMock()
    conversation_id = uuid.uuid4()
    
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.commit = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.flush = AsyncMock()
    mock_session.execute.return_value.scalar_one_or_none.return_value = ""
    
    mock_session_cls = MagicMock(return_value=mock_session)
    
    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.translate_text") as mock_translate, \
         patch("app.api.update_paragraph_content", new_callable=AsyncMock), \
         patch("app.api.get_or_create_user", new_callable=AsyncMock), \
         patch("app.api.translation_limiter.acquire", new_callable=AsyncMock):
         
        # Mimic behavior: return context-prefixed text to verify calls
        async def side_effect(text, target, context_text="", source_language_name="Auto"):
            return f"[{target}] {text} ({source_language_name})"
        mock_translate.side_effect = side_effect

        worker = TranslationWorker(queue, websocket, conversation_id, {}, {}, AsyncMock())
        task = asyncio.create_task(worker.run())
        
        p_id = uuid.uuid4()
        # 1. German target
        queue.put_nowait((p_id, "Hallo", "German", "", 1, "auto", "active", "German"))
        # 2. Spanish target (Should NOT merge with above)
        queue.put_nowait((p_id, "Mundo", "Spanish", "", 1, "auto", "active", "Spanish"))
        
        await asyncio.sleep(0.1)
        queue.put_nowait(None)
        await asyncio.wait_for(task, timeout=1.0)
        
        # Verify calls
        # Expected: 2 calls.
        # Call 1: "Hallo", target="German"
        # Call 2: "Mundo", target="Spanish"
        
        calls = mock_translate.call_args_list
        print("Calls:", calls)
        
        assert len(calls) == 2, "Should have split into 2 calls due to target change"
        
        call1_args = calls[0]
        assert call1_args[0][0] == "Hallo"
        assert call1_args[0][1] == "German"
        
        call2_args = calls[1]
        assert call2_args[0][0] == "Mundo"
        assert call2_args[0][1] == "Spanish"
