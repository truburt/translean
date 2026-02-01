
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from app import api
from tests.utils import DummyLimiter

@pytest.mark.asyncio
async def test_two_way_translation_logic():
    """
    Test 2-way translation logic:
    1. Chunk 1 (German) -> Translated to Target (Spanish). History updated.
    2. Chunk 2 (Spanish=Target) -> Should revert to prevalent language (German).
    """
    mock_websocket = AsyncMock()
    mock_websocket.receive_json.return_value = {
        "source_language": "auto", 
        "target_language": "Spanish",
        "title": None
    }
    
    # Mock User
    mock_user = MagicMock()
    mock_user.email = "test@example.com"
    
    # Mock DB
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
    mock_conversation.target_language = "Spanish"
    mock_conversation.source_language = "auto"
    mock_conversation.paragraphs = []
    
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_conversation
    mock_session.execute.return_value = mock_result
    
    # Mock Repositories
    mock_create_conversation = AsyncMock(return_value=mock_conversation)
    
    # Mock update_paragraph_content
    # We need to simulate that the first paragraph ADDS 'de' to the detected detected_language
    async def side_effect_update(session, pid, source_append, translation_append, date=None, detected_language=None, source_override=None, translation_override=None, **kwargs):
        # We need to persist detected_language to conversation paragraphs to simulate history
        current_p = next((p for p in mock_conversation.paragraphs if p.id == pid), None)
        if not current_p:
            current_p = MagicMock()
            current_p.id = pid
            current_p.paragraph_index = 0 
            mock_conversation.paragraphs.append(current_p)
        
        if source_override is not None:
            current_p.source_text = source_override
        else:
            current_p.source_text = source_append
        if translation_override is not None:
            current_p.translated_text = translation_override
        else:
            current_p.translated_text = translation_append
        if detected_language:
            current_p.detected_language = detected_language
        return current_p

    mock_update_paragraph_content = AsyncMock(side_effect=side_effect_update)
    mock_add_paragraph = AsyncMock(return_value=MagicMock(id=uuid4(), paragraph_index=0, detected_language=None))

    # Mock TranslationWorker to capture queue items
    # We want to see what is put into the queue
    captured_queue = None
    
    class MockTranslationWorker:
        def __init__(self, queue, *args, **kwargs):
            nonlocal captured_queue
            captured_queue = queue
        
        async def run(self):
            # Just drain queue to avoid blocking join()
            while True:
                item = await captured_queue.get()
                captured_queue.task_done()
                if item is None:
                    break

    # Mock Stream
    async def mock_stream(*args, **kwargs):
        # Chunk 1: German input
        yield {"text": "Hallo Welt", "is_final": True, "language": "de"}
        
        # Short pause to let loop process and update DB "history"
        await asyncio.sleep(0.01)
        
        # Chunk 2: Spanish input (matches target)
        # Should trigger 2-way logic
        yield {"text": "Hola Mundo", "is_final": True, "language": "es"}
        
    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.get_current_user_websocket", AsyncMock(return_value=mock_user)), \
         patch("app.api.get_or_create_user", AsyncMock()), \
         patch("app.api.create_conversation", mock_create_conversation), \
         patch("app.api.add_paragraph", mock_add_paragraph), \
         patch("app.api.update_paragraph_content", mock_update_paragraph_content), \
         patch("app.api.stream_transcription", side_effect=mock_stream), \
         patch("app.api.translate_text", AsyncMock(return_value="Valid Translation")), \
         patch("app.api.TranslationWorker", MockTranslationWorker), \
          patch("app.api.run_warm_up", AsyncMock()), \
          patch("app.api.translation_limiter", DummyLimiter()), \
          patch("app.api._gather_audio", MagicMock()):
         
        # Run
        await api.websocket_stream(mock_websocket)
        
@pytest.mark.asyncio
async def test_two_way_translation_verification():
    mock_websocket = AsyncMock()
    mock_websocket.receive_json.return_value = {
        "source_language": "auto", 
        "target_language": "Spanish",
        "title": None
    }
    
    mock_user = MagicMock()
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.commit = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.flush = AsyncMock()
    
    mock_session_cls = MagicMock(return_value=mock_session)
    
    mock_conversation = MagicMock()
    mock_conversation.id = uuid4()
    mock_conversation.target_language = "Spanish"
    mock_conversation.source_language = "auto"
    mock_conversation.paragraphs = [] # Mutable list
    
    # We need to ensure `get_conversation` returns our convo so logic can reload it
    mock_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
    
    # Paragraph management
    # We need a robust `update_paragraph_content` that updates our `mock_conversation.paragraphs`
    # so the "prevalence" check works.
    
    async def side_effect_update(session, pid, source_append, translation_append, detected_language=None, source_override=None, translation_override=None, **kwargs):
        # Find or create paragraph
        p = next((x for x in mock_conversation.paragraphs if x.id == pid), None)
        if not p:
            p = MagicMock()
            p.id = pid
            p.paragraph_index = 0
            mock_conversation.paragraphs.append(p)
        
        if detected_language:
            p.detected_language = detected_language
        if source_override is not None:
            p.source_text = source_override
        elif source_append:
            p.source_text = source_append
        if translation_override is not None:
            p.translated_text = translation_override
        elif translation_append:
            p.translated_text = translation_append
        return p

    mock_update = AsyncMock(side_effect=side_effect_update)
    mock_add = AsyncMock(return_value=MagicMock(id=uuid4(), paragraph_index=0, detected_language=None))
    mock_get_or_create_user = AsyncMock()

    # Capture the queue exposed to worker
    captured_queue = None
    mock_worker_cls = MagicMock()
    
    def worker_init(queue, *args, **kwargs):
        nonlocal captured_queue
        captured_queue = queue
        mock_worker = AsyncMock()
        # Mock run to just drain queue so join() works
        async def run():
            while True:
                item = await queue.get()
                queue.task_done()
                if item is None: break
                # Store item for verification
                mock_worker.processed_items.append(item)
        mock_worker.run = run
        mock_worker.processed_items = []
        return mock_worker
        
    mock_worker_cls.side_effect = worker_init

    async def mock_stream(*args, **kwargs):
        # 1. German Input
        yield {"text": "German", "is_final": True, "language": "de"}
        await asyncio.sleep(0.01) # Let it process
        
        # 2. Spanish Input (Target)
        yield {"text": "Spanish", "is_final": True, "language": "es"}
        await asyncio.sleep(0.01)
        
    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.get_current_user_websocket", AsyncMock(return_value=mock_user)), \
         patch("app.api.get_or_create_user", mock_get_or_create_user), \
         patch("app.api.create_conversation", AsyncMock(return_value=mock_conversation)), \
         patch("app.api.get_conversation", AsyncMock(return_value=mock_conversation)),\
         patch("app.api.add_paragraph", mock_add), \
         patch("app.api.update_paragraph_content", mock_update), \
         patch("app.api.stream_transcription", side_effect=mock_stream), \
         patch("app.api.translate_text", AsyncMock(return_value="Trans")), \
          patch("app.api.TranslationWorker", mock_worker_cls), \
          patch("app.api.run_warm_up", AsyncMock()), \
          patch("app.api.translation_limiter", DummyLimiter()), \
          patch("app.api._gather_audio", MagicMock()):
         
        await api.websocket_stream(mock_websocket)
        
        # Verify
        assert captured_queue is not None
        # We need the worker instance that was created
        worker_instance = mock_worker_cls.return_value
        pass

    # We need to access the `processed_items` from the worker we created.
    # Since we are inside the test, let's just use a list in outer scope.
    processed = []
    
    mock_worker_cls = MagicMock()
    def worker_init_2(queue, *args, **kwargs):
        async def run():
            while True:
                item = await queue.get()
                queue.task_done()
                if item is None: break
                processed.append(item)
        worker = AsyncMock()
        worker.run = run
        return worker
    mock_worker_cls.side_effect = worker_init_2
    
    # RERUN with correct capture
    with patch("app.api.SessionLocal", mock_session_cls), \
         patch("app.api.get_current_user_websocket", AsyncMock(return_value=mock_user)), \
         patch("app.api.get_or_create_user", mock_get_or_create_user), \
         patch("app.api.create_conversation", AsyncMock(return_value=mock_conversation)), \
         patch("app.api.get_conversation", AsyncMock(return_value=mock_conversation)),\
         patch("app.api.add_paragraph", mock_add), \
         patch("app.api.update_paragraph_content", mock_update), \
         patch("app.api.stream_transcription", side_effect=mock_stream), \
         patch("app.api.translate_text", AsyncMock(return_value="Trans")), \
          patch("app.api.TranslationWorker", mock_worker_cls), \
          patch("app.api.run_warm_up", AsyncMock()), \
          patch("app.api.translation_limiter", DummyLimiter()), \
          patch("app.api._gather_audio", MagicMock()):
         
        await api.websocket_stream(mock_websocket)

    # ASSERTIONS
    print(f"Processed items: {processed}")
    assert len(processed) >= 2
    
    # Item 1: German -> Spanish
    it1 = processed[0]
    # (p_id, text, target, ctx, idx, lang, type)
    assert it1[1] == "German"
    assert it1[2] == "Spanish"
    assert "de" in it1[5]

    # Item 2: Spanish -> Prevalent (German)
    # If logic NOT implemented, it defaults to original behavior:
    # `effective_lang` (es) == `target_code` (es) -> Skip queue!
    # "Optimization: Immediate assignment" block updates DB but DOES NOT queue translation.
    
    # So if logic is missing, `processed` will only have 1 item! (or 2 if fallback logic puts it in queue?)
    # Existing logic: `if effective_lang != "auto" and effective_lang == target_code:` -> `update_paragraph_content` -> NO QUEUE.
    
    # So we expect FAILURE: processed[1] will be missing.
    assert len(processed) == 2, "Expected 2 translation tasks, got " + str(len(processed))
    
    it2 = processed[1]
    assert it2[1] == "Spanish"
    assert it2[2] == "German" # Prevalent language
