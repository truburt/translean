
import pytest
import logging
import os
import sys
from pathlib import Path

# Ensure backend/app is importable when running from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm_client import translate_text, generate_title, summarize_text, warm_up

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_llm_warmup():
    """Verify that the LLM warm-up completes successfully."""
    logger.info("Testing LLM warm-up...")
    success = await warm_up()
    assert success, "LLM warm-up failed"

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_llm_translation_en_to_es():
    """Test English to Spanish translation."""
    source_text = "Hello, how are you today?"
    target_lang = "Spanish"
    
    logger.info(f"Testing translation: '{source_text}' -> {target_lang}")
    translation = await translate_text(
        text=source_text,
        target_language_name=target_lang,
        source_language_name="English"
    )
    
    logger.info(f"Translation result: {translation}")
    assert translation, "Translation returned empty string"
    # Basic check - look for common Spanish greetings
    assert any(word in translation.lower() for word in ["hola", "cómo", "estas", "estás", "qué", "tal"]), \
        f"Unexpected translation: {translation}"

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_llm_translation_with_context():
    """Test translation with context."""
    context = "I am a software engineer."
    source_text = "I write code every day."
    target_lang = "French"
    
    logger.info(f"Testing contextual translation to {target_lang}")
    translation = await translate_text(
        text=source_text,
        target_language_name=target_lang,
        context_text=context,
        source_language_name="English"
    )
    
    logger.info(f"Translation result: {translation}")
    assert translation, "Translation returned empty string"
    # "code" in french is "code" too usually, verify it's not just the source text returned unaltered if possible
    # But checking for non-empty and different from source is a decent start
    assert translation != source_text, "Translation identical to source"

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_llm_title_generation():
    """Test title generation from a short paragraph."""
    text = (
        "Artificial Intelligence has rapidly evolved in recent years. "
        "From basic pattern recognition to complex generative models, AI is reshaping industries. "
        "Developers are now building agents that can plan and execute tasks autonomously."
    )
    target_lang = "English"
    
    logger.info("Testing title generation...")
    title = await generate_title(translated_text=text, target_language_name=target_lang)
    
    logger.info(f"Generated Title: {title}")
    assert title, "Title generation failed"
    # Logic check: Title should be short
    assert len(title.split()) < 20, f"Title too long: {title}"

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_llm_summarization():
    """Test summarization of a longer text."""
    text = (
        "The Python programming language was created by Guido van Rossum and first released in 1991. "
        "Python's design philosophy emphasizes code readability with its significant whitespace. "
        "Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects. "
        "Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly, procedural), object-oriented, and functional programming. "
        "Python is often described as a 'batteries included' language due to its comprehensive standard library."
    )
    label = "Technical History"
    target_lang = "English"
    
    logger.info("Testing summarization...")
    summary = await summarize_text(
        text=text,
        label=label,
        target_language_name=target_lang
    )
    
    logger.info(f"Summary: {summary}")
    assert summary, "Summarization returned empty string"
    # Summary should significantly shorter than original text if text was huge, 
    # but here text is moderate. Just check it exists and is not identical to input.
    assert summary != text, "Summary is identical to input"
