
import pytest
from app.llm_client import QwenClient, estimate_max_chars

# Instantiate a client for testing
client = QwenClient("test-model")

def test_clean_llm_response_no_tags():
    raw = "Just a normal response."
    assert client._clean_response(raw) == "Just a normal response."

def test_clean_llm_response_simple_think():
    raw = "<think>Hmm, let me think.</think>Here is the answer."
    assert client._clean_response(raw) == "Here is the answer."

def test_clean_llm_response_multiline_think():
    raw = """<think>
    Thinking process...
    Step 1.
    </think>
    Actual result.
    """
    cleaned = client._clean_response(raw)
    assert "Thinking process" not in cleaned
    assert "Actual result." in cleaned
    assert cleaned.strip() == "Actual result."

def test_clean_llm_response_slash_tags():
    raw = "Some text /think and /no_think tags."
    expected = "Some text  and  tags."
    # Note: double spaces might remain if we just replace with empty string.
    # The user said "remove ... tags".
    assert client._clean_response(raw).replace("  ", " ") == expected.replace("  ", " ")

def test_clean_llm_response_dangling_end_tag():
    # User mentioned "...thinking text...</think>...actual text..."
    # implying potentially missing start tag
    raw = "Thinking logic here...</think>Final Answer"
    assert client._clean_response(raw) == "Final Answer"

def test_clean_llm_response_mixed():
    raw = "/think <think>Process</think> /no_think Result"
    assert client._clean_response(raw).strip() == "Result"


def test_build_translation_prompt_omits_auto_source():
    prompt = client._build_translation_prompt(
        source_text="Hello", 
        target_language_name="French", 
        context_text="",
        source_language_name="Auto"
    )
    assert "Source language" not in prompt
    assert "Target language: French" in prompt
    assert "following text" in prompt
    
def test_estimate_max_chars():
    
    # CHARS_PER_TOKEN_ESTIMATE = 4
    
    # Case 1: 1000 tokens, safe_mode=True
    # Limit = 500 tokens
    # Chars = 2000
    assert estimate_max_chars(1000, safe_mode=True) == 2000
    
    # Case 2: 1000 tokens, safe_mode=False
    # Limit = 1000 tokens
    # Chars = 4000
    assert estimate_max_chars(1000, safe_mode=False) == 4000
    
    # Case 3: Small limit
    # 100 tokens, safe=True -> 50 tokens -> 200 chars
    assert estimate_max_chars(100, safe_mode=True) == 200
