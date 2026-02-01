
from app.whisper_client import _clean_transcription

def test_clean_transcription():
    input_text = ", it means something beautiful. Yeah, it means you didn't give up. yet. You're... still here. Still learning. , even if it's hard."
    expected_output = "it means something beautiful. Yeah, it means you didn't give up. yet. You're still here. Still learning, even if it's hard."
    
    cleaned = _clean_transcription(input_text)
    
    print(f"Input:    '{input_text}'")
    print(f"Expected: '{expected_output}'")
    print(f"Actual:   '{cleaned}'")
    
    if cleaned == expected_output:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    test_clean_transcription()
