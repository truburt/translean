curl -X POST http://localhost:7860/transcribe -F file=@sample.wav -F params='{"language": "en", "beam_size": 5, "task": "transcribe"}'
