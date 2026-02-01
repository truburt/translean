class DummyLimiter:
    async def acquire(self):
        return None

    def release(self):
        return None

async def mock_warm_up(ws, lang, whisper_ready, llm_ready):
    whisper_ready.set()
    llm_ready.set()
