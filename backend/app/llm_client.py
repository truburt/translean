"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

Lightweight Ollama translation client.
"""
import logging
import re
import math
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Dict

import httpx

from .runtime_config import runtime_config
from .languages import get_language_code, get_language_name

logger = logging.getLogger(__name__)

# Configs
MAX_SUMMARY_CONTEXT_TOKENS = 6000
CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_token_count(text: str) -> int:
    """Lightweight token estimate using average characters per token."""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def estimate_max_chars(token_limit: int, safe_mode: bool = True) -> int:
    """
    Estimate max characters that fit into a token limit.
    If safe_mode is True, we assume the output will be roughly the same length as input, 
    so we should only use half the available tokens for input.
    """
    if safe_mode:
        token_limit = token_limit // 2
        
    return token_limit * CHARS_PER_TOKEN_ESTIMATE


class RateLimiter:
    """
    A rate limiter that ensures actions are not performed more frequently than a given interval.
    """
    def __init__(self, interval: float):
        self.interval = interval
        self.last_check = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Wait until enough time has passed since the last action.
        """
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_check
            wait_time = self.interval - elapsed
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            self.last_check = time.monotonic()

# Global translation rate limiter (1 request per second)
translation_limiter = RateLimiter(1.0)


# ==================================================================================
# Base Client & Helpers
# ==================================================================================

class BaseLLMClient(ABC):
    """
    Abstract base class for LLM interactions.
    Subclasses implement model-specific prompting and cleanup logic.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def translate(
        self, 
        text: str, 
        target_language: str, 
        context: str = "", 
        source_language: str = "Auto", 
        num_ctx: int = 4096, 
        timeout: float = 30.0
    ) -> str:
        """Translate text into target language."""
        pass

    @abstractmethod
    async def summarize(
        self, 
        text: str, 
        label: str, 
        target_language: str, 
        num_ctx: int = 8192
    ) -> str:
        """Summarize text."""
        pass

    @abstractmethod
    async def generate_title(
        self, 
        translated_text: str, 
        target_language: str
    ) -> str:
        """Generate a title for the content."""
        pass

    @abstractmethod
    async def cleanup(
        self, 
        text: str, 
        num_ctx: int = 8192
    ) -> str:
        """Clean up raw transcription."""
        pass

    @abstractmethod
    async def warm_up(self) -> bool:
        """Ensure model is loaded."""
        pass

    async def _post_ollama(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: float,
        purpose: str
    ) -> Dict[str, Any]:
        """Shared HTTP execution logic."""
        logger.debug("Ollama %s via %s (%s)", purpose, self.model_name, endpoint)
        async with httpx.AsyncClient(base_url=runtime_config.ollama_base_url, timeout=timeout) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            return data

    def _extract_response(self, data: Dict[str, Any]) -> str:
        """Extract content from either chat or generate response."""
        message = data.get("message")
        if isinstance(message, dict):
            return message.get("content") or ""
        return data.get("response") or data.get("output") or ""


# ==================================================================================
# Qwen Implementation (Existing + Refactored)
# ==================================================================================

class QwenClient(BaseLLMClient):
    """
    Client for Qwen models (e.g. Qwen2, Qwen2.5, Qwen3).
    """

    def _clean_response(self, text: str) -> str:
        """Remove think tags and other artifacts."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "</think>" in text:
            text = text.split("</think>", 1)[-1]
        text = text.replace("/think", "").replace("/no_think", "")
        return text.strip()

    def _build_system_prompt(self, target_language_name: str) -> str:
        return f"You are a live translator to {target_language_name} /no_think"

    def _build_translation_prompt(
        self,
        source_text: str,
        target_language_name: str,
        context_text: str,
        source_language_name: str,
    ) -> str:
        source_label = (source_language_name or "").strip()
        has_source = bool(source_label) and source_label.lower() != "auto"

        header = ""
        if has_source:
            header += f"Source language: {source_label}\n"
        header += f"Target language: {target_language_name}\n\n"

        if context_text:
            return (
                f"{header}"
                f"Preceding {'speech' if not has_source else f'{source_label} speech'} was:\n{context_text}\n\n"
                f"Translate the following {'speech' if not has_source else f'{source_label} speech'} into {target_language_name} as the continuation of the preceding speech:\n"
                f"{source_text}"
            )

        return (
            f"{header}"
            f"Translate the following {'text' if not has_source else f'{source_label} text'} into {target_language_name}:\n"
            f"{source_text}"
        )

    async def translate(
        self, 
        text: str, 
        target_language: str, 
        context: str = "", 
        source_language: str = "Auto", 
        num_ctx: int = 4096, 
        timeout: float = 30.0
    ) -> str:
        system = self._build_system_prompt(target_language)
        prompt = self._build_translation_prompt(text, target_language, context, source_language)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": runtime_config.ollama_keep_alive_seconds,
            "options": {"num_ctx": num_ctx}
        }

        try:
            data = await self._post_ollama("/api/generate", payload, timeout, "translation")
            content = self._extract_response(data)
            return self._clean_response(content)
        except Exception as e:
            logger.error("Qwen translation error: %r", e)
            return ""

    async def summarize(
        self, 
        text: str, 
        label: str, 
        target_language: str, 
        num_ctx: int = 8192
    ) -> str:
        prompt = (
            f"Provide a concise summary (2-3 sentences) in {target_language} of the {label.lower()} speech below "
            f"so it can be displayed as a recap. {label} speech: {text}"
        )
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": runtime_config.ollama_keep_alive_seconds,
            "options": {"num_ctx": num_ctx},
            "think": True
        }

        try:
            data = await self._post_ollama("/api/generate", payload, 60.0, f"{label} summary")
            content = self._extract_response(data)
            return self._clean_response(content)
        except Exception as e:
            logger.error("Qwen summary error: %r", e)
            return ""

    async def generate_title(self, translated_text: str, target_language: str) -> str:
        prompt = (
            f"Generate a short, descriptive title (max 12 words) in {target_language} for a speech. Output only the single title. The speech is:"
            f"\n\n{translated_text}"
        )
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": runtime_config.ollama_keep_alive_seconds,
            "think": False
        }
        
        try:
            data = await self._post_ollama("/api/generate", payload, 30.0, "title generation")
            content = self._extract_response(data)
            return self._clean_response(content)
        except Exception as e:
            logger.error("Qwen title error: %r", e)
            return ""

    async def cleanup(self, text: str, num_ctx: int = 8192) -> str:
        prompt = (
            "The following text is a raw transcription of speech. "
            "Please clean it up by removing duplicates, correction artifacts, and disfluencies to improve readability. "
            "Do not summarize or change the meaning, just fix the grammar and flow. "
            "Output ONLY the cleaned text.\n\n"
            f"Raw Transcription: {text}"
        )
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": runtime_config.ollama_keep_alive_seconds,
            "options": {"num_ctx": num_ctx},
            "think": False
        }
        
        try:
            data = await self._post_ollama("/api/generate", payload, 60.0, "cleanup")
            content = self._extract_response(data)
            return self._clean_response(content)
        except Exception as e:
            logger.error("Qwen cleanup error: %r", e)
            return text

    async def warm_up(self) -> bool:
        try:
            await self._post_ollama(
                "/api/generate", 
                {"model": self.model_name, "prompt": "", "stream": False, "keep_alive": runtime_config.ollama_keep_alive_seconds}, 
                300.0, 
                "warm-up"
            )
            return True
        except Exception:
            return False


# ==================================================================================
# TranslateGemma Implementation
# ==================================================================================

class TranslateGemmaClient(BaseLLMClient):
    """
    Client for TranslateGemma models which require strict prompt formatting.
    """
    
    def _get_lang_details(self, lang_name_or_code: str) -> tuple[str, str]:
        """Return (Language Name, Language Code)."""
        code = get_language_code(lang_name_or_code)
        name = get_language_name(code)
        return name, code

    def _build_strict_prompt(self, source_text: str, source_lang_in: str, target_lang_in: str) -> str:
        # Resolve source/target to (Name, Code)
        # Handle Auto
        src_name, src_code = self._get_lang_details(source_lang_in)
        tgt_name, tgt_code = self._get_lang_details(target_lang_in)
        
        # Prompt template from documentation
        return (
            f"You are a professional {src_name} ({src_code}) to {tgt_name} ({tgt_code}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the original {src_name} text "
            f"while adhering to {tgt_name} grammar, vocabulary, and cultural sensitivities. "
            f"Produce only the {tgt_name} translation, without any additional explanations or commentary. "
            f"Please translate the following {src_name} text into {tgt_name}:\n\n\n{source_text}"
        )

    async def translate(
        self, 
        text: str, 
        target_language: str, 
        context: str = "", 
        source_language: str = "Auto", 
        num_ctx: int = 4096, 
        timeout: float = 30.0
    ) -> str:
        # NOTE: TranslateGemma guide doesn't mention context support in the strict template.
        # We will ignore context for now to ensure robustness with the model.
        
        prompt_content = self._build_strict_prompt(text, source_language, target_language)
        
        # Use Chat endpoint format as it describes "user message"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_content}],
            "stream": False,
            "keep_alive": runtime_config.ollama_keep_alive_seconds,
            "options": {"num_ctx": num_ctx}
        }

        try:
            data = await self._post_ollama("/api/chat", payload, timeout, "translation")
            content = self._extract_response(data)
            return content.strip()
        except Exception as e:
            logger.error("TranslateGemma translation error: %r", e)
            return ""

    async def summarize(
        self, 
        text: str, 
        label: str, 
        target_language: str, 
        num_ctx: int = 8192
    ) -> str:
        # TranslateGemma is specialized. Using it for summarization might yield poor results.
        # We try a generic instruction using the chat format.
        prompt = (
            f"Provide a concise summary (2-3 sentences) in {target_language} of the following text. Output only the summary. The text is:\n\n{text}"
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": runtime_config.ollama_keep_alive_seconds,
            "options": {"num_ctx": num_ctx}
        }
        try:
            data = await self._post_ollama("/api/chat", payload, 60.0, f"{label} summary")
            return self._extract_response(data).strip()
        except Exception as e:
            logger.error("TranslateGemma summary error: %r", e)
            return ""

    async def generate_title(self, translated_text: str, target_language: str) -> str:
        prompt = (
            f"Generate a short, descriptive title (max 12 words) in {target_language} for a text. Output only the title. The text is:\n\n{translated_text}"
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": runtime_config.ollama_keep_alive_seconds
        }
        try:
            data = await self._post_ollama("/api/chat", payload, 30.0, "title generation")
            return self._extract_response(data).strip()
        except Exception as e:
            logger.error("TranslateGemma title error: %r", e)
            return ""

    async def cleanup(self, text: str, num_ctx: int = 8192) -> str:
        prompt = (
            "Clean up the following text by fixing grammar and removing disfluencies. Output ONLY the cleaned text. The text is:\n\n" + text
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": runtime_config.ollama_keep_alive_seconds,
            "options": {"num_ctx": num_ctx}
        }
        try:
            data = await self._post_ollama("/api/chat", payload, 60.0, "cleanup")
            return self._extract_response(data).strip()
        except Exception as e:
            logger.error("TranslateGemma cleanup error: %r", e)
            return text

    async def warm_up(self) -> bool:
        try:
            await self._post_ollama(
                "/api/chat", 
                {"model": self.model_name, "messages": [{"role": "user", "content": "hi"}], "stream": False, "keep_alive": runtime_config.ollama_keep_alive_seconds}, 
                300.0, 
                "warm-up"
            )
            return True
        except Exception:
            return False


# ==================================================================================
# Factory & Public Interface
# ==================================================================================

def get_llm_client(model_name: str) -> BaseLLMClient:
    """Factory to return the appropriate LLM client based on model name."""
    lower_name = model_name.lower()
    
    if "translategemma" in lower_name:
        return TranslateGemmaClient(model_name)
    
    # Fallback to Qwen/Generic for qwen, gpt-oss, gemma3 (standard), etc.
    return QwenClient(model_name)


async def warm_up() -> bool:
    """Send an empty prompt to warm up the model and keep it alive."""
    client = get_llm_client(runtime_config.llm_model_translation)
    return await client.warm_up()


async def translate_text(
    text: str,
    target_language_name: str,
    context_text: str = "",
    source_language_name: str = "Auto",
    num_ctx: int = 4096,
    timeout: float = 30.0,
) -> str:
    """Translate a single text string."""
    client = get_llm_client(runtime_config.llm_model_translation)
    return await client.translate(
        text, 
        target_language_name, 
        context_text, 
        source_language_name, 
        num_ctx, 
        timeout
    )


async def generate_title(translated_text: str, target_language_name: str) -> str:
    """Create a concise conversation title from translated content."""
    client = get_llm_client(runtime_config.llm_model_translation)
    return await client.generate_title(translated_text, target_language_name)


async def summarize_text(
    text: str, 
    label: str, 
    target_language_name: str, 
    num_ctx: int = 8192,
) -> str:
    """Summarize a block of text with a short description label."""
    client = get_llm_client(runtime_config.llm_model_translation)
    return await client.summarize(text, label, target_language_name, num_ctx)


async def cleanup_text(text: str, num_ctx: int = 8192) -> str:
    """Clean up raw transcription text."""
    client = get_llm_client(runtime_config.llm_model_translation)
    return await client.cleanup(text, num_ctx)
