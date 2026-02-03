"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

Pydantic schemas for API IO models.
"""
import uuid
from datetime import datetime
from typing import List, Optional, Union
from urllib.parse import urlparse
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParagraphOut(BaseModel):
    id: int
    paragraph_index: int
    source_text: str
    translated_text: str
    detected_language: Optional[str] = None
    type: str
    created_at: datetime
    updated_at: datetime


    model_config = ConfigDict(from_attributes=True)


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    source_language: str
    target_language: str
    is_title_manual: bool
    created_at: datetime
    updated_at: datetime
    paragraphs: List[ParagraphOut] = Field(default_factory=list)


    model_config = ConfigDict(from_attributes=True)


class ConversationTitleUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class ConversationCreate(BaseModel):
    source_language: Union[str, List[str]] = Field(...)
    target_language: str = Field(..., min_length=2, max_length=50)
    title: Optional[str] = None


class ConversationListItem(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    source_language: str
    target_language: str
    created_at: datetime
    updated_at: datetime


    model_config = ConfigDict(from_attributes=True)


class UserInfo(BaseModel):
    sub: str
    email: str
    name: Optional[str] = None


class ServerConfig(BaseModel):
    whisper_base_url: str = Field(..., min_length=1, max_length=255)
    whisper_model: str = Field(..., min_length=1, max_length=255)
    whisper_keep_alive_seconds: int = Field(..., ge=0, le=86400)
    ollama_base_url: str = Field(..., min_length=1, max_length=255)
    llm_model_translation: str = Field(..., min_length=1, max_length=255)
    ollama_keep_alive_seconds: int = Field(..., ge=0, le=86400)
    commit_timeout_seconds: float = Field(..., ge=1.0, le=60.0)
    silence_finalize_seconds: float = Field(..., ge=0.1, le=10.0)
    min_preview_buffer_seconds: float = Field(..., ge=0.1, le=5.0)
    stable_window_seconds: float = Field(..., ge=1.0, le=30.0)
    no_speech_prob_skip: float = Field(..., ge=0.0, le=1.0)
    no_speech_prob_logprob_skip: float = Field(..., ge=0.0, le=1.0)
    avg_logprob_skip: float = Field(..., ge=-10.0, le=0.0)
    compression_ratio_skip: float = Field(..., ge=1.0, le=10.0)

    @field_validator("whisper_base_url", "ollama_base_url")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Endpoint must be a valid http(s) URL")
        return value

    @field_validator("stable_window_seconds")
    @classmethod
    def validate_stable_window(cls, value: float, info) -> float:
        min_preview = info.data.get("min_preview_buffer_seconds")
        if min_preview is not None and value < min_preview:
            raise ValueError("Stable window must be at least the preview buffer duration")
        return value
