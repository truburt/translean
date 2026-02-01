"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

Pydantic schemas for API IO models.
"""
import uuid
from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


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
