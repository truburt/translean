"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

SQLAlchemy ORM models describing core domain entities.
"""
import datetime as dt
import uuid
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _utcnow():
    """Return a timezone-aware UTC datetime."""
    return dt.datetime.now(dt.timezone.utc)


class TimestampMixin:
    """Shared created/updated timestamps."""

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    oidc_sub: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=True)

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_language: Mapped[str] = mapped_column(String(255), nullable=False)
    target_language: Mapped[str] = mapped_column(String(50), nullable=False)
    is_title_manual: Mapped[bool] = mapped_column(default=False, nullable=False)
    pending_deletion: Mapped[bool] = mapped_column(default=False, nullable=False)

    user: Mapped[User] = relationship(back_populates="conversations")
    paragraphs: Mapped[list["Paragraph"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Paragraph.paragraph_index"
    )


class Paragraph(Base, TimestampMixin):
    __tablename__ = "paragraphs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    paragraph_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    translated_text: Mapped[str] = mapped_column(Text, nullable=False)
    detected_language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    type: Mapped[str] = mapped_column(String(50), default="active", nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="paragraphs")


class AppConfig(Base, TimestampMixin):
    __tablename__ = "app_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    whisper_base_url: Mapped[str] = mapped_column(String(255), nullable=False)
    whisper_model: Mapped[str] = mapped_column(String(255), nullable=False)
    whisper_keep_alive_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    ollama_base_url: Mapped[str] = mapped_column(String(255), nullable=False)
    llm_model_translation: Mapped[str] = mapped_column(String(255), nullable=False)
    ollama_keep_alive_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    commit_timeout_seconds: Mapped[float] = mapped_column(nullable=False)
    silence_finalize_seconds: Mapped[float] = mapped_column(nullable=False)
    min_preview_buffer_seconds: Mapped[float] = mapped_column(nullable=False)
    stable_window_seconds: Mapped[float] = mapped_column(nullable=False)
    no_speech_prob_skip: Mapped[float] = mapped_column(nullable=False)
    no_speech_prob_logprob_skip: Mapped[float] = mapped_column(nullable=False)
    avg_logprob_skip: Mapped[float] = mapped_column(nullable=False)
    compression_ratio_skip: Mapped[float] = mapped_column(nullable=False)
