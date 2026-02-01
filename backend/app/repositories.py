"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

Database access helpers for conversations and paragraphs.
"""
import logging
import uuid

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from . import models
from .schemas import ConversationCreate, UserInfo

logger = logging.getLogger(__name__)


def model_to_dict(model):
    """Convert a SQLAlchemy model into a plain dictionary."""
    return {c.name: getattr(model, c.name) for c in model.__table__.columns}


async def get_or_create_user(session: AsyncSession, user: UserInfo) -> models.User:
    """Return an existing user or create a new one from OIDC details."""
    result = await session.execute(select(models.User).where(models.User.oidc_sub == user.sub))
    existing = result.scalar_one_or_none()
    if existing:
        return existing
    db_user = models.User(oidc_sub=user.sub, email=user.email, display_name=user.name)
    session.add(db_user)
    await session.flush()
    return db_user


async def create_conversation(session: AsyncSession, user: models.User, data: ConversationCreate) -> models.Conversation:
    """Create a conversation record for the user."""
    conversation = models.Conversation(
        user=user,
        title=data.title,
        source_language=data.source_language if isinstance(data.source_language, str) else ",".join(data.source_language),
        target_language=data.target_language,
    )
    session.add(conversation)
    await session.flush()
    return conversation


async def add_paragraph(
    session: AsyncSession,
    conversation: models.Conversation,
    paragraph_index: int,
    source_text: str,
    translated_text: str,
    detected_language: str = None,
    is_refined: bool = False,
    paragraph_type: str = "active",
) -> models.Paragraph:
    """Add a paragraph entry to a conversation."""
    paragraph = models.Paragraph(
        conversation=conversation,
        paragraph_index=paragraph_index,
        source_text=source_text,
        translated_text=translated_text,
        detected_language=detected_language,
        type=paragraph_type,
    )
    session.add(paragraph)
    if conversation.title is None:
        conversation.title = (translated_text or source_text)[:80]
    await session.flush()
    await session.refresh(paragraph)
    return paragraph


async def update_paragraph_content(
    session: AsyncSession,
    paragraph_id: uuid.UUID,
    source_append: str | None,
    translation_append: str | None,
    detected_language: str = None,
    paragraph_type: str | None = None,
    source_override: str | None = None,
    translation_override: str | None = None,
) -> models.Paragraph:
    """Append new content and metadata to a paragraph.

    translation_override replaces the existing translated_text instead of appending.
    """
    result = await session.execute(select(models.Paragraph).where(models.Paragraph.id == paragraph_id))
    paragraph = result.scalar_one_or_none()

    if paragraph:
        if source_override is not None:
            paragraph.source_text = source_override
        elif source_append:
            if paragraph.source_text:
                paragraph.source_text += " " + source_append
            else:
                paragraph.source_text = source_append
        
        if translation_override is not None:
            paragraph.translated_text = translation_override
        elif translation_append:
            if paragraph.translated_text:
                paragraph.translated_text += " " + translation_append
            else:
                paragraph.translated_text = translation_append
        
        if detected_language:
            paragraph.detected_language = detected_language

        # Preserve existing type if not provided so queued updates don't revert finalized paragraphs.
        if paragraph_type is not None:
            paragraph.type = paragraph_type
                
        await session.flush()
    return paragraph


async def list_conversations(session: AsyncSession, user: models.User, limit: int = 20, offset: int = 0):
    """List conversations for a user ordered by recent update time."""
    result = await session.execute(
        select(models.Conversation)
        .where(models.Conversation.user_id == user.id)
        .where(models.Conversation.pending_deletion == False)
        .where(models.Conversation.paragraphs.any())
        .order_by(models.Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


async def get_conversation(session: AsyncSession, user: models.User | None, conversation_id: uuid.UUID):
    """Retrieve a conversation with its paragraphs for the user (or by ID if user is None)."""
    stmt = select(models.Conversation).where(models.Conversation.id == conversation_id)
    if user:
        stmt = stmt.where(models.Conversation.user_id == user.id)
        
    result = await session.execute(stmt.options(selectinload(models.Conversation.paragraphs)))
    return result.scalar_one_or_none()


async def delete_conversation(session: AsyncSession, user: models.User, conversation_id: uuid.UUID) -> bool:
    """Delete a single conversation for the user."""
    stmt = delete(models.Conversation).where(
        models.Conversation.id == conversation_id, models.Conversation.user_id == user.id
    )
    result = await session.execute(stmt)
    return result.rowcount > 0


async def delete_all_conversations(session: AsyncSession, user: models.User):
    """Delete all user conversations."""
    stmt = delete(models.Conversation).where(models.Conversation.user_id == user.id)
    await session.execute(stmt)


async def mark_conversation_pending_deletion(session: AsyncSession, user: models.User, conversation_id: uuid.UUID) -> bool:
    """Mark a conversation as pending deletion."""
    result = await session.execute(
        select(models.Conversation)
        .where(models.Conversation.id == conversation_id)
        .where(models.Conversation.user_id == user.id)
    )
    conversation = result.scalar_one_or_none()
    if conversation:
        conversation.pending_deletion = True
        await session.flush()
        return True
    return False


async def restore_conversation(session: AsyncSession, user: models.User, conversation_id: uuid.UUID) -> bool:
    """Restore a conversation from pending deletion."""
    result = await session.execute(
        select(models.Conversation)
        .where(models.Conversation.id == conversation_id)
        .where(models.Conversation.user_id == user.id)
    )
    conversation = result.scalar_one_or_none()
    if conversation:
        conversation.pending_deletion = False
        await session.flush()
        return True
    return False


async def delete_pending_conversations(session: AsyncSession, user: models.User):
    """Permanently delete all conversations marked as pending deletion for a user."""
    stmt = delete(models.Conversation).where(
        models.Conversation.user_id == user.id,
        models.Conversation.pending_deletion == True
    )
    await session.execute(stmt)
