"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
"""
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
import pytest_asyncio

from app.db import Base
from app.repositories import (
    add_paragraph,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_or_create_user,
    list_conversations,
    mark_conversation_pending_deletion,
    restore_conversation,
    delete_pending_conversations,
)
from app.schemas import ConversationCreate, UserInfo


@pytest_asyncio.fixture()
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    TestSession = async_sessionmaker(engine, expire_on_commit=False)
    async with TestSession() as sess:
        yield sess


@pytest.mark.asyncio
async def test_conversation_lifecycle(session):
    user = await get_or_create_user(session, UserInfo(sub="sub1", email="user@example.com", name="User"))
    convo = await create_conversation(
        session,
        user,
        ConversationCreate(source_language="en", target_language="es", title=None),
    )
    await session.commit()

    await add_paragraph(session, convo, 0, "hello", "hola")
    await session.commit()

    fetched = await get_conversation(session, user, convo.id)
    assert fetched is not None
    assert fetched.paragraphs[0].translated_text == "hola"
    assert fetched.title.startswith("hola")

    removed = await delete_conversation(session, user, convo.id)
    await session.commit()
    assert removed is True


@pytest.mark.asyncio
async def test_pending_deletion_lifecycle(session):
    user = await get_or_create_user(session, UserInfo(sub="sub2", email="user2@example.com", name="User2"))
    
    # Create two conversations
    convo1 = await create_conversation(
        session,
        user,
        ConversationCreate(source_language="en", target_language="es", title="Convo 1"),
    )
    convo2 = await create_conversation(
        session,
        user,
        ConversationCreate(source_language="en", target_language="fr", title="Convo 2"),
    )
    await add_paragraph(session, convo1, 0, "Hello", "Hola")
    await add_paragraph(session, convo2, 0, "Bonjour", "Hello")
    await session.commit()

    # Initially both should be listed
    conversations = await list_conversations(session, user)
    assert len(conversations) == 2

    # Mark first conversation as pending deletion
    marked = await mark_conversation_pending_deletion(session, user, convo1.id)
    await session.commit()
    assert marked is True

    # Now only one should be listed
    conversations = await list_conversations(session, user)
    assert len(conversations) == 1
    assert conversations[0].id == convo2.id

    # Restore the conversation
    restored = await restore_conversation(session, user, convo1.id)
    await session.commit()
    assert restored is True

    # Both should be listed again
    conversations = await list_conversations(session, user)
    assert len(conversations) == 2

    # Mark for deletion again
    await mark_conversation_pending_deletion(session, user, convo1.id)
    await session.commit()

    # Delete all pending conversations
    await delete_pending_conversations(session, user)
    await session.commit()

    # Only one conversation should remain
    conversations = await list_conversations(session, user)
    assert len(conversations) == 1
    assert conversations[0].id == convo2.id
    
    # The pending deletion conversation should be gone completely
    fetched = await get_conversation(session, user, convo1.id)
    assert fetched is None
