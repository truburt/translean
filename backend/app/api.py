"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

API router definitions for health, conversations, and streaming WebSocket.
"""
import datetime as dt
import os
import json
import math
import re
import time
import contextlib

import logging
import secrets
import urllib.parse
import uuid
import asyncio
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from jose import jwt
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from . import models
from .auth import decode_token, exchange_code_for_token, get_current_user, get_current_user_websocket, get_authorization_endpoint
from .config import settings
from .db import SessionLocal, get_session
from .llm_client import (
    generate_title,
    summarize_text,
    translate_text,
    warm_up as warm_up_llm,
    cleanup_text,
    translation_limiter,
    estimate_token_count,
    estimate_max_chars,
    MAX_SUMMARY_CONTEXT_TOKENS,
)
from .repositories import (
    add_paragraph,
    create_conversation,
    delete_conversation,
    delete_all_conversations,
    delete_pending_conversations,
    get_conversation,
    get_or_create_user,
    list_conversations,
    mark_conversation_pending_deletion,
    restore_conversation,
    update_paragraph_content,
)
from .schemas import ConversationCreate, ConversationListItem, ConversationOut, UserInfo, ConversationTitleUpdate
from .whisper_client import stream_transcription, warm_up as warm_up_whisper

logger = logging.getLogger(__name__)

REBUILD_CONTEXT_TOKENS = 4096

class ConnectionManager:
    """Manages active WebSocket connections by conversation ID."""
    def __init__(self):
        self.active_connections: dict[uuid.UUID, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, conversation_id: uuid.UUID, websocket: WebSocket):
        async with self._lock:
            if conversation_id not in self.active_connections:
                self.active_connections[conversation_id] = []
            self.active_connections[conversation_id].append(websocket)
            logger.info("ConnectionManager: Client connected to conversation %s. Total clients: %s", conversation_id, len(self.active_connections[conversation_id]))

    async def disconnect(self, conversation_id: uuid.UUID, websocket: WebSocket):
        async with self._lock:
            if conversation_id in self.active_connections:
                if websocket in self.active_connections[conversation_id]:
                    self.active_connections[conversation_id].remove(websocket)
                    logger.info("ConnectionManager: Client disconnected from conversation %s", conversation_id)
                if not self.active_connections[conversation_id]:
                    del self.active_connections[conversation_id]
                    logger.info("ConnectionManager: No more clients for conversation %s. Removed from active_connections.", conversation_id)

    async def broadcast(self, conversation_id: uuid.UUID, message: str | dict):
        # logging.info("ConnectionManager: Broadcasting to %s. Active keys: %s", conversation_id, list(self.active_connections.keys()))
        if not self.active_connections.get(conversation_id):
            logger.warning("ConnectionManager: Broadcast skipped. No active connections for conversation %s. Keys: %s", conversation_id, list(self.active_connections.keys()))
            return

        if isinstance(message, dict):
            message = json.dumps(message)

        # Snapshot current connections to avoid issues if one disconnects during iteration
        async with self._lock:
            clients = list(self.active_connections.get(conversation_id, []))
        
        logger.info("ConnectionManager: Sending message to %d clients for conversation %s", len(clients), conversation_id)
        for connection in clients:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning("ConnectionManager: Failed to send to client: %s", e)
                # Cleanup handled by the websocket route handler on disconnect,
                # but we could proactively remove here if we wanted.

manager = ConnectionManager()

router = APIRouter()


def _safe_paragraph_index(p):
    """Return a sortable paragraph index, guarding against mocks or missing attributes."""
    try:
        return int(getattr(p, "paragraph_index", 0) or 0)
    except Exception:
        return 0


@router.get("/health")
async def health():
    """Liveness probe for the API service."""
    return {"status": "ok", "service": settings.app_name}


async def run_warm_up(websocket: WebSocket, whisper_lang: str, whisper_ready: asyncio.Event, llm_ready: asyncio.Event):
    """Warm up Whisper and LLM models, notifying the client when each is ready."""
    try:
        try:
            await websocket.send_json({"status": "warming_up"})
        except Exception:
            pass  # Client might have disconnected

        # measure warm-up time
        start_time = time.time()
        logger.info("Starting Whisper model warm-up...")
        res_whisper = await warm_up_whisper(whisper_lang)
        if not res_whisper:
            logger.error("Whisper model warm-up failed.")
            try:
                await websocket.send_json({
                    "status": "error",
                    "service": "whisper",
                    "error_code": "WHISPER_INIT_FAILED",
                    "error": "Whisper model initialization failed. Please check backend logs."
                })
            except Exception as send_exc:
                logger.error("Failed to send Whisper initialization error to client: %s", send_exc)
                pass
            return
            
        # Whisper is ready
        whisper_ready.set()
        logger.info("Whisper warm-up completed in %s seconds", time.time() - start_time)
        try:
            await websocket.send_json({"status": "whisper_ready"})
        except Exception:
            pass

        logger.info("Starting Ollama model warm-up...")
        llm_start_time = time.time()
        res_llm = await warm_up_llm()
        if not res_llm:
            logger.error("LLM model warm-up failed.")
            try:
                await websocket.send_json({
                    "status": "error",
                    "service": "llm",
                    "error_code": "LLM_INIT_FAILED",
                    "error": "LLM model initialization failed. Please check backend logs."
                })
            except Exception:
                pass
            # Don't return, allow whisper to continue if already ready
        else:
            llm_ready.set()
            logger.info("LLM warm-up completed in %s seconds", time.time() - llm_start_time)
            try:
                await websocket.send_json({"status": "llm_ready"})
            except Exception:
                pass

        logger.info("Total warm-up sequence completed in %s seconds", time.time() - start_time)
        
    except Exception as e:
        logger.warning("Model warm-up encountered an error: %s", e)
        try:
            await websocket.send_json({
                "status": "error",
                "service": "system",
                "error": f"Model initialization error: {str(e)}"
            })
        except Exception:
            pass


from .languages import SOURCE_LANGUAGES, TARGET_LANGUAGES, get_language_code, get_language_name, pick_language_name, get_supported_languages
from .config import settings



def get_prevalent_other_language(conversation: models.Conversation, excluded_name: str) -> str:
    """
    Scan conversation paragraphs to find the most frequent detected language
    that is NOT the excluded language. Returns 'English' fallback if none found.
    """
    if not conversation or not conversation.paragraphs:
        return "English"

    excluded_code = get_language_code(excluded_name)
    counts = {}

    for p in conversation.paragraphs:
        if not p.detected_language:
            continue
            
        langs = [l.strip() for l in p.detected_language.split(",") if l.strip()]
        for lang_code in langs:
            # Normalize to code
            code = get_language_code(lang_code)
            if code == "auto": 
                continue
            if code == excluded_code:
                continue
                
            counts[code] = counts.get(code, 0) + 1

    if not counts:
        return None

    # Find max
    top_code = max(counts, key=counts.get)
    return get_language_name(top_code)


def chunk_text_by_sentence(text: str, max_chars: int = 1000) -> list[str]:
    """Split text into sentence-aware chunks capped at roughly `max_chars` characters."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        # If no sentences found, hard split by chars
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        # +1 for space if current is not empty
        added_len = sentence_len + (1 if current else 0)
        
        if current_len + added_len > max_chars and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def split_text_for_summary(text: str, token_limit: int = MAX_SUMMARY_CONTEXT_TOKENS) -> list[str]:
    """Split text into chunks that are likely to fit within the summarization window."""
    if not text.strip():
        return []

    words = text.split()
    estimated_tokens = estimate_token_count(text)
    chunks_needed = max(2, math.ceil(estimated_tokens / token_limit))
    
    # Calculate chars per chunk based on total length
    chars_per_chunk = max(300, math.ceil(len(text) / chunks_needed))

    chunks = chunk_text_by_sentence(text, max_chars=chars_per_chunk)
    if not chunks:
        return [text[i : i + chars_per_chunk] for i in range(0, len(text), chars_per_chunk)]
    return chunks


@router.get("/api/languages")
async def get_languages():
    """Return supported source and target languages."""
    langs = get_supported_languages(settings.llm_model_translation)
    return {
        "source": langs,
        "target": langs,
    }


@router.get("/status")
async def status_check(session: AsyncSession = Depends(get_session)):
    """Report health of database, Whisper, and Ollama services."""
    db_status = "ok"
    try:
        await session.execute(text("select 1"))
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        db_status = "error"

    whisper_status = "ok"
    ollama_status = "ok"
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            await client.get(f"{settings.whisper_base_url.rstrip('/')}/health")
        except Exception as e:
            logger.warning("Whisper service unreachable: %s", e)
            whisper_status = "unreachable"
        try:
            await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
        except Exception as e:
            logger.warning("Ollama service unreachable: %s", e)
            ollama_status = "unreachable"

    return {
        "database": db_status,
        "whisper": whisper_status,
        "ollama": ollama_status,
    }


@router.get("/auth/login")
async def start_login():
    """Initiate the OIDC login flow or return a dev token when in dev mode."""
    state = secrets.token_urlsafe(16)
    
    query = urllib.parse.urlencode(
        {
            "client_id": settings.oidc_client_id,
            "redirect_uri": settings.oidc_redirect_uri,
            "response_type": "code",
            "scope": settings.oidc_scope,
            "state": state,
        }
    )
    if settings.dev_mode:
        # Returns a valid token immediately without OIDC
        payload = {
            "sub": "dev-user",
            "email": "dev@local",
            "name": "Dev User",
            "iat": dt.datetime.now(dt.timezone.utc),
        }
        token = jwt.encode(
            payload,
            settings.oidc_client_secret,
            algorithm="HS256",
        )
        html = f"""
        <html><body><script>
        localStorage.setItem('access_token', {json.dumps(token)});
        localStorage.setItem('id_token', {json.dumps(token)});
        localStorage.setItem('user_name', "Dev User");
        window.location.href = '/';
        </script></body></html>
        """
        return HTMLResponse(content=html)

    auth_endpoint = get_authorization_endpoint()
    auth_url = f"{auth_endpoint}?{query}"
    
    redirect_response = RedirectResponse(url=auth_url)
    redirect_response.set_cookie(
        "oidc_state",
        state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=settings.use_secure_cookies,
    )
    return redirect_response


@router.get("/auth/callback")
async def auth_callback(request: Request):
    """Handle OIDC callback, storing tokens in local storage."""
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    state_cookie = request.cookies.get("oidc_state")
    
    if not code or not state or state != state_cookie:
        logger.error(
            "OIDC Callback Error: code=%s, state_param=%s, state_cookie=%s, cookies_keys=%s",
            code, state, state_cookie, list(request.cookies.keys())
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid state or code")

    token_response = await exchange_code_for_token(code)
    access_token = token_response.get("access_token")
    id_token = token_response.get("id_token")
    if not access_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No access token returned")

    user = decode_token(id_token or access_token)
    name = user.name if user else ""

    # Prefer id_token for stateless JWT authentication, as access_token might be opaque
    final_token = id_token if id_token else access_token
    logger.info("Final token: %s", final_token)
    html = f"""
    <html><body><script>
    const token = {json.dumps(final_token)};
    localStorage.setItem('access_token', token);
    localStorage.setItem('id_token', token);
    localStorage.setItem('user_name', {json.dumps(name)});
    window.location.href = '/';
    </script></body></html>
    """
    response = HTMLResponse(content=html)
    # Clear the state cookie after successful authentication
    response.delete_cookie("oidc_state", samesite="lax")
    return response


@router.get("/auth/verify")
async def verify_auth(user: UserInfo = Depends(get_current_user)):
    """Verify checks if the current token is valid."""
    return {"status": "authenticated", "user": user.sub}


@router.get("/api/conversations", response_model=list[ConversationListItem])
async def list_conversation_endpoint(
    session: AsyncSession = Depends(get_session), 
    user: UserInfo = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List conversations for the authenticated user."""
    db_user = await get_or_create_user(session, user)
    conversations = await list_conversations(session, db_user, limit=limit, offset=offset)
    return conversations


@router.post("/api/conversations", response_model=ConversationOut)
async def create_conversation_endpoint(
    payload: ConversationCreate,
    session: AsyncSession = Depends(get_session),
    user: UserInfo = Depends(get_current_user),
):
    """Create a new conversation with initial metadata."""
    db_user = await get_or_create_user(session, user)
    
    # Normalize languages
    if isinstance(payload.source_language, str):
        payload.source_language = get_language_code(payload.source_language)
    elif isinstance(payload.source_language, list):
        payload.source_language = [get_language_code(l) for l in payload.source_language]
    
    payload.target_language = get_language_name(payload.target_language)

    conversation = await create_conversation(session, db_user, payload)
    await session.commit()
    await session.commit()
    # Re-fetch to load relationships (paragraphs) to avoid MissingGreenlet on lazy load
    conversation = await get_conversation(session, db_user, conversation.id)
    return conversation


@router.get("/api/conversations/{conversation_id}", response_model=ConversationOut)
async def get_conversation_endpoint(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    user: UserInfo = Depends(get_current_user),
):
    """Fetch a conversation and its paragraphs for the user."""
    db_user = await get_or_create_user(session, user)
    conversation = await get_conversation(session, db_user, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    await session.refresh(conversation)
    # log conversation paragraphs as plain text
    logger.info("Conversation paragraphs: %r", [p.source_text for p in conversation.paragraphs])
    return conversation


@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    user: UserInfo = Depends(get_current_user),
):
    """Delete a single conversation."""
    db_user = await get_or_create_user(session, user)
    deleted = await delete_conversation(session, db_user, conversation_id)
    await session.commit()
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"deleted": True}


@router.delete("/api/conversations")
async def delete_all_conversations_endpoint(
    session: AsyncSession = Depends(get_session),
    user: UserInfo = Depends(get_current_user),
):
    """Delete all conversations for the authenticated user."""
    db_user = await get_or_create_user(session, user)
    await delete_all_conversations(session, db_user)
    await session.commit()
    return {"deleted": True}


@router.patch("/api/conversations/{conversation_id}/title", response_model=ConversationOut)
async def update_conversation_title_endpoint(
    conversation_id: uuid.UUID,
    payload: ConversationTitleUpdate,
    session: AsyncSession = Depends(get_session),
    user: UserInfo = Depends(get_current_user),
):
    """Update a conversation title and mark it as user-provided."""
    db_user = await get_or_create_user(session, user)
    conversation = await get_conversation(session, db_user, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    
    conversation.title = payload.title
    conversation.is_title_manual = True
    await session.commit()
    await session.refresh(conversation)
    return conversation


@router.post("/api/conversations/{conversation_id}/pending-deletion")
async def mark_pending_deletion_endpoint(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    user: UserInfo = Depends(get_current_user),
):
    """Mark a conversation as pending deletion."""
    db_user = await get_or_create_user(session, user)
    marked = await mark_conversation_pending_deletion(session, db_user, conversation_id)
    await session.commit()
    if not marked:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"pending_deletion": True}


@router.delete("/api/conversations/{conversation_id}/pending-deletion")
async def restore_conversation_endpoint(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    user: UserInfo = Depends(get_current_user),
):
    """Restore a conversation from pending deletion."""
    db_user = await get_or_create_user(session, user)
    restored = await restore_conversation(session, db_user, conversation_id)
    await session.commit()
    if not restored:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"pending_deletion": False}



# Global registry of active rebuild tasks
active_rebuild_tasks: dict[uuid.UUID, asyncio.Task] = {}

async def rebuild_conversation_task(conversation_id: uuid.UUID):
    """Background task to rebuild translation and summary, broadcasting updates."""
    logger.info("Starting background rebuild for conversation %s", conversation_id)
    try:
        async with SessionLocal() as session:
            conversation = await get_conversation(session, None, conversation_id) # User implicitly authorized by endpoint check
            if conversation is None:
                 logger.error("Rebuild task: Conversation %s not found", conversation_id)
                 return

            paragraphs = sorted(conversation.paragraphs, key=_safe_paragraph_index)
            live_paragraphs = [p for p in paragraphs if p.type not in ("refined", "summary")]

            full_source = " ".join(p.source_text for p in live_paragraphs if p.source_text).strip()
            if not full_source:
                 logger.warning("Rebuild task: No source text for conversation %s", conversation_id)
                 return

            max_chunk_chars = estimate_max_chars(REBUILD_CONTEXT_TOKENS, safe_mode=True)
            chunks = chunk_text_by_sentence(full_source, max_chars=max_chunk_chars) or [full_source]

            all_langs = []
            for p in live_paragraphs:
                if p.detected_language:
                    for lang in p.detected_language.split(","):
                        lang = lang.strip()
                        if lang and lang not in all_langs:
                            all_langs.append(lang)

            combined_languages = ",".join(all_langs)
            if len(combined_languages) > 50:
                combined_languages = combined_languages[:50].rsplit(",", 1)[0]

            next_index = paragraphs[-1].paragraph_index + 1 if paragraphs else 0
            combined_paragraph = await add_paragraph(
                session,
                conversation,
                next_index,
                "",
                "",
                paragraph_type="active",
                detected_language=combined_languages or None,
            )
            await session.commit()
            await session.refresh(combined_paragraph)
            
            source_language_name = pick_language_name(conversation.source_language, combined_languages)
            aggregated_source = ""
            aggregated_translation = ""

            # Check cancellation before costly operations
            try:
                for chunk in chunks:
                    cleaned = await cleanup_text(chunk, num_ctx=REBUILD_CONTEXT_TOKENS)
                    if not cleaned:
                        cleaned = chunk
                        
                    target_lang_name = get_language_name(conversation.target_language)
                    source_code = get_language_code(source_language_name)
                    target_code = get_language_code(target_lang_name)
                    
                    if source_code != "auto" and source_code == target_code:
                        translated = cleaned
                    else:
                        # Throttling to prevent starvation of other tasks (e.g. warm-up)
                        await translation_limiter.acquire()
                        translated = await translate_text(
                            cleaned,
                            target_lang_name,
                            source_language_name=source_language_name,
                            num_ctx=REBUILD_CONTEXT_TOKENS,
                            timeout=60.0,
                        )
                        if not translated:
                            translated = ""

                    aggregated_source = " ".join(filter(None, [aggregated_source, cleaned])).strip()
                    aggregated_translation = " ".join(filter(None, [aggregated_translation, translated])).strip()
                    
                    await update_paragraph_content(
                        session,
                        combined_paragraph.id,
                        cleaned,
                        translated,
                    )
                    await session.commit()

                    # Broadcast Chunk
                    # Use standard keys 'source' and 'translation' to match frontend expectation
                    await manager.broadcast(conversation_id, {
                        "type": "chunk", # Custom type for rebuild progress? Or mimic 'active' params?
                        # Frontend likely renders based on 'source' and 'translation' presence
                        "paragraph_id": str(combined_paragraph.id),
                        "paragraph_index": combined_paragraph.paragraph_index,
                        "source": aggregated_source,       # FIXED: source_text -> source
                        "translation": aggregated_translation, # FIXED: translated_text -> translation
                        "detected_language": combined_languages,
                        "is_final": False,
                    })

                async def broadcast_status(progress: int):
                    status_text = f"Summarizing: {progress}%"
                    # For status updates, we temporarily show status in source field? 
                    # Or maybe just send a distinct status message.
                    # Previous code updated 'status_text' into the DB as source?
                    # "source_text": status_text in previous code.
                    
                    await update_paragraph_content(
                        session,
                        combined_paragraph.id,
                        status_text,
                        aggregated_translation,
                        detected_language=combined_languages or None,
                        paragraph_type="active",
                    )
                    await session.commit()
                    
                    await manager.broadcast(conversation_id, {
                        "type": "status",
                        "paragraph_id": str(combined_paragraph.id),
                        "paragraph_index": combined_paragraph.paragraph_index,
                        "source": status_text,       # FIXED
                        "translation": aggregated_translation, # FIXED
                    })

                # Recursively summarize (simplified for background task structure, skipping generator yield complexity for broadcast)
                # Actually, we can reuse the logic but just await broadcast calls instead of yielding
                
                async def run_summerization(text_in, label_in):
                    if not text_in.strip():
                        return ""
                    
                    if estimate_token_count(text_in) <= MAX_SUMMARY_CONTEXT_TOKENS:
                        await translation_limiter.acquire()
                        target_lang_name = get_language_name(conversation.target_language)
                        return await summarize_text(text_in, label_in, target_lang_name, num_ctx=REBUILD_CONTEXT_TOKENS) or ""

                    chunks_sum = split_text_for_summary(text_in, MAX_SUMMARY_CONTEXT_TOKENS)
                    partial_summaries = []
                    total = len(chunks_sum)
                    
                    for i, c in enumerate(chunks_sum, 1):
                       p_percent = int(((i-1)/max(total, 1))*100)
                       if label_in == "Translated text": # Only show top level progress
                           await broadcast_status(p_percent)
                       
                       partial = await run_summerization(c, label_in)
                       if partial:
                           partial_summaries.append(partial)
                            
                       if label_in == "Translated text":
                           await broadcast_status(int((i/max(total, 1))*100))

                    combined = " ".join(filter(None, partial_summaries)).strip()
                    return await run_summerization(combined, f"Summary of {label_in}")

                translation_summary = await run_summerization(aggregated_translation or aggregated_source, "Translated text")

                summary_paragraph = await add_paragraph(
                    session,
                    conversation,
                    next_index + 1,
                    "",
                    translation_summary,
                    paragraph_type="summary",
                )

                await update_paragraph_content(
                    session,
                    combined_paragraph.id,
                    None,
                    None,
                    detected_language=combined_languages or None,
                    paragraph_type="refined",
                )

                is_default_title = conversation.title and conversation.title.startswith("New Session")
                if (
                    (conversation.title is None or is_default_title)
                    and not conversation.is_title_manual
                    and aggregated_translation
                ):
                    target_lang_name = get_language_name(conversation.target_language)
                    await translation_limiter.acquire()
                    generated_title = await generate_title(aggregated_translation, target_lang_name)
                    if generated_title:
                        conversation.title = generated_title

                await session.commit()
                
                # Broadcast Final
                # For final message, we want to replace the 'chunk' progress with the refined paragraph
                await manager.broadcast(conversation_id, {
                    "type": "final",
                    "dataset_updates": [ # Frontend might not support this key? 
                        # Let's send individual paragraph updates to be safe, or check how client handles 'final'
                        # Standard stream doesn't have 'final' type.
                        # We might need to send a 'stable' update for the combined paragraph?
                        {
                            "paragraph_id": str(combined_paragraph.id),
                            "paragraph_index": combined_paragraph.paragraph_index,
                            "source": combined_paragraph.source_text,
                            "translation": combined_paragraph.translated_text,
                            "type": "refined"
                        },
                        {
                            "paragraph_id": str(summary_paragraph.id),
                            "paragraph_index": summary_paragraph.paragraph_index,
                            "source": summary_paragraph.source_text,
                            "translation": summary_paragraph.translated_text,
                            "type": "summary"
                        }
                    ],
                    # Fallback/Direct fields for simple clients:
                    "paragraph_id": str(combined_paragraph.id),
                    "source": combined_paragraph.source_text,
                    "translation": combined_paragraph.translated_text,
                    "title": conversation.title,
                })
                logger.info("Background rebuild completed for conversation %s", conversation_id)

            except asyncio.CancelledError:
                 logger.warning("Rebuild task cancelled for conversation %s", conversation_id)
                 # Rollback changes if cancelled mid-flight? Or commit partial? 
                 # Safer to rollback active editing on 'combined_paragraph' if it wasn't finalized.
                 # But we might have committed intermediate chunks. 
                 # In this design we commit per chunk.
                 # Let's clean up the combined paragraph if it wasn't marked stable/refined?
                 # Actually, cancellation usually implies new input, so we probably want to discard the 'rebuild' attempt entirely
                 # to avoid race conditions with the stream.
                 # We can try to delete the combined paragraph.
                 await session.delete(combined_paragraph)
                 await session.commit()
                 raise

    except asyncio.CancelledError:
        logger.info("Rebuild task properly cancelled clean-up")
        
    except Exception as exc:
        logger.exception("Failed background rebuild task: %s", exc)
        await manager.broadcast(conversation_id, {"type": "error", "message": "Failed to rebuild translation"})
    
    finally:
        # Cleanup from active tasks registry
        if conversation_id in active_rebuild_tasks:
            del active_rebuild_tasks[conversation_id]


@router.post("/api/conversations/{conversation_id}/rebuild")
async def rebuild_translation_endpoint(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    user: UserInfo = Depends(get_current_user),
):
    """Rebuild translation and summary content from existing conversation text."""
    db_user = await get_or_create_user(session, user)
    conversation = await get_conversation(session, db_user, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    # Deduplication
    if conversation_id in active_rebuild_tasks:
        existing_task = active_rebuild_tasks[conversation_id]
        if not existing_task.done():
            return {"status": "accepted", "message": "Rebuild task already running."}
            
    # Start background task
    task = asyncio.create_task(rebuild_conversation_task(conversation_id))
    active_rebuild_tasks[conversation_id] = task
    
    return {"status": "accepted", "message": "Rebuild process started in background."}



async def _gather_audio(
    websocket: WebSocket,
    conversation_id: str | None = None,
    recording_format: str | None = None,
) -> AsyncIterator[bytes]:
    """Yield binary audio chunks or streaming config updates from the WebSocket."""
    def _resolve_audio_extension(format_hint: str | None) -> str:
        lowered = (format_hint or "").lower()
        if lowered in {"mp4", "m4a"}:
            return "mp4"
        if lowered in {"pcm", "pcm16", "pcm16le", "pcm_s16le"}:
            return "pcm"
        return "webm"

    recording_extension = _resolve_audio_extension(recording_format)

    def _open_new_log_file():
        if not settings.dev_mode:
            return None
        try:
             # extract path from settings.log_file if present
            if settings.log_file:
                log_dir = os.path.dirname(settings.log_file)
            else:
                log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            timestamp = int(dt.datetime.now().timestamp())
            # Use microsecond to avoid collision if fast restart
            micros = dt.datetime.now().microsecond
            filename = f"{log_dir}/audio_{conversation_id or 'unknown'}_{timestamp}_{micros}.{recording_extension}"
            logger.info(f"Dev mode enabled. Saving audio to {filename}")
            return open(filename, "wb")
        except Exception as e:
            logger.error(f"Failed to open audio log file: {e}")
            return None

    def _close_log_file(file_handle):
        if file_handle:
            try:
                file_handle.close()
                return None
            except Exception as e:
                logger.error(f"Failed to close audio log file: {e}")
                return file_handle

    file_handle = None
    # Don't open immediately to avoid empty file if first chunk is header (which triggers rotation)

    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data and data["bytes"]:
                chunk = data["bytes"]
                
                # Check for WebM header signature 
                if len(chunk) >= 4 and chunk.startswith(b'\x1a\x45\xdf\xa3'):
                    # Rotate file
                    file_handle = _close_log_file(file_handle)
                    file_handle = _open_new_log_file()
                elif file_handle is None:
                    # Lazy open if we receive data but have no handle (e.g. stream start without header, or just first chunk)
                    file_handle = _open_new_log_file()

                if file_handle:
                    try:
                        file_handle.write(chunk)
                        file_handle.flush()
                    except Exception as e:
                        logger.error(f"Failed to write audio chunk: {e}")
                yield chunk
            elif "text" in data and data["text"]:
                # Check if this is a config update (JSON)
                try:
                    parsed = json.loads(data["text"])
                    
                    if parsed.get("type") == "client_log":
                        logger.info(f"CLIENT_LOG: {parsed.get('message')}")
                        continue

                    # If stop_recording, close the file so next start gets a fresh one? 
                    # Actually, next start will have a header, so rotation handles it. 
                    # But we can close early if we want. 
                    # Let's rely on header detection for robustness, OR explicit stop.
                    if parsed.get("type") == "stop_recording":
                        file_handle = _close_log_file(file_handle)

                    if parsed.get("recording_format"):
                        recording_extension = _resolve_audio_extension(parsed.get("recording_format"))

                    if "source_language" in parsed or "target_language" in parsed or "recording_format" in parsed or "type" in parsed:
                        yield parsed
                        continue
                except json.JSONDecodeError as e:
                    logger.exception("Failed to decode JSON: %s", e)
                    pass
            else:
                return
    finally:
        file_handle = _close_log_file(file_handle)

def _merge_source_with_overlap(existing: str | None, incoming: str | None) -> str:
    """Merge incoming stable text into the existing paragraph, removing overlap."""
    incoming_clean = str(incoming or "").strip()
    existing_clean = str(existing or "").strip()

    if not existing_clean:
        return incoming_clean
    if not incoming_clean:
        return existing_clean

    if incoming_clean.startswith(existing_clean):
        return incoming_clean
    if existing_clean.endswith(incoming_clean):
        return existing_clean

    overlap = 0
    max_overlap = min(len(existing_clean), len(incoming_clean))
    for i in range(1, max_overlap + 1):
        if existing_clean.endswith(incoming_clean[:i]):
            overlap = i

    if overlap:
        return f"{existing_clean}{incoming_clean[overlap:]}"
    return f"{existing_clean} {incoming_clean}"


class TranslationWorker:
    """
    Background worker that consumes translation tasks from a queue,
    coalesces requests for the same paragraph, and processes them.
    """
    def __init__(self, queue: asyncio.Queue, websocket: WebSocket, conversation_id: uuid.UUID, translation_cache: dict[uuid.UUID, str], unstable_cache: dict[uuid.UUID, str], llm_ready_event: asyncio.Event):
        self.queue = queue
        self.websocket = websocket
        self.conversation_id = conversation_id
        self.translation_cache = translation_cache
        self.unstable_cache = unstable_cache
        self.llm_ready_event = llm_ready_event
        self.running = True

    async def run(self):
        """Consume translation tasks from the queue with coalescing."""
        # Wait for LLM to be ready before processing any translations
        await self.llm_ready_event.wait()

        pending_item = None
        
        while self.running:
            if pending_item:
                item = pending_item
                pending_item = None
            else:
                item = await self.queue.get()
            
            if item is None:
                self.queue.task_done()
                break

            logger.info("[CONVO_FLOW] Worker popped item: %s", item)

            # Start batch with the first item
            # Item: (p_id, text_segment, tgt_lang, ctx_text, p_idx, p_lang, p_type, src_lang_name)
            p_id, combined_text, tgt_lang, ctx_text, p_idx, p_lang, p_type, src_lang_name = item
            
            should_stop = False
            
            # Opportunistically grab more items if they match the current paragraph
            # We need to drain rapidly to throttle effectively
            while True:
                try:
                    # Non-blocking peek-and-consume
                    if self.queue.empty():
                        break
                        
                    next_item = self.queue.get_nowait()
                    if next_item is None:
                        # Shutdown signal found.
                        self.queue.task_done()
                        pending_item = None # Stop outer loop
                        # Process current batch then exit
                        should_stop = True
                        break
                    
                    next_pid = next_item[0]
                    next_tgt = next_item[2]
                    next_src_lang = next_item[7] if len(next_item) > 7 else None
                    
                    if next_pid == p_id and next_tgt == tgt_lang:
                        # Coalesce by taking the latest snapshot of the paragraph text.
                        combined_text = next_item[1]
                        ctx_text = next_item[3]
                        tgt_lang = next_item[2]
                        p_lang = next_item[5]
                        p_type = next_item[6]
                        if next_src_lang:
                            src_lang_name = next_src_lang

                        self.queue.task_done()
                    else:
                        # Mismatch. This belongs to next batch.
                        pending_item = next_item
                        break
                except asyncio.QueueEmpty:
                    break
            
            logger.info(
                "[CONVO_FLOW] Coalesced batch for p_id=%s: combined_text=\"%s\", text_len=%d, target=%s",
                p_id, combined_text, len(combined_text), tgt_lang
            )
            
            # Throttling
            await translation_limiter.acquire()
            
            try:
                # Perform Translation
                # Clean up spacing in combined text
                clean_source_segment = combined_text.strip()
                if not clean_source_segment:
                    if should_stop: break
                    continue

                logger.info(
                    "[CONVO_FLOW] Translating segment for p_id=%s. Text: \"%s\", Context: \"%s\", Target: %s...",
                    p_id, clean_source_segment, ctx_text, tgt_lang
                )
                resolved_source_lang = src_lang_name or "Auto"
                translation = await translate_text(
                    clean_source_segment,
                    tgt_lang,
                    context_text=ctx_text,
                    source_language_name=resolved_source_lang,
                    num_ctx=4096,
                )
                if not translation:
                    # translation failed, skip this paragraph
                    continue
                
                logger.info(
                     "[CONVO_FLOW] Translation received for paragraph %s: \"%s\"",
                     p_id, translation
                )

                async with SessionLocal() as session:
                    # Update DB without overriding paragraph type; stabilize logic updates type earlier in the pipeline.
                    updated_p = await update_paragraph_content(
                        session,
                        p_id,
                        source_append=None,
                        translation_append=translation,
                        detected_language=p_lang,
                        paragraph_type=None,
                        translation_override=None,
                    )
                    await session.commit()
                    
                    # Update cache for main thread
                    if updated_p.translated_text:
                        self.translation_cache[p_id] = updated_p.translated_text

                    logger.info("[CONVO_FLOW] Updated paragraph %s in DB. Append translation: \"%s\"", p_id, translation)
                    
                    # Title Generation Check
                    # Reload conversation to see full context
                    result = await session.execute(
                        select(models.Conversation)
                        .where(models.Conversation.id == self.conversation_id)
                        .options(selectinload(models.Conversation.paragraphs))
                    )
                    fresh_conv = result.scalar_one_or_none()
                    
                    if fresh_conv:
                        all_text = " ".join([
                            p.translated_text for p in fresh_conv.paragraphs if p.translated_text
                        ])
                        total_words = len(all_text.split())
                        
                        current_title = fresh_conv.title
                        is_default_title = current_title and current_title.startswith("New Session")
                        is_manual_title = fresh_conv.is_title_manual

                        if (current_title is None or is_default_title) and not is_manual_title and total_words >= 30:
                            context_for_title = " ".join(all_text.split()[:100])
                            new_title = await generate_title(context_for_title, tgt_lang)
                            if new_title:
                                fresh_conv.title = new_title
                                await session.commit()
                                try:
                                    await self.websocket.send_json({
                                        "conversation_id": str(fresh_conv.id),
                                        "title": new_title
                                    })
                                except Exception:
                                    pass
                    
                    # Notify WebSocket
                    try:
                        logger.info("[CONVO_FLOW] Sending translation update to WS for p_id=%s: source =%s, translation=%s", updated_p.id, updated_p.source_text, updated_p.translated_text)
                        await self.websocket.send_json({
                            "conversation_id": str(self.conversation_id),
                            "paragraph_id": str(updated_p.id),
                            "paragraph_index": p_idx,
                            "source": updated_p.source_text,
                            "unstable_text": self.unstable_cache.get(updated_p.id, ""),
                            "translation": updated_p.translated_text,
                            "detected_language": updated_p.detected_language,
                            "is_final": True,
                            "type": updated_p.type,
                            "translation_pending": False,
                        })
                    except Exception:
                        pass
                
            except Exception as e:
                logger.error("Background translation failed: %r", e)
            finally:
                # For the initial item
                self.queue.task_done()
            
            if should_stop:
                break


@router.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    conversation_id: str | None = None,
):
    """Stream transcription and translation updates over WebSocket."""
    user = await get_current_user_websocket(websocket)
    if user is None:
        logger.error("User not found")
        return

    await websocket.accept()
    try:
        init = await websocket.receive_json()
        # Initial stream config provided by the client
        source_language = init.get("source_language", "en")
        target_language = init.get("target_language", "en")
        recording_format = init.get("recording_format")
        recording_mime_type = init.get("recording_mime_type")
        recording_sample_rate = init.get("recording_sample_rate")
        recording_channels = init.get("recording_channels")
        
        # Normalize languages
        if isinstance(source_language, list):
            source_language = [get_language_code(l) for l in source_language]
        else:
            source_language = get_language_code(source_language)
            
        target_language = get_language_name(target_language)
        
        title = init.get("title")
        logger.info(
            "[CONVO_FLOW] WebSocket stream initialized: conversation_id=%s, src=%s, tgt=%s, format=%s",
            conversation_id, source_language, target_language, recording_format
        )
    except WebSocketDisconnect:
        logger.info("Client disconnected before sending initialization.")
        return
    except Exception as e:
        logger.error("Failed to receive WebSocket initialization: %s", e)
        try:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        except Exception:
            pass
        return

    # Background Translation Worker
    translation_queue = asyncio.Queue()
    translation_cache: dict[uuid.UUID, str] = {}
    unstable_cache: dict[uuid.UUID, str] = {}


    async with SessionLocal() as session:
        db_user = await get_or_create_user(session, user)
        conversation = None
        manager_connected = False
        
        if conversation_id:
            try:
                c_uuid = uuid.UUID(conversation_id)
                conversation = await get_conversation(session, db_user, c_uuid)
                if conversation:
                    logger.info("Resuming conversation %s", conversation_id)
                    if target_language and conversation.target_language != target_language:
                         logger.info("Updating conversation target language to %s", target_language)
                         conversation.target_language = target_language
                         await session.commit()
            except ValueError:
                logger.warning("Invalid conversation_id format: %s", conversation_id)
        

        if conversation and conversation.id:
            await manager.connect(conversation.id, websocket)
            manager_connected = True
            
            # Notify if rebuild is in progress
            if conversation.id in active_rebuild_tasks:
                 task = active_rebuild_tasks[conversation.id]
                 if not task.done():
                     try:
                         await websocket.send_json({
                             "type": "status",
                             "source_text": "Rebuilding translation...",
                             "translated_text": "",
                             "paragraph_id": None,
                         })
                     except Exception:
                         pass


        if not conversation:
            if not title:
                now = dt.datetime.now()
                title = f"New Session {now.strftime('%b %d, %H:%M')}"

            # Clear any pending deletions before creating new conversation
            await delete_pending_conversations(session, db_user)

            conversation = await create_conversation(
                session,
                db_user,
                ConversationCreate(title=title, source_language=source_language, target_language=target_language),
            )
            await session.commit()
            await session.refresh(conversation, attribute_names=["paragraphs"])
            if not manager_connected:
                # Register new sessions immediately so rebuild progress streams to the active client.
                await manager.connect(conversation.id, websocket)
                manager_connected = True

        if not hasattr(conversation, "is_title_manual"):
            conversation.is_title_manual = False
        if getattr(conversation, "paragraphs", None) is None:
            conversation.paragraphs = []

        if isinstance(source_language, list):
            if len(source_language) == 1:
                whisper_lang = source_language[0]
            elif "auto" in source_language:
                 whisper_lang = "auto"
            else:
                whisper_lang = "auto"
        else:
            whisper_lang = source_language

        warmup_task: asyncio.Task | None = None
        whisper_ready_event = asyncio.Event()
        llm_ready_event = asyncio.Event()
        
        # Start warm-up in background
        warmup_task = asyncio.create_task(run_warm_up(websocket, whisper_lang, whisper_ready_event, llm_ready_event))

        try:
            # Provide initial state to client, including languages so selectors update
            await websocket.send_json({
                "conversation_id": str(conversation.id),
                "title": conversation.title,
                "source_language": conversation.source_language,
                "target_language": conversation.target_language,
            })
        except Exception:
            pass
        
        # Start Worker
        worker = TranslationWorker(translation_queue, websocket, conversation.id, translation_cache, unstable_cache, llm_ready_event)
        worker_task = asyncio.create_task(worker.run())

        audio_iter = _gather_audio(websocket, str(conversation.id), recording_format=recording_format)
        transcript_iter = stream_transcription(
            audio_iter,
            language_code=whisper_lang,
            model_ready_event=whisper_ready_event,
            recording_format=recording_format,
            recording_mime_type=recording_mime_type,
            recording_sample_rate=recording_sample_rate,
            recording_channels=recording_channels,
        )
        
        current_paragraph_index = 0
        paragraph_sentence_count = 0
        
        if not conversation.paragraphs:
            current_paragraph = await add_paragraph(session, conversation, 0, "", "")
            await session.commit()
        else:
            sorted_paragraphs = sorted(conversation.paragraphs, key=_safe_paragraph_index)
            last_p = sorted_paragraphs[-1]
            
            if last_p.type == "active":
                current_paragraph = last_p
                current_paragraph_index = last_p.paragraph_index
                paragraph_sentence_count = 0 
                logger.info("Resuming active paragraph %s", current_paragraph.id)
            else:
                current_paragraph_index = last_p.paragraph_index + 1
                current_paragraph = await add_paragraph(session, conversation, current_paragraph_index, "", "")
                await session.commit()
                logger.info("Starting new paragraph %s (last was %s)", current_paragraph.id, last_p.type)
        
        last_paragraph_source = ""

        try:
            async for chunk in transcript_iter:
                logger.info("[CONVO_FLOW] Audio chunk received: %s", chunk)
                if "target_language" in chunk:
                    target_language = get_language_name(chunk["target_language"])
                    logger.info("Updating target language to: %s", target_language)
                    if conversation and conversation.target_language != target_language:
                        conversation.target_language = target_language
                        await session.commit()
                    continue

                if "source_language" in chunk:
                    sl = chunk["source_language"]
                    logger.info("Updating source language to: %s", sl)
                    if conversation:
                        if isinstance(sl, list):
                            conversation.source_language = [get_language_code(l) for l in sl]
                        else:
                            conversation.source_language = get_language_code(sl)
                        await session.commit()
                    continue
                
                if "status" in chunk and chunk["status"] == "processing_complete":
                     logger.info("Processing complete. Notifying client.")
                     logger.info("[CONVO_FLOW] Processing complete signal received.")
                     if current_paragraph:
                         # Capture last source for context
                         last_paragraph_source = current_paragraph.source_text
                         
                         if current_paragraph.type == "active":
                             logger.info("Stabilizing active paragraph %s on stop.", current_paragraph.id)
                             logger.info("[CONVO_FLOW] Stabilizing active paragraph %s on stop.", current_paragraph.id)
                             current_paragraph = await update_paragraph_content(
                                session,
                                current_paragraph.id,
                                source_append=None,
                                translation_append=None,
                                paragraph_type="stable",
                             )
                             await session.commit()
                             try:
                                await websocket.send_json({
                                    "conversation_id": str(conversation.id),
                                    "paragraph_id": str(current_paragraph.id),
                                    "paragraph_index": current_paragraph.paragraph_index,
                                    "source": current_paragraph.source_text,
                                    "unstable_text": "",
                                    "translation": current_paragraph.translated_text,
                                    "detected_language": current_paragraph.detected_language,
                                    "is_final": True,
                                    "type": "stable",
                                })
                             except Exception:
                                pass
                     
                     # Reset current paragraph so next text triggers a new one
                     current_paragraph = None

                     try:
                        await websocket.send_json({"status": "processing_complete"})
                     except Exception:
                        pass
                     continue

                text = chunk.get("text")
                unstable_text = chunk.get("unstable_text")
                is_final = chunk.get("is_final")
                
                if not text and not unstable_text:
                    continue

                # Should create new paragraph if we don't have one (e.g. after stop/processing_complete)
                if current_paragraph is None:
                    current_paragraph_index += 1
                    paragraph_sentence_count = 0
                    current_paragraph = await add_paragraph(
                        session, conversation, current_paragraph_index, "", ""
                    )
                    await session.commit()
                    await session.commit()
                    logger.info("[CONVO_FLOW] Created new paragraph %s (resumed session)", current_paragraph.id)
                    
                    # Cancel any running rebuild task since we have new input
                    if conversation.id in active_rebuild_tasks:
                        task = active_rebuild_tasks[conversation.id]
                        if not task.done():
                            logger.info("Cancelling active rebuild task for conversation %s due to new input.", conversation.id)
                            task.cancel()

                raw_source = getattr(current_paragraph, "source_text", "") or ""
                raw_trans = getattr(current_paragraph, "translated_text", "") or ""

                base_source = raw_source if isinstance(raw_source, str) else ""
                base_trans = raw_trans if isinstance(raw_trans, str) else ""

                # Check shared cache for background updates
                if current_paragraph.id in translation_cache:
                    cached_trans = translation_cache[current_paragraph.id]
                    # Update local reference to avoid reverting
                    if cached_trans:
                         current_paragraph.translated_text = cached_trans
                chunk_lang = chunk.get("language")
                if chunk_lang and chunk_lang != "auto":
                    existing_langs = (current_paragraph.detected_language or "").split(",")
                    existing_langs = [l.strip() for l in existing_langs if l.strip()]
                    
                    if chunk_lang not in existing_langs:
                        existing_langs.append(chunk_lang)
                        new_lang_str = ",".join(existing_langs)
                        if len(new_lang_str) <= 50:
                           current_paragraph.detected_language = new_lang_str
                        else:
                            pass              

                if not is_final:
                    unstable_cache[current_paragraph.id] = unstable_text

                    prefix = " " if base_source and not base_source.endswith(" ") else ""
                    
                    target_code = get_language_code(target_language)
                    effective_lang = chunk_lang or whisper_lang
                    
                    should_immediate_assign = False
                    start_reverse_trans = False
                    reverse_target_name = None

                    if effective_lang != "auto" and effective_lang == target_code:
                        reverse_target_name = get_prevalent_other_language(conversation, target_language)
                        if reverse_target_name and whisper_lang == "auto":
                             start_reverse_trans = True
                             logger.info("Auto-switching reverse translation target to: %s", reverse_target_name)
                        else:
                             should_immediate_assign = True
                    
                    display_trans = base_trans
                    display_source = base_source

                    if text:
                        if should_immediate_assign:
                            display_trans += prefix + text
                        display_source += prefix + text
                    
                    logger.info("[CONVO_FLOW] Sending preview for p_id=%s: source=%s, translation=%s, unstable=%s", current_paragraph.id, display_source, display_trans, unstable_text)
                    await websocket.send_json({
                        "conversation_id": str(conversation.id),
                        "paragraph_id": str(current_paragraph.id),
                        "paragraph_index": current_paragraph_index,
                        "source": display_source,
                        "unstable_text": unstable_text,
                        "translation": display_trans,
                        "detected_language": current_paragraph.detected_language,
                        "is_final": False,
                        "type": current_paragraph.type,
                    })
                else:
                    unstable_cache[current_paragraph.id] = ""

                    context_text = current_paragraph.source_text or ""
                    if not context_text:
                         context_text = last_paragraph_source
                    
                    target_code = get_language_code(target_language)
                    effective_lang = chunk_lang or whisper_lang
                    
                    should_immediate_assign = False
                    start_reverse_trans = False
                    reverse_target_name = None

                    if effective_lang != "auto" and effective_lang == target_code:
                        reverse_target_name = get_prevalent_other_language(conversation, target_language)
                        if reverse_target_name and whisper_lang == "auto":
                             start_reverse_trans = True
                             logger.info("Auto-switching reverse translation target to: %s", reverse_target_name)
                        else:
                             should_immediate_assign = True
                    
                    if should_immediate_assign:
                         # Skip translation when detected/source language matches target language
                         logger.info("[CONVO_FLOW] Immediate assignment for p_id=%s: text=%s", current_paragraph.id, text)
                         translation = text
                         current_paragraph = await update_paragraph_content(
                             session, current_paragraph.id, text, translation, 
                             detected_language=current_paragraph.detected_language,
                             paragraph_type="active"
                         )
                         await session.commit()
                         
                         await websocket.send_json({
                            "conversation_id": str(conversation.id),
                            "paragraph_id": str(current_paragraph.id),
                            "paragraph_index": current_paragraph_index,
                            "source": current_paragraph.source_text,
                            "unstable_text": "",
                            "translation": current_paragraph.translated_text,
                            "detected_language": current_paragraph.detected_language,
                            "is_final": True,
                            "type": current_paragraph.type,
                         })
                    else:
                         merged_source = _merge_source_with_overlap(base_source, text)
                         current_paragraph = await update_paragraph_content(
                             session,
                             current_paragraph.id,
                             source_append=None,
                             translation_append=None,
                             detected_language=current_paragraph.detected_language,
                             paragraph_type="active",
                             source_override=merged_source,
                         )
                         await session.commit()

                         await websocket.send_json({
                            "conversation_id": str(conversation.id),
                            "paragraph_id": str(current_paragraph.id),
                            "paragraph_index": current_paragraph_index,
                            "source": current_paragraph.source_text,
                            "unstable_text": "",
                            "translation": current_paragraph.translated_text,
                            "detected_language": current_paragraph.detected_language,
                            "is_final": True,
                            "type": current_paragraph.type,
                            "translation_pending": True,
                         })

                         actual_target = target_language
                         if start_reverse_trans and reverse_target_name:
                             actual_target = reverse_target_name

                         source_language_name = pick_language_name(
                             chunk_lang,
                             current_paragraph.detected_language,
                             conversation.source_language,
                             whisper_lang,
                         )

                         logger.info("[CONVO_FLOW] Enqueueing translation task for p_id=%s, text=%s, target=%s", current_paragraph.id, text, actual_target)
                         translation_queue.put_nowait(
                             (
                                 current_paragraph.id,
                                 text,
                                 actual_target,
                                 "",
                                 current_paragraph_index,
                                 current_paragraph.detected_language,
                                 "active",
                                 source_language_name,
                             )
                         )
                    
                    # Splitting Check
                    paragraph_sentence_count += 1
                    full_source = current_paragraph.source_text or ""
                    total_chars = len(full_source)
                    total_words = len(full_source.split())
                    
                    should_split = False
                    if total_chars >= 300:
                        should_split = True
                    elif paragraph_sentence_count >= 2 and total_words >= 20:
                        should_split = True
                        
                    if should_split:
                        logger.info("[CONVO_FLOW] Splitting paragraph. Old p_id=%s", current_paragraph.id)
                        last_paragraph_source = current_paragraph.source_text or ""
                        await update_paragraph_content(
                            session, current_paragraph.id, source_append=None, translation_append=None,
                            detected_language=current_paragraph.detected_language,
                            paragraph_type="stable"
                        )
                        await session.commit()
                        logger.info("[CONVO_FLOW] Stabilized old paragraph %s", current_paragraph.id)
                        
                        try:
                            await websocket.send_json({
                                "conversation_id": str(conversation.id),
                                "paragraph_id": str(current_paragraph.id),
                                "paragraph_index": current_paragraph.paragraph_index,
                                "source": current_paragraph.source_text,
                                "unstable_text": "",
                                "translation": current_paragraph.translated_text,
                                "detected_language": current_paragraph.detected_language,
                                "is_final": True,
                                "type": "stable",
                            })
                        except Exception:
                            pass
                        
                        current_paragraph_index += 1
                        paragraph_sentence_count = 0
                        current_paragraph = await add_paragraph(
                            session, conversation, current_paragraph_index, "", ""
                        )
                        await session.commit()
                        # immediately provide unstable text for the new paragraph
                        try:
                            await websocket.send_json({
                                "conversation_id": str(conversation.id),
                                "paragraph_id": str(current_paragraph.id),
                                "paragraph_index": current_paragraph.paragraph_index,
                                "source": "",
                                "unstable_text": unstable_text,
                                "translation": "",
                                "detected_language": current_paragraph.detected_language,
                                "is_final": False,
                                "type": "active",
                            })
                        except Exception:
                            pass
                        logger.info("[CONVO_FLOW] Created new paragraph %s", current_paragraph.id)
                        
                        # Cancel any running rebuild task since we have new input
                        if conversation.id in active_rebuild_tasks:
                            task = active_rebuild_tasks[conversation.id]
                            if not task.done():
                                logger.info("Cancelling active rebuild task for conversation %s due to new input.", conversation.id)
                                task.cancel()

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error("Streaming error: %r", e)
            try:
                await websocket.send_json({"error": str(e)})
            except Exception:
                pass
        finally:
            try:
                # Stop worker
                translation_queue.put_nowait(None)
                # Wait for worker to finish processing pending items and shutdown
                await translation_queue.join()
                worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker_task
            except Exception:
                pass
            finally:
                if warmup_task and not warmup_task.done():
                    warmup_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await warmup_task

            try:
                await session.refresh(conversation, attribute_names=["paragraphs"])
                has_content = any(
                    (p.source_text and p.source_text.strip()) or (p.translated_text and p.translated_text.strip())
                    for p in conversation.paragraphs
                )
                if not has_content:
                    logger.info("Deleting empty conversation %s", conversation.id)
                    await delete_conversation(session, db_user, conversation.id)
                    await session.commit()
            except Exception as e:
                logger.error("Error during conversation cleanup: %s", e)

            try:
                await websocket.close()
            except RuntimeError:
                pass
            except Exception as e:
                logger.warning("Error closing websocket: %s", e)
            
            if conversation and conversation.id:
                await manager.disconnect(conversation.id, websocket)
