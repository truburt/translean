"""
Streaming Whisper client (v2).

This implementation focuses on two complementary paths:
- A fast preview loop that uses a short tail window to keep the UI responsive.
- A stabilized loop that uses a longer window, VAD, and overlap trimming to
  produce high-quality, finalized text that feeds translation.

The public interface remains compatible with ``api.py``: ``warm_up`` primes the
model, and ``stream_transcription`` yields dictionaries with ``text`` or
``unstable_text`` plus an ``is_final`` flag.
"""
from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Dict, List, Tuple

import httpx

from .runtime_config import runtime_config

logger = logging.getLogger(__name__)

# Request cadence
PREVIEW_THROTTLE_SECONDS = 0.35
STABLE_THROTTLE_SECONDS = 1.1

# Windows
MIN_STABLE_BUFFER_SECONDS = 1.5
UNSTABLE_TAIL_SECONDS = 1.2

# Text/context
PROMPT_CONTEXT_CHARS = 400
OVERLAP_CONTEXT_CHARS = 400
FUZZY_MATCH_THRESHOLD = 0.82
FUZZY_MATCH_MIN_LENGTH = 4
MAX_COMMITTED_HISTORY_CHARS = 1000

# Whisper response filters are runtime-configurable.

# Payloads
MIN_AUDIO_CHUNK_SIZE = 2048

PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH_BYTES = 2
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
WEBM_HEADER_MAGIC = b"\x1a\x45\xdf\xa3"
MP4_FTYP_MAGIC = b"ftyp"


@dataclass
class RecordingConfig:
    """Describe how incoming audio is encoded for streaming."""

    mode: str
    container_format: str | None
    mime_type: str | None
    sample_rate: int
    channels: int


def _normalize_recording_config(
    recording_format: str | None,
    recording_mime_type: str | None,
    sample_rate: int | None,
    channels: int | None,
) -> RecordingConfig:
    """Normalize client recording metadata into a single config object."""

    format_hint = (recording_format or "").lower().strip()
    mime_hint = (recording_mime_type or "").lower().strip()

    resolved_sample_rate = sample_rate or PCM_SAMPLE_RATE
    resolved_channels = channels or PCM_CHANNELS

    if resolved_sample_rate != PCM_SAMPLE_RATE or resolved_channels != PCM_CHANNELS:
        logger.warning(
            "Client provided sample rate/channels (%s Hz, %s ch); expected %s Hz/%s ch. PCM timing assumes 16 kHz mono.",
            resolved_sample_rate,
            resolved_channels,
            PCM_SAMPLE_RATE,
            PCM_CHANNELS,
        )

    if format_hint in {"pcm", "pcm16", "pcm_s16le", "pcm16le"} or "pcm" in mime_hint:
        return RecordingConfig(
            mode="pcm",
            container_format=None,
            mime_type=recording_mime_type,
            sample_rate=resolved_sample_rate,
            channels=resolved_channels,
        )

    if format_hint in {"mp4", "m4a"} or mime_hint.startswith("audio/mp4"):
        return RecordingConfig(
            mode="container",
            container_format="mp4",
            mime_type=recording_mime_type,
            sample_rate=resolved_sample_rate,
            channels=resolved_channels,
        )

    return RecordingConfig(
        mode="container",
        container_format="webm",
        mime_type=recording_mime_type,
        sample_rate=resolved_sample_rate,
        channels=resolved_channels,
    )


@dataclass
class BufferTiming:
    """Simple timing snapshot for the current rolling buffer."""

    start_seconds: float | None
    end_seconds: float | None
    duration_seconds: float

    def cutoff(self, tail_seconds: float) -> float:
        """Return the absolute timestamp that separates stable from unstable audio."""

        if self.end_seconds is None:
            return max(self.duration_seconds - max(tail_seconds, 0.0), 0.0)

        return max(self.end_seconds - max(tail_seconds, 0.0), 0.0)


class ContainerToPCMTranscoder:
    """Stream containerized audio bytes through ffmpeg and collect PCM output."""

    def __init__(
        self,
        pcm_buffer: bytearray,
        *,
        sample_rate: int = PCM_SAMPLE_RATE,
        channels: int = PCM_CHANNELS,
        input_format: str | None = None,
        on_pcm: Callable[[], None] | None = None,
    ) -> None:
        self.pcm_buffer = pcm_buffer
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_format = input_format
        self.on_pcm = on_pcm
        self.process: aio_subprocess.Process | None = None
        self.stdout_task: asyncio.Task | None = None
        self.stderr_task: asyncio.Task | None = None
        self._closed = False
        self.total_bytes_written = 0
        self._corrector_buffer = bytearray()

    async def start(self) -> None:
        if self.process:
            return

        # Mobile recorders can emit non-monotonic DTS in streaming WebM chunks; ignore
        # timing and synthesize PTS so ffmpeg continues decoding instead of stalling.
        ffmpeg_args = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts+igndts",
        ]

        if self.input_format:
            ffmpeg_args.extend(["-f", self.input_format])

        ffmpeg_args.extend([
            "-i",
            "pipe:0",
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-vn",
            "-f",
            "s16le",
            "pipe:1",
        ])

        self.process = await aio_subprocess.create_subprocess_exec(
            *ffmpeg_args,
            stdin=aio_subprocess.PIPE,
            stdout=aio_subprocess.PIPE,
            stderr=aio_subprocess.PIPE,
        )

        self.stdout_task = asyncio.create_task(self._read_stdout())
        self.stderr_task = asyncio.create_task(self._log_stderr())

    async def _read_stdout(self) -> None:
        if not self.process or not self.process.stdout:
            return

        remainder = b""
        while True:
            chunk = await self.process.stdout.read(4096)
            if not chunk:
                break
            
            if remainder:
                chunk = remainder + chunk
                remainder = b""
            
            # Ensure we only ingest complete 16-bit frames (2 bytes)
            if len(chunk) % 2 != 0:
                remainder = chunk[-1:]
                chunk = chunk[:-1]

            if chunk:
                self.pcm_buffer.extend(chunk)
                if self.on_pcm:
                    self.on_pcm()

    async def _log_stderr(self) -> None:
        if not self.process or not self.process.stderr:
            return
        while True:
            line = await self.process.stderr.readline()
            if not line:
                break
            logger.warning("ffmpeg stderr: %s", line.decode(errors="ignore").strip())

    async def write(self, data: bytes) -> None:
        if self._closed:
            return
        if not self.process:
            await self.start()
        if not self.process or not self.process.stdin:
            return

        # Buffering Strategy:
        # ffmpeg's matroska demuxer can fail if incomplete EBML elements (like block headers)
        # are written to the pipe in separate calls, causing partial parsing errors 
        # (e.g. "Invalid track number").
        # We buffer incoming chunks until we have a safe amount of data or a timeout/flush
        # to ensure we pass larger, more complete blocks to ffmpeg.

        self._corrector_buffer.extend(data)
        
        # Threshold: 1024 bytes is a reasonable chunk size for pipe writes to avoid fragmentation
        # while keeping latency low (approx 250ms for 32kbps Opus).
        if len(self._corrector_buffer) < 1024:
            return

        payload = bytes(self._corrector_buffer)
        self._corrector_buffer.clear()

        try:
            # logger.info(f"Writing {len(payload)} bytes to ffmpeg stdin")
            self.process.stdin.write(payload)
            await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            logger.warning(f"ffmpeg stdin write failed: {e}")
            self._closed = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self.process and self.process.stdin:
            try:
                # Flush remaining buffer
                if self._corrector_buffer:
                    logger.info(f"Flushing {len(self._corrector_buffer)} buffered bytes to ffmpeg")
                    # Must convert to bytes to avoid BufferError (export resize)
                    payload = bytes(self._corrector_buffer)
                    self._corrector_buffer.clear()
                    self.process.stdin.write(payload)
            except (BrokenPipeError, AttributeError) as e:
                pass

            try:
                logger.info("Sending EOF to ffmpeg stdin")
                self.process.stdin.write_eof()
            except (BrokenPipeError, AttributeError) as e:
                logger.info(f"Failed to write EOF to ffmpeg: {e}")
                pass
            try:
                await self.process.stdin.drain()
            except Exception:
                pass
            try:
                self.process.stdin.close()
            except Exception:
                pass

        if self.stdout_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self.stdout_task
        if self.stderr_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self.stderr_task

        if self.process:
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.process.kill()


@dataclass
class StreamState:
    """Mutable streaming state."""

    preview_delta_buffer: bytearray = field(default_factory=bytearray)
    stable_pcm_buffer: bytearray = field(default_factory=bytearray)
    full_pcm_buffer: bytearray = field(default_factory=bytearray)
    processed_pcm_bytes: int = 0
    buffer_changed: bool = False
    committed_text: str = ""
    pending_unstable: str = ""
    last_preview_request_time: float = field(default_factory=time.time)
    last_stable_request_time: float = field(default_factory=time.time)
    last_audio_time: float = field(default_factory=time.time)
    last_commit_time: float = field(default_factory=time.time)
    buffer_committed_offset: float = 0.0
    accumulated_preview: str = ""

    def reset(self) -> None:
        now = time.time()
        self.preview_delta_buffer = bytearray()
        self.stable_pcm_buffer = bytearray()
        self.full_pcm_buffer = bytearray()
        self.processed_pcm_bytes = 0
        self.buffer_changed = False
        self.committed_text = ""
        self.pending_unstable = ""
        self.accumulated_preview = ""
        self.last_preview_request_time = now
        self.last_stable_request_time = now
        self.last_audio_time = now
        self.last_commit_time = now
        self.buffer_committed_offset = 0.0

    def append_committed(self, text: str) -> None:
        self.committed_text = (self.committed_text + " " + text).strip()
        # Prevent indefinite growth; we only need the tail for functionality
        if len(self.committed_text) > MAX_COMMITTED_HISTORY_CHARS:
            self.committed_text = self.committed_text[-MAX_COMMITTED_HISTORY_CHARS:]


# ------------------
# Audio helpers
# ------------------


def _compute_pcm_timing(pcm_buffer: bytearray) -> BufferTiming:
    duration = len(pcm_buffer) / PCM_BYTES_PER_SECOND if pcm_buffer else 0.0
    if duration <= 0:
        return BufferTiming(None, None, 0.0)
    return BufferTiming(0.0, duration, duration)


def _tail_pcm_payload(pcm_buffer: bytearray, tail_seconds: float) -> tuple[bytes, float]:
    if not pcm_buffer or tail_seconds <= 0:
        return b"", 0.0

    carry_bytes = int(max(tail_seconds, 0.0) * PCM_BYTES_PER_SECOND)
    # Ensure even alignment for 16-bit PCM
    if carry_bytes % 2 != 0:
        carry_bytes += 1

    if carry_bytes <= 0:
        return b"", 0.0

    payload = bytes(pcm_buffer[-carry_bytes:])
    retained = len(payload) / PCM_BYTES_PER_SECOND
    return payload, retained


def _trim_pcm_buffer(pcm_buffer: bytearray, tail_seconds: float) -> tuple[bytearray, float]:
    payload, retained = _tail_pcm_payload(pcm_buffer, tail_seconds)
    if not payload:
        pcm_buffer.clear()
        return pcm_buffer, 0.0
    if len(payload) != len(pcm_buffer):
        pcm_buffer[:] = payload
    return pcm_buffer, retained


def _ingest_pcm_bytes(state: StreamState) -> None:
    """Append newly decoded PCM bytes into preview + stable buffers."""

    if not state.full_pcm_buffer:
        return

    if state.processed_pcm_bytes >= len(state.full_pcm_buffer):
        return

    new_pcm = bytes(state.full_pcm_buffer[state.processed_pcm_bytes :])
    state.processed_pcm_bytes = len(state.full_pcm_buffer)

    if not new_pcm:
        return
    if all(b == 0 for b in new_pcm):
        logger.warning("Received all-zero PCM chunk from client (%d bytes).", len(new_pcm))

    state.preview_delta_buffer.extend(new_pcm)
    state.stable_pcm_buffer.extend(new_pcm)

    if state.processed_pcm_bytes:
        del state.full_pcm_buffer[: state.processed_pcm_bytes]
        state.processed_pcm_bytes = 0


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = PCM_SAMPLE_RATE, channels: int = PCM_CHANNELS) -> bytes:
    bits_per_sample = PCM_SAMPLE_WIDTH_BYTES * 8
    byte_rate = sample_rate * channels * PCM_SAMPLE_WIDTH_BYTES
    block_align = channels * PCM_SAMPLE_WIDTH_BYTES
    data_size = len(pcm_data)

    header = bytearray()
    header.extend(b"RIFF")
    header.extend((36 + data_size).to_bytes(4, "little"))
    header.extend(b"WAVE")
    header.extend(b"fmt ")
    header.extend((16).to_bytes(4, "little"))
    header.extend((1).to_bytes(2, "little"))  # PCM format
    header.extend(channels.to_bytes(2, "little"))
    header.extend(sample_rate.to_bytes(4, "little"))
    header.extend(byte_rate.to_bytes(4, "little"))
    header.extend(block_align.to_bytes(2, "little"))
    header.extend(bits_per_sample.to_bytes(2, "little"))
    header.extend(b"data")
    header.extend(data_size.to_bytes(4, "little"))
    return bytes(header + pcm_data)


# ------------------
# Request utilities
# ------------------


def _clean_transcription(text: str) -> str:
    """Normalize text and strip garbage punctuation."""

    text = text.strip()
    if not any(c.isalnum() for c in text):
        return ""

    text = re.sub(r"^[,.;:!?]+", "", text).strip() # remove leading punctuation
    text = re.sub(r"\.{2,}", " ", text) # replace double periods with space
    text = re.sub(r"([,.;:!?])\s*(?=[,.;:!?])", "", text) # remove consecutive punctuation
    text = re.sub(r"\s+", " ", text) # replace multiple spaces with single space

    if re.fullmatch(r"[.,;:!?]+", text): # remove pure punctuation
        return ""

    return text.strip()


def _tokenize_words(text: str) -> List[re.Match]:
    return list(re.finditer(r"\w+", text))


def _deduplicate_overlap_text(current_text: str, committed_context: str, max_overlap_chars: int = OVERLAP_CONTEXT_CHARS) -> str:
    """Remove duplicated prefixes caused by overlapping audio buffers."""

    if not committed_context or not current_text:
        return current_text

    context_window = committed_context[-max_overlap_chars:]
    context_tokens = [m.group(0).lower() for m in _tokenize_words(context_window)]
    current_matches = _tokenize_words(current_text)
    current_tokens = [m.group(0).lower() for m in current_matches]

    if not context_tokens or not current_tokens:
        return current_text

    max_token_window = len(current_tokens)
    for overlap_size in range(min(len(context_tokens), max_token_window), 0, -1):
        ctx_slice = context_tokens[-overlap_size:]
        cur_slice = current_tokens[:overlap_size]

        match = True
        for t1, t2 in zip(ctx_slice, cur_slice):
            if t1 == t2:
                continue
            if len(t1) < FUZZY_MATCH_MIN_LENGTH or len(t2) < FUZZY_MATCH_MIN_LENGTH:
                match = False
                break
            if (
                re.sub(r"\d", "", t1) == re.sub(r"\d", "", t2)
                and abs(len(t1) - len(t2)) <= 2
            ):
                continue
            similarity = _sequence_similarity(t1, t2)
            if similarity < FUZZY_MATCH_THRESHOLD:
                match = False
                break

        if match:
            prefix_end = current_matches[overlap_size - 1].end()
            remainder = current_text[prefix_end:].lstrip()
            return re.sub(r"^[,.;:!?]+", "", remainder).strip()

    return current_text


def _sequence_similarity(a: str, b: str) -> float:
    """Lightweight similarity metric."""

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / max(len(a), len(b))



def _filter_segments(segments: List[dict], chars_removed: int) -> List[dict]:
    """Filter out segments that were removed by deduplication."""
    if chars_removed <= 0 or not segments:
        return segments

    new_segments = []
    removed_count = 0
    for seg in segments:
        seg_text = seg.get("text", "")
        # Use approx length matching
        seg_len = len(seg_text)
        
        # If the segment is fully within the removed region, drop it
        if removed_count + seg_len <= chars_removed:
            removed_count += seg_len
            continue
            
        # If partially removed, just keep for simplicity
        new_segments.append(seg)
        removed_count += seg_len
        
    return new_segments


def _split_stable_unstable(
    text: str,
    segments: List[dict],
    timing: BufferTiming,
    unstable_tail_seconds: float,
) -> Tuple[str, str, float]:
    """Split transcription text into stable/unstable regions using segment timecodes."""

    if not text:
        if unstable_tail_seconds <= 0.0:
            return "", "", timing.duration_seconds
        return "", text, 0.0

    if unstable_tail_seconds <= 0.0:
        return text, "", timing.duration_seconds

    abs_cutoff = timing.cutoff(unstable_tail_seconds)
    rel_cutoff = abs_cutoff if timing.start_seconds is None else max(abs_cutoff - timing.start_seconds, 0.0)

    stable_len = 0
    total_len = 0

    if segments:
        for seg in segments:
            seg_text = seg.get("text", "") or ""
            seg_end = seg.get("end", 0.0) or 0.0
            seg_len = len(seg_text)
            total_len += seg_len
            if seg_end <= rel_cutoff:
                stable_len += seg_len

    if total_len <= 0:
        total_len = len(text)
        stable_len = int(max(total_len - (len(text) * (unstable_tail_seconds / max(timing.duration_seconds, 0.01))), 0))

    ratio = min(max(stable_len / max(total_len, 1), 0.0), 1.0)
    split_idx = int(len(text) * ratio)
    split_idx = max(0, min(len(text), split_idx))

    boundary_matches = list(re.finditer(r"[\s.,!?;:()\-\u2013\u2014]", text[:split_idx]))
    if boundary_matches:
        split_idx = boundary_matches[-1].start()

    stable_text = text[:split_idx].rstrip()
    unstable_text = text[split_idx:].lstrip()

    # Calculate effective cutoff using segment interpolation
    effective_cutoff = 0.0
    found_time = False
    
    if segments:
        cumulative_len = 0
        for seg in segments:
             seg_text = seg.get("text", "")
             seg_len = len(seg_text)
             if split_idx <= cumulative_len + seg_len:
                  # inside this segment
                  offset_chars = max(0, split_idx - cumulative_len)
                  seg_start = seg.get("start", 0.0)
                  seg_end = seg.get("end", 0.0)
                  seg_dur = max(0.0, seg_end - seg_start)
                  
                  time_offset = (offset_chars / max(seg_len, 1)) * seg_dur
                  effective_cutoff = seg_start + time_offset
                  found_time = True
                  break
             cumulative_len += seg_len
             
        # If we went past all segments (e.g. split_idx at end), use the last segment's end
        if not found_time and segments:
             effective_cutoff = segments[-1].get("end", 0.0)
             found_time = True

    if not found_time:
         # Fallback to ratio
         effective_cutoff = (split_idx / max(len(text), 1)) * timing.duration_seconds

    return stable_text, unstable_text, effective_cutoff


def _segment_is_filtered(segment: dict) -> bool:
    no_speech_prob = segment.get("no_speech_prob")
    avg_logprob = segment.get("avg_logprob")
    compression_ratio = segment.get("compression_ratio")

    if no_speech_prob is not None and no_speech_prob > runtime_config.no_speech_prob_skip:
        return True

    if (
        avg_logprob is not None
        and no_speech_prob is not None
        and avg_logprob < runtime_config.avg_logprob_skip
        and no_speech_prob > runtime_config.no_speech_prob_logprob_skip
    ):
        return True

    if compression_ratio is not None and compression_ratio > runtime_config.compression_ratio_skip:
        return True

    return False


async def _send_transcription_request(
    client: httpx.AsyncClient,
    audio_data: bytes,
    language_code: str,
    prompt: str | None,
    mode: str,
) -> dict:
    """Send audio to Whisper and return parsed response."""

    files = {"file": ("audio.wav", audio_data, "audio/wav")}
    params: Dict[str, object] = {
        "vad_filter": True,
        "condition_on_previous_text": False,
        "compression_ratio_threshold": runtime_config.compression_ratio_skip,
        "suppress_blank": True,
        "no_repeat_ngram_size": 4,
    }

    if language_code and language_code.lower() != "auto":
        params["language"] = language_code

    if prompt:
        params["initial_prompt"] = prompt

    if runtime_config.whisper_keep_alive_seconds > 0:
        params["keep_alive"] = runtime_config.whisper_keep_alive_seconds

    if mode == "preview":
        params.update(
            {
                "beam_size": 2,
                "best_of": 2,
                "log_prob_threshold": -1.0,
                "temperature": 0.0,
            }
        )
    else:
        params.update(
            {
                "beam_size": 5,
                "best_of": 5,
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "repetition_penalty": 1.05,
            }
        )

    data = {
        "model_name": runtime_config.whisper_model,
        "params": json.dumps(params),
    }

    response = await client.post(
        "transcribe",
        files=files,
        data=data,
    )
    response.raise_for_status()
    payload = response.json()

    language = payload.get("language") or payload.get("info", {}).get("language", "auto")
    segments = payload.get("segments", [])

    # Dynamic Hallucination Filtering
    # If a segment is significantly longer than the input audio, it is likely a hallucination
    # (e.g. "Thank you for watching" on a 1s silence).
    # We use a dynamic ratio: if segment_len > audio_len * 2.0, filter it out.
    # We also enforce a minimum audio duration to avoid false positives on very short chunks.
    
    # Calculate audio duration from WAV bytes (subtract 44-byte header)
    audio_duration = max(0.0, (len(audio_data) - 44) / PCM_BYTES_PER_SECOND)

    filtered_segments = []
    if segments:
        for segment in segments:
            if _segment_is_filtered(segment):
                continue
            
            # Duration-based hallucination check
            seg_start = segment.get("start", 0.0)
            seg_end = segment.get("end", 0.0)
            seg_dur = seg_end - seg_start
            
            # Only apply if audio is long enough to be meaningful (e.g. > 1.0s)
            # and if the segment is anomalously long compared to the input.
            if audio_duration > 1.0 and seg_dur > (audio_duration * 2.0):
                logger.warning(
                    "Filtering hallucination: segment_dur=%.2fs > limit=%.2fs (audio=%.2fs) text='%s'",
                    seg_dur, audio_duration * 2.0, audio_duration, segment.get("text", "").strip()
                )
                continue
                
            filtered_segments.append(segment)
            
        text = _clean_transcription("".join(s.get("text", "") for s in filtered_segments))
        segments = filtered_segments
    else:
        text = _clean_transcription(payload.get("text", ""))

    if not text:
        return {"text": "", "language": language, "segments": segments}

    return {"text": text, "language": language, "segments": segments}


# ------------------
# Streaming helpers
# ------------------


def _build_base_url() -> str:
    base = runtime_config.whisper_base_url
    if not base.endswith("/"):
        base += "/"
    return base


def _update_prompt_context(committed_text: str) -> str | None:
    if not committed_text:
        return None
    return committed_text[-PROMPT_CONTEXT_CHARS:]


async def _run_preview_pass(
    *,
    client: httpx.AsyncClient,
    state: StreamState,
    language_code: str,
    preview_timing: BufferTiming,
    now: float,
) -> list[dict]:
    """Handle the short-lived preview transcription flow using only new audio."""

    preview_seconds = preview_timing.duration_seconds
    if preview_seconds < runtime_config.min_preview_buffer_seconds:
        return []

    if now - state.last_preview_request_time < PREVIEW_THROTTLE_SECONDS:
        return []

    preview_pcm = bytes(state.preview_delta_buffer)
    if len(preview_pcm) < MIN_AUDIO_CHUNK_SIZE:
        return []

    prompt = _update_prompt_context(state.committed_text)
    logger.info(
        'Sending preview transcription request: committed_text="%s", preview_pcm=%d',
        state.committed_text,
        len(preview_pcm),
    )
    preview_result = await _send_transcription_request(
        client,
        _pcm_to_wav(preview_pcm),
        language_code=language_code,
        prompt=prompt,
        mode="preview",
    )
    logger.info("Preview transcription result: %s", preview_result)
    preview_text = _deduplicate_overlap_text(preview_result["text"], state.committed_text)
    preview_delta_text = _deduplicate_overlap_text(preview_text, state.pending_unstable)
    
    # Also deduplicate against what we already accumulated in this cycle
    preview_delta_text = _deduplicate_overlap_text(preview_delta_text, state.accumulated_preview)

    if preview_delta_text:
        state.accumulated_preview = (state.accumulated_preview + " " + preview_delta_text).strip()

    events: list[dict] = []
    if state.accumulated_preview:
        # Don't mutate state.pending_unstable here, just emit the combined view for the UI.
        combined_unstable = " ".join(
            part for part in (state.pending_unstable, state.accumulated_preview) if part
        ).strip()
        
        events.append(
            {
                "unstable_text": combined_unstable,
                "is_final": False,
                "language": preview_result["language"],
            }
        )

    state.last_preview_request_time = now
    state.preview_delta_buffer.clear()
    return events


async def _run_stable_pass(
    *,
    client: httpx.AsyncClient,
    state: StreamState,
    language_code: str,
    stable_timing: BufferTiming,
    force_commit: bool,
    now: float,
) -> list[dict]:
    """Handle the stabilized transcription flow."""

    stable_seconds = stable_timing.duration_seconds
    if stable_seconds <= 0:
        return []

    if not force_commit and stable_seconds < MIN_STABLE_BUFFER_SECONDS:
        return []

    if (
        not force_commit
        and stable_seconds < runtime_config.stable_window_seconds
        and now - state.last_stable_request_time < STABLE_THROTTLE_SECONDS
    ):
        return []

    pcm_payload = bytes(state.stable_pcm_buffer)
    if len(pcm_payload) < MIN_AUDIO_CHUNK_SIZE:
        return []

    is_silence = all(b == 0 for b in pcm_payload)
    if is_silence:
        logger.warning("Sending ALL-ZERO PCM payload to Whisper!")
    else:
        # Check for near-silence or static?
        pass

    prompt = _update_prompt_context(state.committed_text)
    logger.info(
        'Sending stable transcription request: committed_text="%s", pcm_payload=%d',
        state.committed_text,
        len(pcm_payload),
    )
    result = await _send_transcription_request(
        client,
        _pcm_to_wav(pcm_payload),
        language_code=language_code,
        prompt=prompt,
        mode="stable",
    )
    logger.info("Stable transcription result: %s", result)

    # 1. Use full text and segments (filtering proved too aggressive causing data loss)
    segments = result.get("segments", [])
    text = _clean_transcription(result["text"])
    
    # 2. Safety deduplication (text based) for boundary issues
    original_text_len = len(text)
    deduped_text = _deduplicate_overlap_text(text, state.committed_text)
    full_overlap_dedup = (text and not deduped_text)
    text = deduped_text
    
    # Filter segments if we dedicated significant text
    chars_removed = original_text_len - len(text)
    if chars_removed > 0:
        segments = _filter_segments(segments, chars_removed)

    # 3. Split into NEW stable and unstable
    if full_overlap_dedup:
         # Special case: The entirety of 'text' was already committed.
         # Both stable and unstable text are empty.
         # We must trim the buffer, BUT only up to the end of the text we found.
         # If the buffer contains extra audio (tail) that wasn't transcribed, we must keep it.
         if segments:
             # Use a safety margin (e.g. 0.5s) to avoid cutting fresh audio that Whisper might have 
             # carelessly included in the old segment's duration timestamps.
             # This prevents data loss if a segment boundary is fuzzy.
             last_end = segments[-1].get("end", 0.0)
             stabilized_duration = max(0.0, last_end - 0.5)
         else:
             stabilized_duration = stable_timing.duration_seconds
             
         stable_text = ""
         unstable_text = ""
    else:
         unstable_tail = 0.0 if force_commit or stable_seconds <= UNSTABLE_TAIL_SECONDS else UNSTABLE_TAIL_SECONDS
         stable_text, unstable_text, stabilized_duration = _split_stable_unstable(
             text, segments, stable_timing, unstable_tail
         )
         
         # Fix for Data Loss on Force Commit:
         # If we forced a commit (e.g. silence), _split_stable_unstable assumes we stabilized everything.
         # But if segments end early, we must NOT claim we stabilized the tail, or we will delete valid audio.
         if force_commit and segments:
             last_end = segments[-1].get("end", 0.0)
             if last_end < stable_timing.duration_seconds - 0.5:
                 # If we have significant trailing audio, keep it.
                 stabilized_duration = min(stabilized_duration, max(0.0, last_end))

    # 4. Emit events
    events: list[dict] = []

    # Rescue logic: If force_commit (silence/timeout) and we have pending unstable text 
    # that wasn't captured in the new stable_text (e.g. because of strict filtering or silence),
    # we should commit the pending text to prevent data loss.
    rescue_candidate = state.pending_unstable
    if force_commit and rescue_candidate and not stable_text:
         events.append({"text": rescue_candidate, "is_final": True, "language": result["language"]})
         state.append_committed(rescue_candidate)
         state.pending_unstable = ""

    if stable_text:
        events.append({"text": stable_text, "is_final": True, "language": result["language"]})
        state.append_committed(stable_text)
        state.pending_unstable = ""

    if unstable_text and not force_commit:
        events.append({"unstable_text": unstable_text, "is_final": False, "language": result["language"]})
        state.pending_unstable = unstable_text
    elif not unstable_text:
        state.pending_unstable = ""

    # 5. Trim Buffer & Update Overlap Offset
    if not text and force_commit:
        state.stable_pcm_buffer.clear()
        state.buffer_committed_offset = 0.0
    elif stable_text or force_commit or full_overlap_dedup:
        tail_floor = max(0.0, stable_seconds - stabilized_duration)
        
        if full_overlap_dedup or force_commit:
            # If full overlap (duplicate) OR force commit, we only keep what wasn't stabilized.
            # We do NOT force UNSTABLE_TAIL_SECONDS, which avoids loops and allows flushing.
            tail_to_keep = tail_floor
        else:
            # Normal operation: maintain overlap context
            tail_to_keep = max(UNSTABLE_TAIL_SECONDS, tail_floor)
        
        state.stable_pcm_buffer, _ = _trim_pcm_buffer(
            state.stable_pcm_buffer,
            tail_to_keep,
        )

        remaining_unstable = max(0.0, stable_seconds - stabilized_duration)
        state.buffer_committed_offset = max(0.0, tail_to_keep - remaining_unstable)

    elif stable_seconds > runtime_config.stable_window_seconds and not state.pending_unstable:
        state.stable_pcm_buffer, _ = _trim_pcm_buffer(
            state.stable_pcm_buffer,
            runtime_config.stable_window_seconds,
        )
        state.buffer_committed_offset = 0.0

    if stable_text or force_commit or full_overlap_dedup:
        state.last_commit_time = now
    state.last_stable_request_time = now
    
    # We have established a new stable/unstable baseline.
    # The accumulated preview is now invalid/superseded.
    state.accumulated_preview = ""
    state.preview_delta_buffer.clear()
    return events


async def _warmup_request(client: httpx.AsyncClient, language_code: str = "auto") -> bool:
    duration_secs = 1
    pcm_silence = bytes(int(duration_secs * PCM_BYTES_PER_SECOND))
    wav_data = _pcm_to_wav(pcm_silence, sample_rate=PCM_SAMPLE_RATE, channels=PCM_CHANNELS)

    params = {}
    if language_code and language_code.lower() != "auto":
        params["language"] = language_code

    if runtime_config.whisper_keep_alive_seconds > 0:
        params["keep_alive"] = runtime_config.whisper_keep_alive_seconds

    files = {"file": ("warmup.wav", wav_data, "audio/wav")}
    data = {"model_name": runtime_config.whisper_model, "params": json.dumps(params)}

    try:
        resp = await client.post("transcribe", files=files, data=data)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Whisper warm-up failed: %s", exc)
        return False


# ------------------
# Public functions
# ------------------


async def warm_up(language_code: str = "auto") -> bool:
    """Send a short silent audio to warm up the model."""

    base_url = _build_base_url()
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        return await _warmup_request(client, language_code)


async def stream_transcription(
    audio_chunks: AsyncIterator[bytes],
    language_code: str,
    model_ready_event: asyncio.Event | None = None,
    recording_format: str | None = None,
    recording_mime_type: str | None = None,
    recording_sample_rate: int | None = None,
    recording_channels: int | None = None,
) -> AsyncIterator[dict]:
    """Stream audio chunks to Whisper and yield preview + stabilized text.

    The stream maintains two parallel flows:
    - A fast preview buffer (~0.5s minimum) that sends only new audio while
      emitting the accumulated unstable text to keep the UI responsive.
    - A longer, overlapping stable buffer (>=1.5s) that is periodically
      transcribed, split into stable/unstable portions, and deduplicated
      against committed text. Silence gaps force a final stabilization pass.

    Recording metadata provided by the client (container vs PCM) is used to
    decide whether to run an ffmpeg transcoder or ingest PCM bytes directly.
    """

    base_url = _build_base_url()
    state = StreamState()
    recording_config = _normalize_recording_config(
        recording_format,
        recording_mime_type,
        recording_sample_rate,
        recording_channels,
    )
    transcoder: ContainerToPCMTranscoder | None = None
    pcm_remainder = b""

    def mark_buffer_changed() -> None:
        state.buffer_changed = True

    async def shutdown_transcoder() -> None:
        nonlocal transcoder
        if transcoder:
            await transcoder.close()
            transcoder = None

    async def start_transcoder(reset_state: bool = True) -> None:
        nonlocal transcoder
        await shutdown_transcoder()
        
        if reset_state:
            now = time.time()
            state.preview_delta_buffer = bytearray()
            state.stable_pcm_buffer = bytearray()
            state.full_pcm_buffer = bytearray()
            state.processed_pcm_bytes = 0
            state.buffer_changed = False
            state.last_preview_request_time = now
            state.last_stable_request_time = now
            state.last_commit_time = now
            state.last_audio_time = now
            
        transcoder = ContainerToPCMTranscoder(
            state.full_pcm_buffer,
            input_format=recording_config.container_format,
            on_pcm=mark_buffer_changed,
        )
        await transcoder.start()

    async def apply_recording_config(update: dict) -> None:
        nonlocal recording_config
        nonlocal pcm_remainder

        incoming = _normalize_recording_config(
            update.get("recording_format"),
            update.get("recording_mime_type"),
            update.get("recording_sample_rate"),
            update.get("recording_channels"),
        )

        if incoming.mode == recording_config.mode and incoming.container_format == recording_config.container_format:
            recording_config = incoming
            return

        recording_config = incoming
        pcm_remainder = b""
        await shutdown_transcoder()
        if recording_config.mode == "container":
            await start_transcoder(reset_state=False)

    def is_container_header(payload: bytes) -> bool:
        if recording_config.container_format == "mp4":
            return len(payload) >= 8 and payload[4:8] == MP4_FTYP_MAGIC
        return len(payload) >= 4 and payload.startswith(WEBM_HEADER_MAGIC)

    async with httpx.AsyncClient(base_url=base_url, timeout=45.0) as client:
        chunk_queue: asyncio.Queue = asyncio.Queue()
        processing_complete_sent = False

        async def producer():
            try:
                async for chunk in audio_chunks:
                    await chunk_queue.put(chunk)
                await chunk_queue.put(None)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Producer error: %s", exc)
                await chunk_queue.put(None)

        producer_task = asyncio.create_task(producer())

        async def flush_buffer():
            if not state.stable_pcm_buffer:
                return

            prompt = _update_prompt_context(state.committed_text)
            pcm_payload = bytes(state.stable_pcm_buffer)
            if len(pcm_payload) >= MIN_AUDIO_CHUNK_SIZE:
                try:
                    payload = _pcm_to_wav(pcm_payload)
                    logger.info('Sending final transcription request: committed_text="%s", payload=%d', state.committed_text, len(payload))
                    result = await _send_transcription_request(
                        client,
                        payload,
                        language_code=language_code,
                        prompt=prompt,
                        mode="stable",
                    )
                    logger.info('Final transcription result: %s', result)
                    text = _deduplicate_overlap_text(result["text"], state.committed_text)
                    if text:
                        state.append_committed(text)
                        yield {"text": text, "is_final": True, "language": result["language"]}
                except Exception as exc:  # pragma: no cover - network errors
                    logger.warning("Flush failed: %s", exc)

        try:
            while True:
                chunks = []
                try:
                    # Wait for at least one chunk
                    chunks.append(await asyncio.wait_for(chunk_queue.get(), timeout=0.1))
                    # Drain the rest of the queue immediately
                    while not chunk_queue.empty():
                        chunks.append(chunk_queue.get_nowait())
                except asyncio.TimeoutError:
                    pass
                
                found_none = False
                for chunk in chunks:
                    if chunk is None:
                        found_none = True
                        break

                    if isinstance(chunk, dict):
                        if chunk.get("type") == "stop_recording":
                            if state.buffer_changed:
                                _ingest_pcm_bytes(state)
                                state.buffer_changed = False
                            await shutdown_transcoder()
                            _ingest_pcm_bytes(state)
                            async for item in flush_buffer():
                                yield item
                            state.reset()
                            yield {"status": "processing_complete"}
                            processing_complete_sent = True 

                        if "recording_format" in chunk or "recording_mime_type" in chunk:
                            await apply_recording_config(chunk)

                        if "source_language" in chunk:
                            lang_val = chunk["source_language"]
                            if isinstance(lang_val, list):
                                language_code = lang_val[0] if lang_val else "auto"
                            else:
                                language_code = lang_val

                        yield chunk
                        continue

                    if chunk:
                        if not isinstance(chunk, (bytes, bytearray)):
                            logger.warning("Unexpected streaming payload type: %s", type(chunk))
                            continue
                        if recording_config.mode == "pcm":
                            payload = bytes(chunk)
                            if pcm_remainder:
                                payload = pcm_remainder + payload
                                pcm_remainder = b""

                            if len(payload) % 2 != 0:
                                pcm_remainder = payload[-1:]
                                payload = payload[:-1]

                            if payload:
                                state.full_pcm_buffer.extend(payload)
                                state.buffer_changed = True
                                state.last_audio_time = time.time()
                            continue

                        if not transcoder:
                            await start_transcoder(reset_state=False)

                        if is_container_header(chunk):
                            logger.info("New %s header detected in chunk, restarting transcoder (preserving state)", recording_config.container_format)
                            await start_transcoder(reset_state=False)

                        if not transcoder:
                            continue

                        await transcoder.write(chunk)
                        state.last_audio_time = time.time()
                
                if found_none:
                    break

                now = time.time()
                if state.buffer_changed:
                    _ingest_pcm_bytes(state)
                    state.buffer_changed = False

                silence_duration = now - state.last_audio_time
                preview_timing = _compute_pcm_timing(state.preview_delta_buffer)
                stable_timing = _compute_pcm_timing(state.stable_pcm_buffer)

                if not state.stable_pcm_buffer and silence_duration < runtime_config.silence_finalize_seconds:
                    continue

                if model_ready_event and not model_ready_event.is_set():
                    continue

                force_commit = (now - state.last_commit_time) >= runtime_config.commit_timeout_seconds or (
                    silence_duration >= runtime_config.silence_finalize_seconds
                )

                stable_events = await _run_stable_pass(
                    client=client,
                    state=state,
                    language_code=language_code,
                    stable_timing=stable_timing,
                    force_commit=force_commit,
                    now=now,
                )
                for event in stable_events:
                    yield event

                preview_events = await _run_preview_pass(
                    client=client,
                    state=state,
                    language_code=language_code,
                    preview_timing=preview_timing,
                    now=now,
                )
                for event in preview_events:
                    yield event

        finally:
            await shutdown_transcoder()
            try:
                if state.buffer_changed:
                    _ingest_pcm_bytes(state)
                    state.buffer_changed = False
                async for item in flush_buffer():
                    yield item
            finally:
                if not processing_complete_sent:
                    yield {"status": "processing_complete"}
                producer_task.cancel()
