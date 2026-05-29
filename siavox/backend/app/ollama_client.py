import io
import json
import wave
import base64
import logging
import httpx2 as httpx
from typing import Optional, Tuple, List
from backend.app.config import settings

logger = logging.getLogger("siavox.ollama")

def _make_wav_bytes(raw_frames: bytes, nchannels: int, sampwidth: int, framerate: int) -> bytes:
    """Wraps raw PCM frames in a WAV container and returns the result as bytes."""
    chunk_io = io.BytesIO()
    with wave.open(chunk_io, 'wb') as wav_out:
        wav_out.setnchannels(nchannels)
        wav_out.setsampwidth(sampwidth)
        wav_out.setframerate(framerate)
        wav_out.writeframes(raw_frames)
    return chunk_io.getvalue()


def _split_wav_bytes(audio_bytes: bytes, max_chunk_duration: float = 10.0) -> List[bytes]:
    """Splits WAV bytes into chunks that respect natural speech boundaries.

    Strategy:
        1. Use auditok's energy-based VAD to locate non-silent speech segments.
        2. Walk the segments and, whenever the current chunk would exceed
           *max_chunk_duration*, place the split in the middle of the silence
           gap between the last completed segment and the next one.  This
           guarantees that chunk boundaries never fall inside a spoken word.
        3. If VAD returns no segments (e.g. extremely low energy audio), fall
           back to the original fixed-duration frame slicer so transcription
           can still proceed.

    Args:
        audio_bytes: Raw bytes of a WAV file.
        max_chunk_duration: Soft upper bound in seconds for each chunk (default
            10 s — small enough to keep latency and GPU memory low, while still
            giving the model enough context to avoid cross-boundary errors).

    Returns:
        A list of valid WAV byte buffers ready to be sent for transcription.
        If the total audio duration is within *max_chunk_duration*, the
        original bytes are returned as a single-element list.
    """
    try:
        wav_in = wave.open(io.BytesIO(audio_bytes), 'rb')
    except Exception as e:
        logger.warning(f"Failed to open audio as WAV: {e}. Processing as a single chunk.")
        return [audio_bytes]

    with wav_in:
        nchannels = wav_in.getnchannels()
        sampwidth = wav_in.getsampwidth()
        framerate = wav_in.getframerate()
        nframes = wav_in.getnframes()
        total_duration = nframes / framerate
        raw_frames = wav_in.readframes(nframes)

    if total_duration <= max_chunk_duration:
        return [audio_bytes]

    # --- Step 1: detect speech segments with auditok VAD ---
    try:
        import auditok
        segments = list(auditok.split(
            raw_frames,
            sampling_rate=framerate,
            sample_width=sampwidth,
            channels=nchannels,
            min_dur=0.2,      # ignore blips shorter than 200 ms
            max_dur=None,     # let segments be as long as needed
            max_silence=0.3,  # tolerate up to 300 ms of intra-segment silence
            energy_threshold=50,
        ))
    except Exception as e:
        logger.warning(f"auditok VAD failed ({e}); falling back to fixed-duration chunking.")
        segments = []

    # --- Step 2: fallback – fixed-duration slicer ---
    if not segments:
        logger.info("No VAD segments detected; using fixed-duration chunking.")
        bytes_per_chunk = int(max_chunk_duration * framerate) * sampwidth * nchannels
        chunks = []
        for offset in range(0, len(raw_frames), bytes_per_chunk):
            chunk_data = raw_frames[offset: offset + bytes_per_chunk]
            if chunk_data:
                chunks.append(_make_wav_bytes(chunk_data, nchannels, sampwidth, framerate))
        return chunks

    # --- Step 3: group segments into chunks at silence gaps ---
    split_intervals: List[tuple] = []
    current_start = 0.0
    chunk_first_seg_idx = 0
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.end - current_start > max_chunk_duration:
            if i > chunk_first_seg_idx:
                # Split in the middle of the gap between segments[i-1] and segments[i]
                prev_end = segments[i - 1].end
                split_time = (prev_end + seg.start) / 2.0
                split_intervals.append((current_start, split_time))
                current_start = split_time
                chunk_first_seg_idx = i
                # Do NOT advance i – re-evaluate this segment from the new start
            else:
                # Single segment already exceeds the limit; keep it whole
                split_time = current_start + max_chunk_duration
                split_intervals.append((current_start, split_time))
                current_start = split_time
                chunk_first_seg_idx = i
                i += 1
        else:
            i += 1

    if current_start < total_duration:
        split_intervals.append((current_start, total_duration))

    # --- Step 4: slice raw PCM bytes and wrap in WAV containers ---
    frame_size = sampwidth * nchannels
    chunks = []
    for start, end in split_intervals:
        start_byte = (int(start * framerate * sampwidth * nchannels) // frame_size) * frame_size
        end_byte = (int(end * framerate * sampwidth * nchannels) // frame_size) * frame_size
        start_byte = max(0, min(start_byte, len(raw_frames)))
        end_byte = max(start_byte, min(end_byte, len(raw_frames)))
        chunk_data = raw_frames[start_byte:end_byte]
        if chunk_data:
            chunks.append(_make_wav_bytes(chunk_data, nchannels, sampwidth, framerate))

    logger.info(
        f"VAD chunking: {total_duration:.1f}s audio → {len(chunks)} chunk(s) "
        f"(max {max_chunk_duration}s, {len(segments)} speech segments detected)"
    )
    return chunks

class OllamaClient:
    def __init__(self, host: str = settings.OLLAMA_HOST, model: str = settings.OLLAMA_MODEL):
        self.host = host.rstrip('/')
        self.model = model

    async def check_connection(self) -> bool:
        """Verifies if the Ollama service is reachable and responding."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.host}/api/tags", timeout=5.0)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.host}: {e}")
            return False

    async def is_model_loaded(self) -> bool:
        """Verifies if the configured model is currently loaded in Ollama's memory."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.host}/api/ps", timeout=5.0)
                if response.status_code != 200:
                    logger.error(f"Failed to fetch active models: Ollama returned status {response.status_code}")
                    return False
                
                data = response.json()
                models = data.get("models", [])
                
                # Normalize target model name
                target = self.model.strip().lower()
                target_with_tag = target if ":" in target else f"{target}:latest"
                
                for m in models:
                    name = m.get("name", "").strip().lower()
                    model_field = m.get("model", "").strip().lower()
                    
                    # Match options:
                    # 1. Exact match
                    if name == target or name == target_with_tag or model_field == target or model_field == target_with_tag:
                        return True
                    # 2. Substring match
                    if target in name or target in model_field:
                        return True
                        
                return False
        except Exception as e:
            logger.error(f"Failed to check if model {self.model} is loaded at {self.host}: {e}")
            return False

    async def _transcribe_chunk(self, chunk_bytes: bytes) -> str:
        """Transcribes a single audio chunk using the official Gemma 4 ASR prompt."""
        audio_b64 = base64.b64encode(chunk_bytes).decode('utf-8')
        
        prompt = (
            "Transcribe the following speech segment in its original language. "
            "Follow these specific instructions for formatting the answer:\n"
            "* Only output the transcription, with no newlines.\n"
            "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."
        )
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [audio_b64]
                }
            ],
            "stream": False
        }
        
        url = f"{self.host}/api/chat"
        logger.info(f"Transcribing audio chunk with model {self.model}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, timeout=90.0)
                if response.status_code != 200:
                    logger.error(f"Ollama ASR returned status {response.status_code}: {response.text}")
                    raise httpx.HTTPStatusError(
                        f"Ollama ASR error: status {response.status_code}",
                        request=response.request,
                        response=response
                    )
                
                result_json = response.json()
                content = result_json.get("message", {}).get("content", "").strip()
                return content
            except httpx.RequestError as e:
                logger.error(f"Connection error to Ollama during ASR: {e}")
                raise

    async def _refine_transcript(
        self,
        raw_transcript: str,
        output_format: str,
        target_language: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """Refines and translates the transcription using a text-only prompt.
        
        Returns:
            Tuple of (source_language, target_language, transformed_text)
        """
        format_description = (
            "a structured 'note' (a clear, formatted, bulleted list or concise summary of the spoken thoughts)"
            if output_format == "note" else
            "a polished 'message' (a cohesive, natural, and professional message ready to be sent on Slack/Email/SMS)"
        )
        
        translation_clause = ""
        if target_language and target_language.lower() not in ["auto", "same", "none"]:
            translation_clause = f"Translate the final transformed text into {target_language}."
        else:
            translation_clause = (
                "Determine if the user's custom instructions, speech, or context request translation. "
                "If so, translate the final transformed text into that target language. "
                "Otherwise, the final transformed text should remain in the detected source language."
            )
            
        extra_instructions = f"Additional instructions: {custom_instructions}" if custom_instructions else ""
        
        prompt = f"""Analyze the following transcription:
"{raw_transcript}"

Perform the following tasks:
1. Automatically detect the source language of the transcription. Save it in the 'source_language' field (e.g. 'English', 'Spanish', 'French').
2. Determine the target language for translation based on the user's instructions (like translation requested: '{target_language}') or custom instructions. If no translation is requested/needed, the target language is the same as the source language. Save the name of this target language in the 'target_language' field (e.g. 'English', 'Spanish', 'French').
3. Transform the transcription into {format_description}, and ensure it is written/translated in the determined target language. Save the resulting finalized and translated output in the 'transformed_text' field.
4. {translation_clause}
5. {extra_instructions}

Return the response ONLY as a JSON object with the following fields:
{{
  "source_language": "detected source language",
  "target_language": "determined target language",
  "transformed_text": "the finalized, formatted, and translated output"
}}
Do not add any text before or after the JSON block."""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "format": "json",
            "stream": False
        }
        
        url = f"{self.host}/api/chat"
        logger.info("Refining and formatting transcript with Ollama")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, timeout=90.0)
                if response.status_code != 200:
                    logger.error(f"Ollama refinement returned status {response.status_code}: {response.text}")
                    raise httpx.HTTPStatusError(
                        f"Ollama refinement error: status {response.status_code}",
                        request=response.request,
                        response=response
                    )
                
                result_json = response.json()
                content = result_json.get("message", {}).get("content", "").strip()
                
                try:
                    data = json.loads(content)
                    source_language = data.get("source_language", "").strip()
                    target_lang_detected = data.get("target_language", "").strip()
                    transformed_text = data.get("transformed_text", "").strip()
                    
                    if not target_lang_detected:
                        target_lang_detected = target_language if (target_language and target_language.lower() not in ["auto", "same", "none"]) else source_language
                    
                    return source_language, target_lang_detected, transformed_text
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse model refinement response as JSON: {content}")
                    fallback_target = target_language if (target_language and target_language.lower() not in ["auto", "same", "none"]) else "Unknown"
                    return "Unknown", fallback_target, content
            except httpx.RequestError as e:
                logger.error(f"Connection error to Ollama during refinement: {e}")
                raise

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribes audio data, splitting into VAD-aware chunks as necessary.

        Chunks are split at silence boundaries (never mid-word) using a 10-second
        soft ceiling to keep per-request latency and GPU memory consumption low.
        """
        chunks = _split_wav_bytes(audio_bytes, max_chunk_duration=10.0)
        
        raw_transcripts = []
        for index, chunk_bytes in enumerate(chunks):
            if len(chunks) > 1:
                logger.info(f"Processing audio chunk {index + 1}/{len(chunks)}")
            transcript = await self._transcribe_chunk(chunk_bytes)
            if transcript:
                raw_transcripts.append(transcript)
                
        return " ".join(raw_transcripts).strip()

    async def refine_transcript(
        self,
        raw_transcript: str,
        output_format: str,
        target_language: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """Refines and translates the transcription using a text-only prompt.
        
        Returns:
            Tuple of (source_language, target_language, transformed_text)
        """
        return await self._refine_transcript(raw_transcript, output_format, target_language, custom_instructions)

    async def process_audio(
        self,
        audio_bytes: bytes,
        output_format: str,  # "note" or "message"
        target_language: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> Tuple[str, str, str, str]:
        """Processes audio data by transcribing it first, then refining/translating the transcript."""
        raw_transcript = await self.transcribe_audio(audio_bytes)
        
        if not raw_transcript:
            return "", "Unknown", "Unknown", ""
            
        source_lang, target_lang, transformed_text = await self.refine_transcript(
            raw_transcript=raw_transcript,
            output_format=output_format,
            target_language=target_language,
            custom_instructions=custom_instructions
        )
        
        return raw_transcript, source_lang, target_lang, transformed_text
