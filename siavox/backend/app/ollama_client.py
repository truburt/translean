import json
import base64
import logging
import httpx2 as httpx
from typing import Optional, Tuple, List
from backend.app.config import settings

logger = logging.getLogger("siavox.ollama")

def _split_wav_bytes(audio_bytes: bytes, max_chunk_duration: float = 30.0) -> List[bytes]:
    """Splits WAV bytes into chunks of at most max_chunk_duration seconds.
    
    If the bytes do not represent a valid WAV file, returns the original bytes as a single chunk.
    """
    import io
    import wave
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
        
        duration = nframes / framerate
        if duration <= max_chunk_duration:
            return [audio_bytes]
        
        # Calculate frames per chunk
        frames_per_chunk = int(max_chunk_duration * framerate)
        chunks = []
        
        for i in range(0, nframes, frames_per_chunk):
            wav_in.setpos(i)
            chunk_frames = wav_in.readframes(frames_per_chunk)
            if not chunk_frames:
                break
            
            # Write chunk to WAV bytes in memory
            chunk_io = io.BytesIO()
            wav_out = wave.open(chunk_io, 'wb')
            with wav_out:
                wav_out.setnchannels(nchannels)
                wav_out.setsampwidth(sampwidth)
                wav_out.setframerate(framerate)
                wav_out.writeframes(chunk_frames)
            
            chunks.append(chunk_io.getvalue())
            
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

    async def process_audio(
        self,
        audio_bytes: bytes,
        output_format: str,  # "note" or "message"
        target_language: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> Tuple[str, str, str, str]:
        """Sends audio data base64-encoded to Gemma 4 via Ollama chat API.
        
        Returns:
            Tuple of (raw_transcript, source_language, target_language, transformed_text)
        """
        # Determine if chunking is needed (Gemma 4 audio limit is 30s)
        chunks = _split_wav_bytes(audio_bytes, max_chunk_duration=30.0)
        
        if len(chunks) == 1:
            # Encode WAV to base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Build prompt based on requested format
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

            prompt = f"""Analyze the attached audio clip and perform the following tasks:
1. Transcribe the audio verbatim. Save this text into the 'raw_transcript' field.
2. Automatically detect the source language of the speech. Save it in the 'source_language' field (e.g. 'English', 'Spanish', 'French').
3. Determine the target language for translation based on the user's instructions (like translation requested via dropdown or custom instructions) or spoken instructions. If no translation is requested/needed, the target language is the same as the source language. Save the name of this target language in the 'target_language' field (e.g. 'English', 'Spanish', 'French').
4. Transform the transcription into {format_description}, and ensure it is written/translated in the determined target language. Save the resulting finalized and translated output in the 'transformed_text' field.
5. {translation_clause}
6. {extra_instructions}

Return the response ONLY as a JSON object with the following fields:
{{
  "raw_transcript": "the verbatim transcription",
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
                        "content": prompt,
                        "images": [audio_b64]
                    }
                ],
                "format": "json",
                "stream": False
            }

            url = f"{self.host}/api/chat"
            logger.info(f"Sending audio processing request to Ollama ({self.model}) at {url}")

            async with httpx.AsyncClient() as client:
                try:
                    # Transcribing and processing locally might take some time, set high timeout (e.g., 90s)
                    response = await client.post(url, json=payload, timeout=90.0)
                    
                    if response.status_code != 200:
                        logger.error(f"Ollama returned status {response.status_code}: {response.text}")
                        raise httpx.HTTPStatusError(
                            f"Ollama service error: status {response.status_code}",
                            request=response.request,
                            response=response
                        )
                    
                    result_json = response.json()
                    content = result_json.get("message", {}).get("content", "")
                    
                    # Parse JSON output from model
                    try:
                        data = json.loads(content)
                        raw_transcript = data.get("raw_transcript", "").strip()
                        source_language = data.get("source_language", "").strip()
                        target_lang_detected = data.get("target_language", "").strip()
                        transformed_text = data.get("transformed_text", "").strip()
                        
                        # Fallback for target_language if not provided
                        if not target_lang_detected:
                            target_lang_detected = target_language if (target_language and target_language.lower() not in ["auto", "same", "none"]) else source_language
                        
                        return raw_transcript, source_language, target_lang_detected, transformed_text
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse model response as JSON: {content}. Trying regex or fallback.")
                        # Fallback if JSON parsing fails but content is present
                        fallback_target = target_language if (target_language and target_language.lower() not in ["auto", "same", "none"]) else "Unknown"
                        return content, "Unknown", fallback_target, content

                except httpx.RequestError as e:
                    logger.error(f"Connection error to Ollama: {e}")
                    raise
        else:
            logger.info(f"Audio duration is longer than 30 seconds. Split into {len(chunks)} chunks.")
            raw_transcripts = []
            source_languages = []
            
            for index, chunk_bytes in enumerate(chunks):
                chunk_b64 = base64.b64encode(chunk_bytes).decode('utf-8')
                chunk_prompt = """Analyze the attached audio clip. 
1. Transcribe the audio verbatim. Save this in 'raw_transcript' field.
2. Detect the source language. Save this in 'source_language' field (e.g. 'English', 'Spanish', 'French').

Return the response ONLY as a JSON object with the following fields:
{
  "raw_transcript": "the verbatim transcription of this audio segment",
  "source_language": "detected source language"
}
Do not add any text before or after the JSON block."""

                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": chunk_prompt,
                            "images": [chunk_b64]
                        }
                    ],
                    "format": "json",
                    "stream": False
                }
                
                url = f"{self.host}/api/chat"
                logger.info(f"Processing audio chunk {index + 1}/{len(chunks)} with Ollama")
                
                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.post(url, json=payload, timeout=90.0)
                        if response.status_code != 200:
                            logger.error(f"Ollama returned status {response.status_code} for chunk {index + 1}")
                            raise httpx.HTTPStatusError(
                                f"Ollama service error: status {response.status_code}",
                                request=response.request,
                                response=response
                            )
                        
                        result_json = response.json()
                        content = result_json.get("message", {}).get("content", "")
                        
                        try:
                            data = json.loads(content)
                            chunk_transcript = data.get("raw_transcript", "").strip()
                            chunk_source_lang = data.get("source_language", "").strip()
                        except json.JSONDecodeError:
                            chunk_transcript = content.strip()
                            chunk_source_lang = "Unknown"
                        
                        if chunk_transcript:
                            raw_transcripts.append(chunk_transcript)
                        if chunk_source_lang:
                            source_languages.append(chunk_source_lang)
                    except httpx.RequestError as e:
                        logger.error(f"Connection error to Ollama for chunk {index + 1}: {e}")
                        raise
            
            combined_transcript = " ".join(raw_transcripts)
            from collections import Counter
            detected_source_language = Counter(source_languages).most_common(1)[0][0] if source_languages else "English"
            
            # Format description for note/message
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
            
            # Refine combined text
            refine_prompt = f"""Analyze the following transcription:
"{combined_transcript}"

Perform the following tasks:
1. Automatically detect the source language of the speech from the transcription text. Save it in the 'source_language' field (e.g. 'English', 'Spanish', 'French').
2. Determine the target language for translation based on the user's instructions (like translation requested: '{target_language}') or custom instructions: '{custom_instructions}'. If no translation is requested/needed, the target language is the same as the source language. Save the name of this target language in the 'target_language' field (e.g. 'English', 'Spanish', 'French').
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

            refine_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": refine_prompt
                    }
                ],
                "format": "json",
                "stream": False
            }
            
            logger.info("Refining combined transcripts with Ollama")
            url = f"{self.host}/api/chat"
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(url, json=refine_payload, timeout=90.0)
                    if response.status_code != 200:
                        logger.error(f"Ollama returned status {response.status_code} during refinement")
                        raise httpx.HTTPStatusError(
                            f"Ollama service error during refinement: status {response.status_code}",
                            request=response.request,
                            response=response
                        )
                    
                    result_json = response.json()
                    content = result_json.get("message", {}).get("content", "")
                    
                    try:
                        data = json.loads(content)
                        source_language = data.get("source_language", "").strip()
                        target_lang_detected = data.get("target_language", "").strip()
                        transformed_text = data.get("transformed_text", "").strip()
                    except json.JSONDecodeError:
                        source_language = detected_source_language
                        target_lang_detected = target_language if (target_language and target_language.lower() not in ["auto", "same", "none"]) else source_language
                        transformed_text = content.strip()
                    
                    return combined_transcript, source_language, target_lang_detected, transformed_text
                except httpx.RequestError as e:
                    logger.error(f"Connection error to Ollama during refinement: {e}")
                    raise
