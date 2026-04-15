"""Lightweight Gemma 4 E4B service for TransLean alternative pipeline."""
from __future__ import annotations

import io
import logging
import os
from functools import lru_cache

import librosa
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_ID = os.getenv("GEMMA_MODEL", "google/gemma-4-E4B-it")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

app = FastAPI(title="translean-gemma4-server")


class InferenceResult(BaseModel):
    text: str
    translation: str
    language: str


@lru_cache(maxsize=1)
def _load_runtime() -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    logger.info("Loading Gemma model %s on %s", MODEL_ID, DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE != "cuda":
        model.to(DEVICE)
    model.eval()
    return processor, model


def _run_prompt(audio: tuple[list[float], int], prompt: str) -> str:
    processor, model = _load_runtime()
    audio_array, sr = audio
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": {"array": audio_array, "sampling_rate": sr}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[formatted], audios=[audio_array], sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = processor.batch_decode(generated, skip_special_tokens=True)
    return (decoded[0] if decoded else "").strip()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/transcribe-translate", response_model=InferenceResult)
async def transcribe_translate(
    audio: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: str = Form("auto"),
    model: str = Form(MODEL_ID),
    keep_alive: int = Form(900),
) -> InferenceResult:
    if model and model != MODEL_ID:
        logger.warning("Request model %s differs from loaded model %s; using loaded model.", model, MODEL_ID)

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="audio payload is empty")

    try:
        waveform, sr = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to decode audio: {exc}") from exc

    if waveform.size == 0:
        raise HTTPException(status_code=400, detail="decoded audio is empty")

    audio_input = (waveform.tolist(), sr)
    src_prompt = (
        f"Transcribe exactly what is spoken in this audio. The source language is {source_language}. "
        "Return only transcription text."
    )
    trans_prompt = (
        f"Translate this audio to {target_language}. Keep meaning and tone. Return only translation text."
    )

    text = _run_prompt(audio_input, src_prompt)
    translation = _run_prompt(audio_input, trans_prompt)

    return InferenceResult(text=text, translation=translation, language=source_language)
