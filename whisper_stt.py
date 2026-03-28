from __future__ import annotations

import io
import os
import tempfile
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_MODEL_NAME = "tiny.en"
DEFAULT_CPU_THREADS = 4

# Hugging Face warns on Windows when symlinks are unavailable in the cache.
# Disable the warning so the app stays quiet while still working normally.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _write_temp_audio(data: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        return Path(handle.name)


def _read_wav_bytes(audio_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        audio = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError("Only 16-bit WAV files are supported without ffmpeg.")

    data = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    try:
        from faster_whisper import WhisperModel  # type: ignore

        return ("faster_whisper", WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=DEFAULT_CPU_THREADS))
    except Exception:
        try:
            import whisper  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "Whisper is not installed. Install `faster-whisper` or `openai-whisper` to enable audio transcription."
            ) from exc

        return ("openai_whisper", whisper.load_model(model_name))


def transcribe_audio(
    audio_bytes: bytes,
    audio_name: str = "voice.wav",
    model_name: str = DEFAULT_MODEL_NAME,
    language: str = "en",
) -> str:
    """Transcribe audio locally with Whisper."""

    data = audio_bytes
    name = str(audio_name).lower()
    suffix = Path(name).suffix or ".wav"
    backend, model = _load_model(model_name)
    language_arg = None if language == "auto" else language

    try:
        if backend == "faster_whisper":
            local_path = _write_temp_audio(data, suffix)
            try:
                segments, _info = model.transcribe(
                    str(local_path),
                    language=language_arg,
                    beam_size=1,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                return " ".join(segment.text.strip() for segment in segments if segment.text).strip()
            finally:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass

        if suffix in {".wav", ".wave"}:
            audio = _read_wav_bytes(data)
            result = model.transcribe(
                audio,
                fp16=False,
                language=language_arg,
                temperature=0,
                best_of=1,
                beam_size=1,
                condition_on_previous_text=False,
                verbose=False,
            )
        else:
            local_path = _write_temp_audio(data, suffix)
            try:
                result = model.transcribe(
                    str(local_path),
                    fp16=False,
                    language=language_arg,
                    temperature=0,
                    best_of=1,
                    beam_size=1,
                    condition_on_previous_text=False,
                    verbose=False,
                )
            finally:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass
        return (result.get("text") or "").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Whisper needs ffmpeg to decode this audio format. Install ffmpeg or upload a WAV file."
        ) from exc
