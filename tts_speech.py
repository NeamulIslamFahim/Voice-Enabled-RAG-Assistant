from __future__ import annotations

import hashlib
import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _get_engine():
    try:
        import pyttsx3  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError("pyttsx3 is not installed. Install `pyttsx3` to enable text-to-speech output.") from exc

    try:
        return pyttsx3.init()
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "Text-to-speech could not start. If you're on Linux or WSL, install `espeak` or `espeak-ng`. "
            "If you're on Windows, make sure a Windows speech voice is available."
        ) from exc


def _voice_cache_path(text: str) -> Path:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    cache_dir = Path(tempfile.gettempdir()) / "voice_rag_tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{digest}.wav"


def text_to_speech_bytes(text: str) -> bytes:
    text = text.strip()
    if not text:
        return b""

    cache_path = _voice_cache_path(text)
    if cache_path.exists():
        return cache_path.read_bytes()

    engine = _get_engine()
    temp_wav = cache_path.with_suffix(".wav")
    engine.save_to_file(text, str(temp_wav))
    try:
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass

    if not temp_wav.exists():
        raise RuntimeError("Text-to-speech did not produce an audio file.")

    data = temp_wav.read_bytes()
    cache_path.write_bytes(data)
    try:
        temp_wav.unlink()
    except OSError:
        pass
    return data
