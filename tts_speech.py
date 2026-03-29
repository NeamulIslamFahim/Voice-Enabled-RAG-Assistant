from __future__ import annotations

import hashlib
import platform
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
        return None


def _write_with_sapi(text: str, output_path: Path) -> bool:
    if platform.system().lower() != "windows":
        return False

    try:
        import win32com.client  # type: ignore
    except Exception:
        try:
            import comtypes.client  # type: ignore
        except Exception:
            return False

        try:
            from comtypes.gen import SpeechLib  # type: ignore
        except Exception:
            SpeechLib = None  # type: ignore

        speaker = comtypes.client.CreateObject("SAPI.SpVoice")
        stream = comtypes.client.CreateObject("SAPI.SpFileStream")
        mode = 3 if SpeechLib is None else SpeechLib.SpeechStreamFileMode.SSFMCreateForWrite
        stream.Open(str(output_path), mode)
        previous_stream = speaker.AudioOutputStream
        speaker.AudioOutputStream = stream
        speaker.Speak(text)
        stream.Close()
        speaker.AudioOutputStream = previous_stream
        return True

    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    stream.Open(str(output_path), 3)
    previous_stream = speaker.AudioOutputStream
    speaker.AudioOutputStream = stream
    speaker.Speak(text)
    stream.Close()
    speaker.AudioOutputStream = previous_stream
    return True


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

    temp_wav = cache_path.with_suffix(".wav")
    engine = _get_engine()
    produced = False
    if engine is not None:
        engine.save_to_file(text, str(temp_wav))
        try:
            engine.runAndWait()
            produced = temp_wav.exists()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    if not produced:
        produced = _write_with_sapi(text, temp_wav)

    if not temp_wav.exists():
        if not produced:
            raise RuntimeError(
                "Text-to-speech could not start. On Windows, a local SAPI voice must be installed; "
                "on Linux or WSL, install `espeak` or `espeak-ng`."
            )
        raise RuntimeError("Text-to-speech did not produce an audio file.")

    data = temp_wav.read_bytes()
    cache_path.write_bytes(data)
    try:
        temp_wav.unlink()
    except OSError:
        pass
    return data
