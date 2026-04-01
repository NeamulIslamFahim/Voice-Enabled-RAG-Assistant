from __future__ import annotations

import io
import os
import shutil
import tempfile
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

DEFAULT_MODEL_NAME = "small"
DEFAULT_CPU_THREADS = 4
DEFAULT_SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.012
SILENCE_PADDING_SECONDS = 0.12

# Hugging Face warns on Windows when symlinks are unavailable in the cache.
# Disable the warning so the app stays quiet while still working normally.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _write_temp_audio(data: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        return Path(handle.name)


def _read_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        audio = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError("Only 16-bit WAV files are supported without ffmpeg.")

    data = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data, sample_rate


def _resample_audio(samples: np.ndarray, source_rate: int, target_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)

    duration = samples.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _trim_silence(samples: np.ndarray, threshold: float = SILENCE_THRESHOLD, padding_seconds: float = SILENCE_PADDING_SECONDS) -> np.ndarray:
    if samples.size == 0:
        return samples

    energy = np.abs(samples)
    mask = energy > threshold
    if not np.any(mask):
        return samples

    indices = np.flatnonzero(mask).tolist()
    if not indices:
        return samples

    pad = int(DEFAULT_SAMPLE_RATE * padding_seconds)
    start = max(0, indices[0] - pad)
    end = min(samples.size, indices[-1] + pad + 1)
    return samples[start:end]


def _normalize_audio(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples / peak
    return np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)


def _write_wav_samples(samples: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
        path = Path(handle.name)

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return path


def _prepare_speech_audio(audio_bytes: bytes, audio_name: str) -> np.ndarray:
    suffix = Path(str(audio_name).lower()).suffix or ".wav"

    if suffix in {".wav", ".wave"}:
        samples, sample_rate = _read_wav_bytes(audio_bytes)
        samples = _resample_audio(samples, sample_rate)
        samples = _trim_silence(samples)
        return _normalize_audio(samples)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Whisper needs ffmpeg to decode this audio format. Install ffmpeg or upload a WAV file."
        )

    input_path = _write_temp_audio(audio_bytes, suffix)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
        output_path = Path(handle.name)

    try:
        import subprocess

        command = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            "-vn",
            str(output_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with output_path.open("rb") as handle:
            converted_bytes = handle.read()
        samples, sample_rate = _read_wav_bytes(converted_bytes)
        samples = _resample_audio(samples, sample_rate)
        samples = _trim_silence(samples)
        return _normalize_audio(samples)
    finally:
        try:
            input_path.unlink()
        except OSError:
            pass
        try:
            output_path.unlink()
        except OSError:
            pass


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


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = str(language).strip().lower()
    if normalized in {"", "auto", "detect", "default"}:
        return None
    return normalized


def _run_transcription(
    backend: str,
    model,
    clean_audio: np.ndarray,
    *,
    language: str | None,
    task: str,
    temperature: float = 0.5,
) -> tuple[str, str | None]:
    language_arg = _normalize_language(language)
    task_arg = (task or "transcribe").strip().lower()
    if task_arg not in {"transcribe", "translate"}:
        task_arg = "transcribe"

    if backend == "faster_whisper":
        local_path = _write_wav_samples(clean_audio)
        try:
            segments, info = model.transcribe(
                str(local_path),
                language=language_arg,
                task=task_arg,
                temperature=temperature,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
            detected_language = getattr(info, "language", None)
            return text, detected_language
        finally:
            try:
                os.unlink(local_path)
            except OSError:
                pass

    result = model.transcribe(
        clean_audio,
        fp16=False,
        language=language_arg,
        task=task_arg,
        temperature=temperature,
        best_of=3,
        beam_size=5,
        condition_on_previous_text=False,
        no_speech_threshold=0.35,
        verbose=False,
    )
    text = (result.get("text") or "").strip()
    detected_language = result.get("language")
    return text, detected_language

def transcribe_audio(
    audio_bytes: bytes,
    audio_name: str = "voice.wav",
    model_name: str = DEFAULT_MODEL_NAME,
    language: str = "en",
    task: str = "transcribe",
) -> str:
    """Transcribe audio locally with Whisper."""

    backend, model = _load_model(model_name)
    language_arg = _normalize_language(language)
    task_arg = (task or "transcribe").strip().lower()
    if task_arg not in {"transcribe", "translate", "auto"}:
        task_arg = "transcribe"

    try:
        clean_audio = _prepare_speech_audio(audio_bytes, audio_name)

        if task_arg == "auto":
            detected_text, detected_language = _run_transcription(
                backend,
                model,
                clean_audio,
                language=None,
                task="transcribe",
                temperature=0.5,
            )
            if not detected_text:
                return ""
            if detected_language and detected_language.lower() != "en":
                translated_text, _ = _run_transcription(
                    backend,
                    model,
                    clean_audio,
                    language=detected_language,
                    task="translate",
                    temperature=0.5,
                )
                return translated_text or detected_text
            return detected_text

        text, _detected_language = _run_transcription(
            backend,
            model,
            clean_audio,
            language=language_arg,
            task=task_arg,
            temperature=0.5,
        )
        return text
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Whisper needs ffmpeg to decode this audio format. Install ffmpeg or upload a WAV file."
        ) from exc

