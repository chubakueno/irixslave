from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import BinaryIO

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
LOCAL_FFMPEG = PACKAGE_ROOT / "02_analysis" / "tools" / "ffmpeg" / "ffmpeg.exe"


def ffmpeg_binary() -> Path:
    configured = os.environ.get("FFMPEG_BINARY")
    candidates = [Path(configured).expanduser() if configured else None, LOCAL_FFMPEG]
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        candidates.append(Path(system_ffmpeg))
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "No se encontró ffmpeg. Se esperaba en "
        f"{LOCAL_FFMPEG} o en la variable FFMPEG_BINARY."
    )


def decode_audio(
    input_file: str | Path | BinaryIO,
    sampling_rate: int = 16_000,
    split_stereo: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Decodifica con FFmpeg sin cargar las DLL no firmadas de PyAV."""
    channels = 2 if split_stereo else 1
    command = [
        str(ffmpeg_binary()),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    stdin_data: bytes | None = None
    if isinstance(input_file, (str, Path)):
        command.extend(["-i", str(input_file)])
    else:
        position = input_file.tell() if hasattr(input_file, "tell") else None
        stdin_data = input_file.read()
        if position is not None and hasattr(input_file, "seek"):
            input_file.seek(position)
        command.extend(["-i", "pipe:0"])
    command.extend(
        [
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            str(channels),
            "-ar",
            str(sampling_rate),
            "pipe:1",
        ]
    )
    completed = subprocess.run(
        command,
        input=stdin_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg no pudo decodificar el audio: {detail}")
    audio = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if split_stereo:
        return audio[0::2].copy(), audio[1::2].copy()
    return audio


def audio_duration_seconds(path: Path, sampling_rate: int = 16_000) -> float:
    return float(decode_audio(path, sampling_rate=sampling_rate).shape[0] / sampling_rate)


def install_faster_whisper_audio_compat() -> None:
    """Sustituye solo el decoder PyAV cuando Windows bloquea sus extensiones."""
    force_ffmpeg = os.environ.get("FORCE_FFMPEG_AUDIO", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or (os.name == "nt" and LOCAL_FFMPEG.is_file())
    try:
        import av  # noqa: F401
        av_available = True
    except (ImportError, OSError):
        av_available = False

    if av_available and not force_ffmpeg:
        return

    if not av_available:
        av_stub = types.ModuleType("av")
        error_namespace = types.SimpleNamespace(InvalidDataError=RuntimeError)
        av_stub.error = error_namespace  # type: ignore[attr-defined]
        sys.modules["av"] = av_stub

    import faster_whisper.audio as faster_audio
    import faster_whisper.transcribe as faster_transcribe

    faster_audio.decode_audio = decode_audio
    faster_transcribe.decode_audio = decode_audio
