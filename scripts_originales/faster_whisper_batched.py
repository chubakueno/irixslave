"""Divide `BatchedInferencePipeline.transcribe` de faster-whisper en dos fases:

    prepare(pipeline, audio)  -> dict   # SOLO CPU: decode + VAD + collect_chunks + log-mel
    generate(pipeline, prepared, ...)   # GPU: _batched_segments_generator

Es una copia fiel del cuerpo de `BatchedInferencePipeline.transcribe`
(faster-whisper 1.2.1) partida en el límite CPU/GPU, para poder correr la fase
CPU por adelantado en otro hilo (mientras la GPU procesa otro job) SIN cambiar
el algoritmo: la salida es idéntica a `pipeline.transcribe(audio, vad_filter=True,
vad_parameters=...)`.

Acoplado a la versión exacta de faster-whisper; si no coincide, el llamador debe
caer al camino normal.
"""

from __future__ import annotations

from typing import Any

import faster_whisper
import numpy as np
from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import (
    TranscriptionInfo,
    TranscriptionOptions,
    get_suppressed_tokens,
    restore_speech_timestamps,
)
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

SUPPORTED_FASTER_WHISPER = "1.2.1"

# Puntuaciones por defecto de BatchedInferencePipeline.transcribe (1.2.1).
_PREPEND_PUNCTUATIONS = "\"'“¿([{-"
_APPEND_PUNCTUATIONS = "\"'.。,，!！?？:：”)]}、"


def is_supported() -> bool:
    return faster_whisper.__version__ == SUPPORTED_FASTER_WHISPER


def prepare(
    pipeline,
    audio,
    *,
    vad_parameters: dict | None = None,
) -> dict[str, Any]:
    """Fase CPU: decodifica (si hace falta), corre el VAD de Silero, agrupa en
    chunks de <=30 s (collect_chunks) y calcula el log-mel de cada chunk.
    Devuelve todo lo que `generate` necesita. Equivale a las líneas 123-213 de
    BatchedInferencePipeline.transcribe."""
    model = pipeline.model
    sampling_rate = model.feature_extractor.sampling_rate

    import time as _time

    t0 = _time.perf_counter()
    if not isinstance(audio, np.ndarray):
        audio = decode_audio(audio, sampling_rate=sampling_rate)
    t_decode = _time.perf_counter()
    duration = audio.shape[0] / sampling_rate

    chunk_length = model.feature_extractor.chunk_length

    if vad_parameters is None:
        vad_options = VadOptions(
            max_speech_duration_s=chunk_length, min_silence_duration_ms=160
        )
    else:
        params = dict(vad_parameters)
        params.pop("max_speech_duration_s", None)
        vad_options = VadOptions(**params, max_speech_duration_s=chunk_length)

    clip_timestamps = get_speech_timestamps(audio, vad_options)
    audio_chunks, chunks_metadata = collect_chunks(
        audio, clip_timestamps, max_duration=chunk_length
    )
    t_vad = _time.perf_counter()

    duration_after_vad = (
        sum((seg["end"] - seg["start"]) for seg in clip_timestamps) / sampling_rate
    )

    features = (
        [model.feature_extractor(chunk)[..., :-1] for chunk in audio_chunks]
        if duration_after_vad
        else []
    )
    t_mel = _time.perf_counter()

    return {
        "features": features,
        "chunks_metadata": chunks_metadata,
        "clip_timestamps": clip_timestamps,  # en muestras, como el camino interno
        "duration": duration,
        "duration_after_vad": duration_after_vad,
        "sampling_rate": sampling_rate,
        "timings": {
            "decode": t_decode - t0,
            "vad": t_vad - t_decode,
            "mel": t_mel - t_vad,
            "total": t_mel - t0,
        },
    }


def generate(
    pipeline,
    prepared: dict[str, Any],
    *,
    language: str,
    task: str = "transcribe",
    beam_size: int = 5,
    temperature: float = 0.0,
    compression_ratio_threshold: float | None = 2.4,
    log_prob_threshold: float | None = -1.0,
    no_speech_threshold: float | None = 0.6,
    word_timestamps: bool = False,
    without_timestamps: bool = True,
    batch_size: int = 8,
    vad_parameters: dict | None = None,
):
    """Fase GPU: construye tokenizer/opciones e invoca
    `_batched_segments_generator`. Equivale a las líneas 215-324. `language` debe
    venir dado (no hace detección). Devuelve `(segments, info)` como transcribe."""
    model = pipeline.model
    sampling_rate = prepared["sampling_rate"]
    features_list = prepared["features"]

    tokenizer = Tokenizer(
        model.hf_tokenizer, model.model.is_multilingual, task=task, language=language
    )

    features = (
        np.stack([pad_or_trim(f) for f in features_list]) if features_list else []
    )

    options = TranscriptionOptions(
        beam_size=beam_size,
        best_of=beam_size,
        patience=1,
        length_penalty=1,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        temperatures=[temperature],
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
        prepend_punctuations=_PREPEND_PUNCTUATIONS,
        append_punctuations=_APPEND_PUNCTUATIONS,
        max_new_tokens=None,
        hotwords=None,
        word_timestamps=word_timestamps,
        hallucination_silence_threshold=None,
        condition_on_previous_text=False,
        clip_timestamps=prepared["clip_timestamps"],
        prompt_reset_on_temperature=0.5,
        multilingual=False,
        without_timestamps=without_timestamps,
        max_initial_timestamp=0.0,
    )

    vad_options = None
    if vad_parameters is not None:
        params = dict(vad_parameters)
        params.pop("max_speech_duration_s", None)
        vad_options = VadOptions(
            **params, max_speech_duration_s=model.feature_extractor.chunk_length
        )

    info = TranscriptionInfo(
        language=language,
        language_probability=1,
        duration=prepared["duration"],
        duration_after_vad=prepared["duration_after_vad"],
        transcription_options=options,
        vad_options=vad_options,
        all_language_probs=None,
    )

    segments = pipeline._batched_segments_generator(
        features,
        tokenizer,
        prepared["chunks_metadata"],
        batch_size,
        options,
        False,
    )
    # El camino interno-VAD siempre restaura los timestamps (clip_timestamps_provided=False).
    segments = restore_speech_timestamps(segments, prepared["clip_timestamps"], sampling_rate)
    return segments, info
