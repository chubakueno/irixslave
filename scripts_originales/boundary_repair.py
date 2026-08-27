from __future__ import annotations

import concurrent.futures
import math
import time
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

from audio_compat import install_faster_whisper_audio_compat

install_faster_whisper_audio_compat()
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import Segment, Word
from faster_whisper.vad import VadOptions, get_speech_timestamps


SAMPLING_RATE = 16_000
MAX_BLOCK_SECONDS = 30.0
WINDOW_BEFORE_SECONDS = 12.0
WINDOW_AFTER_SECONDS = 6.0
ANCHOR_CONTEXT_SECONDS = 1.5
SEAM_LOOKBACK_SECONDS = 12.0
MATCH_TOLERANCE_SECONDS = 1.2
BOUNDARY_TOLERANCE_SECONDS = 2.0
ANCHOR_DISTANCE_SECONDS = 8.0
MIN_MISSING_WORDS = 3
MIN_AVERAGE_PROBABILITY = 0.45
EDGE_MIN_AVERAGE_PROBABILITY = 0.70
REPAIR_BEAM_SIZE = 1
MIN_LEFT_COVERAGE_GAP_SECONDS = 0.8
DEFAULT_WINDOW_WORKERS = 2


@dataclass(frozen=True)
class BoundaryWindow:
    boundary: float
    start: float
    end: float


@dataclass(frozen=True)
class RecoveredSpan:
    boundary: float
    words: tuple[Word, ...]
    left_anchor: str
    right_anchor: str
    average_probability: float


@dataclass(frozen=True)
class RepairStats:
    boundaries: int
    windows_decoded: int
    spans_added: int
    words_added: int
    elapsed_seconds: float


def normalize_word(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in decomposed if character.isalnum())


def _word_midpoint(word: Word) -> float:
    return (float(word.start) + float(word.end)) / 2.0


def _flatten_words(segments: Iterable[Any]) -> list[Word]:
    words: list[Word] = []
    for segment in segments:
        words.extend(getattr(segment, "words", None) or [])
    return sorted(words, key=lambda word: (float(word.start), float(word.end)))


def packed_block_boundaries(
    chunks: Sequence[dict[str, int]],
    *,
    sampling_rate: int = SAMPLING_RATE,
    max_block_seconds: float = MAX_BLOCK_SECONDS,
    include_final: bool = False,
) -> list[float]:
    """Return only internal boundaries created while packing VAD speech.

    The final block end is intentionally excluded: the documented defect occurs
    where one packed block hands off to the next, and excluding the final tail
    avoids an unsafe one-sided repair without a right-hand anchor.
    """

    max_samples = int(max_block_seconds * sampling_rate)
    packed: list[dict[str, int]] = []
    packed_duration = 0
    boundaries: list[float] = []

    for chunk in chunks:
        duration = int(chunk["end"]) - int(chunk["start"])
        if packed and packed_duration + duration > max_samples:
            boundaries.append(float(packed[-1]["end"]) / sampling_rate)
            packed = [chunk]
            packed_duration = duration
        else:
            packed.append(chunk)
            packed_duration += duration

    if include_final and packed:
        final_end = float(packed[-1]["end"]) / sampling_rate
        if not boundaries or abs(boundaries[-1] - final_end) > 1e-6:
            boundaries.append(final_end)
    return boundaries


def detect_packed_boundaries(
    audio: np.ndarray,
    *,
    sampling_rate: int = SAMPLING_RATE,
    max_block_seconds: float = MAX_BLOCK_SECONDS,
    min_silence_duration_ms: int = 700,
    include_final: bool = False,
) -> list[float]:
    chunks = get_speech_timestamps(
        audio,
        VadOptions(
            max_speech_duration_s=max_block_seconds,
            min_silence_duration_ms=min_silence_duration_ms,
        ),
    )
    return packed_block_boundaries(
        chunks,
        sampling_rate=sampling_rate,
        max_block_seconds=max_block_seconds,
        include_final=include_final,
    )


def plan_boundary_windows(
    boundaries: Sequence[float],
    audio_duration: float,
    *,
    before_seconds: float = WINDOW_BEFORE_SECONDS,
    after_seconds: float = WINDOW_AFTER_SECONDS,
) -> list[BoundaryWindow]:
    windows: list[BoundaryWindow] = []
    for boundary in boundaries:
        start = max(0.0, float(boundary) - before_seconds)
        end = min(float(audio_duration), float(boundary) + after_seconds)
        if end > start:
            windows.append(BoundaryWindow(float(boundary), start, end))
    return windows


def plan_adaptive_windows(
    boundaries: Sequence[float],
    audio_duration: float,
    existing: Sequence[Word],
) -> list[BoundaryWindow]:
    """Keep only the local audio needed to bracket each suspicious seam."""

    existing = sorted(existing, key=lambda word: (float(word.start), float(word.end)))
    windows: list[BoundaryWindow] = []
    for boundary in boundaries:
        left = [word for word in existing if float(word.end) <= float(boundary) + 0.25]
        right = [word for word in existing if float(word.start) >= float(boundary) - 0.25]
        if not left:
            continue
        left_anchor = max(left, key=lambda word: float(word.end))
        left_gap = float(boundary) - float(left_anchor.end)
        if MIN_LEFT_COVERAGE_GAP_SECONDS <= left_gap <= WINDOW_BEFORE_SECONDS - 0.5:
            start = float(left_anchor.start) - ANCHOR_CONTEXT_SECONDS
        else:
            start = float(boundary) - SEAM_LOOKBACK_SECONDS

        if right:
            right_anchor = min(right, key=lambda word: float(word.start))
            end = float(right_anchor.end) + ANCHOR_CONTEXT_SECONDS
        else:
            end = float(boundary) + 0.25

        start = max(0.0, start)
        end = min(float(audio_duration), end)
        if end > start:
            windows.append(BoundaryWindow(float(boundary), start, end))
    return windows


def suspicious_boundaries(
    boundaries: Sequence[float],
    existing: Sequence[Word],
    *,
    min_left_gap_seconds: float = MIN_LEFT_COVERAGE_GAP_SECONDS,
    max_right_anchor_seconds: float = WINDOW_AFTER_SECONDS,
) -> list[float]:
    """Select boundaries whose batched words expose a repairable coverage gap.

    Decode only production-observable risk shapes: a coverage gap, a word that
    straddles the packed seam, or an uncovered final tail. This avoids
    redecoding every healthy boundary and does not depend on names or keywords.
    """

    existing = sorted(existing, key=lambda word: (float(word.start), float(word.end)))
    suspicious: list[float] = []
    for boundary_index, boundary in enumerate(boundaries):
        left = [word for word in existing if float(word.end) <= float(boundary) + 0.25]
        right = [word for word in existing if float(word.start) >= float(boundary) - 0.25]
        if not left:
            continue
        left_anchor = max(left, key=lambda word: float(word.end))
        left_gap = float(boundary) - float(left_anchor.end)
        if not right:
            if boundary_index == len(boundaries) - 1 and left_gap >= min_left_gap_seconds:
                suspicious.append(float(boundary))
            continue
        right_anchor = min(right, key=lambda word: float(word.start))
        right_distance = float(right_anchor.start) - float(boundary)
        visible_gap = left_gap >= min_left_gap_seconds
        continuous_seam = (
            -0.25 <= left_gap <= 0.0 and -0.25 <= right_distance <= 0.25
        )
        if right_distance <= max_right_anchor_seconds and (visible_gap or continuous_seam):
            suspicious.append(float(boundary))
    return suspicious


def _match_recovered_to_existing(
    existing: Sequence[Word],
    recovered: Sequence[Word],
    *,
    tolerance_seconds: float = MATCH_TOLERANCE_SECONDS,
) -> list[bool]:
    matched = [False] * len(recovered)
    used_existing: set[int] = set()

    for recovered_index, recovered_word in enumerate(recovered):
        token = normalize_word(recovered_word.word)
        if not token:
            continue
        candidates: list[tuple[float, int]] = []
        recovered_midpoint = _word_midpoint(recovered_word)
        for existing_index, existing_word in enumerate(existing):
            if existing_index in used_existing:
                continue
            if normalize_word(existing_word.word) != token:
                continue
            delta = abs(_word_midpoint(existing_word) - recovered_midpoint)
            if delta <= tolerance_seconds:
                candidates.append((delta, existing_index))
        if candidates:
            _, existing_index = min(candidates)
            matched[recovered_index] = True
            used_existing.add(existing_index)

    return matched


def recoverable_spans(
    existing: Sequence[Word],
    recovered: Sequence[Word],
    boundary: float,
    *,
    min_missing_words: int = MIN_MISSING_WORDS,
    min_average_probability: float = MIN_AVERAGE_PROBABILITY,
    boundary_tolerance_seconds: float = BOUNDARY_TOLERANCE_SECONDS,
    anchor_distance_seconds: float = ANCHOR_DISTANCE_SECONDS,
    allow_left_edge: bool = False,
    allow_right_edge: bool = False,
) -> list[RecoveredSpan]:
    """Find missing runs that are safely bracketed by matching anchor words."""

    if not recovered:
        return []

    recovered = sorted(recovered, key=lambda word: (float(word.start), float(word.end)))
    existing = sorted(existing, key=lambda word: (float(word.start), float(word.end)))
    matched = _match_recovered_to_existing(existing, recovered)
    spans: list[RecoveredSpan] = []
    index = 0

    while index < len(recovered):
        if matched[index] or not normalize_word(recovered[index].word):
            index += 1
            continue

        start_index = index
        while (
            index < len(recovered)
            and not matched[index]
            and normalize_word(recovered[index].word)
        ):
            index += 1
        end_index = index
        run = list(recovered[start_index:end_index])

        if len(run) < min_missing_words:
            continue

        left_index = start_index - 1
        while left_index >= 0 and not matched[left_index]:
            left_index -= 1
        right_index = end_index
        while right_index < len(recovered) and not matched[right_index]:
            right_index += 1
        missing_left_anchor = left_index < 0
        missing_right_anchor = right_index >= len(recovered)
        if missing_left_anchor and not (allow_left_edge and start_index == 0):
            continue
        if missing_right_anchor and not (allow_right_edge and end_index == len(recovered)):
            continue
        if missing_left_anchor and missing_right_anchor:
            continue

        left_anchor = recovered[left_index] if not missing_left_anchor else None
        right_anchor = recovered[right_index] if not missing_right_anchor else None
        if left_anchor and float(run[0].start) - float(left_anchor.end) > anchor_distance_seconds:
            continue
        if right_anchor and float(right_anchor.start) - float(run[-1].end) > anchor_distance_seconds:
            continue

        distance_to_boundary = min(
            abs(float(boundary) - float(run[0].start)),
            abs(float(boundary) - float(run[-1].end)),
            0.0
            if float(run[0].start) <= float(boundary) <= float(run[-1].end)
            else float("inf"),
        )
        if distance_to_boundary > boundary_tolerance_seconds:
            continue

        average_probability = sum(float(word.probability) for word in run) / len(run)
        if average_probability < min_average_probability:
            continue

        overlap = [
            word
            for word in existing
            if _word_midpoint(word) >= float(run[0].start) - 0.25
            and _word_midpoint(word) <= float(run[-1].end) + 0.25
        ]
        if len(overlap) > max(1, len(run) // 3):
            continue

        spans.append(
            RecoveredSpan(
                boundary=float(boundary),
                words=tuple(run),
                left_anchor=normalize_word(left_anchor.word) if left_anchor else "__window_start__",
                right_anchor=normalize_word(right_anchor.word) if right_anchor else "__audio_end__",
                average_probability=average_probability,
            )
        )

    return spans


def _span_segment(span: RecoveredSpan) -> Segment:
    words = list(span.words)
    probabilities = [max(float(word.probability), 1e-9) for word in words]
    return Segment(
        id=0,
        seek=max(0, int(float(words[0].start) * 100)),
        start=float(words[0].start),
        end=float(words[-1].end),
        text="".join(word.word for word in words).strip(),
        tokens=[],
        avg_logprob=sum(math.log(probability) for probability in probabilities)
        / len(probabilities),
        compression_ratio=0.0,
        no_speech_prob=0.0,
        words=words,
        temperature=0.0,
    )


def merge_recovered_spans(
    segments: Sequence[Segment], spans: Sequence[RecoveredSpan]
) -> list[Segment]:
    combined = list(segments) + [_span_segment(span) for span in spans]
    combined.sort(key=lambda segment: (float(segment.start), float(segment.end)))
    return [replace(segment, id=index) for index, segment in enumerate(combined, start=1)]


def _decode_window(
    base_model: Any,
    audio: np.ndarray,
    window: BoundaryWindow,
    *,
    language: str,
    beam_size: int,
) -> tuple[float, list[Word]]:
    start_sample = max(0, int(round(window.start * SAMPLING_RATE)))
    end_sample = min(audio.shape[0], int(round(window.end * SAMPLING_RATE)))
    window_audio = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
    if window_audio.size == 0:
        return window.boundary, []

    window_segments_iter, _ = base_model.transcribe(
        window_audio,
        language=language,
        task="transcribe",
        beam_size=min(int(beam_size), REPAIR_BEAM_SIZE),
        temperature=0,
        vad_filter=False,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=True,
    )
    recovered_words = [
        Word(
            start=float(word.start) + window.start,
            end=float(word.end) + window.start,
            word=word.word,
            probability=float(word.probability),
        )
        for word in _flatten_words(window_segments_iter)
    ]
    return window.boundary, recovered_words


def decode_repair_windows(
    base_model: Any,
    audio: np.ndarray,
    windows: Sequence[BoundaryWindow],
    *,
    language: str,
    beam_size: int,
    window_workers: int,
) -> dict[float, list[Word]]:
    """Decode the same independent windows, optionally in parallel.

    The acceptance and merge phase remains ordered by boundary, so increasing
    workers only changes scheduling and never the repair policy.
    """

    if window_workers < 1:
        raise ValueError("window_workers debe ser positivo")
    if window_workers == 1 or len(windows) <= 1:
        return dict(
            _decode_window(
                base_model,
                audio,
                window,
                language=language,
                beam_size=beam_size,
            )
            for window in windows
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(window_workers, len(windows)),
        thread_name_prefix="boundary-repair",
    ) as executor:
        futures = [
            executor.submit(
                _decode_window,
                base_model,
                audio,
                window,
                language=language,
                beam_size=beam_size,
            )
            for window in windows
        ]
        return dict(future.result() for future in futures)


def repair_batched_boundaries(
    batched_transcriber: Any,
    audio_path: Any,
    segments: Sequence[Segment],
    *,
    language: str,
    beam_size: int,
    min_silence_duration_ms: int = 700,
    window_workers: int = DEFAULT_WINDOW_WORKERS,
) -> tuple[list[Segment], RepairStats]:
    """Repair safely anchored omissions without changing the batched first pass."""

    started = time.perf_counter()
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLING_RATE)
    duration = float(audio.shape[0]) / SAMPLING_RATE
    existing_words = _flatten_words(segments)
    boundaries = detect_packed_boundaries(
        audio,
        min_silence_duration_ms=min_silence_duration_ms,
        include_final=True,
    )
    selected_boundaries = suspicious_boundaries(boundaries, existing_words)
    windows = plan_adaptive_windows(selected_boundaries, duration, existing_words)
    accepted: list[RecoveredSpan] = []

    base_model = getattr(batched_transcriber, "model", None)
    if base_model is None:
        raise TypeError("La reparación de límites requiere BatchedInferencePipeline.model")

    recovered_by_boundary = decode_repair_windows(
        base_model,
        audio,
        windows,
        language=language,
        beam_size=beam_size,
        window_workers=window_workers,
    )
    for window in windows:
        recovered_words = recovered_by_boundary.get(window.boundary, [])
        left_words = [word for word in existing_words if float(word.end) <= window.boundary + 0.25]
        right_words = [word for word in existing_words if float(word.start) >= window.boundary - 0.25]
        left_gap = (
            window.boundary - max(float(word.end) for word in left_words)
            if left_words
            else float("inf")
        )
        allow_left_edge = left_gap > WINDOW_BEFORE_SECONDS - 0.5
        allow_right_edge = not right_words and window.boundary == selected_boundaries[-1]
        new_spans = recoverable_spans(
            existing_words,
            recovered_words,
            window.boundary,
            min_missing_words=2 if allow_right_edge else MIN_MISSING_WORDS,
            min_average_probability=EDGE_MIN_AVERAGE_PROBABILITY
            if allow_left_edge or allow_right_edge
            else MIN_AVERAGE_PROBABILITY,
            allow_left_edge=allow_left_edge,
            allow_right_edge=allow_right_edge,
        )
        accepted.extend(new_spans)
        for span in new_spans:
            existing_words.extend(span.words)
        existing_words.sort(key=lambda word: (float(word.start), float(word.end)))

    repaired = merge_recovered_spans(segments, accepted)
    stats = RepairStats(
        boundaries=len(boundaries),
        windows_decoded=len(windows),
        spans_added=len(accepted),
        words_added=sum(len(span.words) for span in accepted),
        elapsed_seconds=time.perf_counter() - started,
    )
    return repaired, stats
