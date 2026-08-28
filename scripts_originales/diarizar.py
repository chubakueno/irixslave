from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# No enviar telemetría de duración/origen del audio salvo que el usuario la active.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

import av
import numpy as np
import torch
from huggingface_hub import get_token
from pyannote.audio import Pipeline


MODEL_ID = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16_000


@dataclass
class Interval:
    start: float
    end: float
    speaker: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diariza audios con pyannote Community-1 y los alinea con Whisper."
    )
    parser.add_argument("--audio-root", type=Path, default=Path("audios"))
    parser.add_argument("--transcript-root", type=Path, default=Path("transcripciones"))
    parser.add_argument("--output-root", type=Path, default=Path("diarizaciones"))
    parser.add_argument("--cache-dir", type=Path, default=Path("modelos/pyannote-cache"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--file-list", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--test-seconds", type=float)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--segmentation-batch-size", type=int, default=8)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_uri(relative_path: Path) -> str:
    digest = hashlib.sha1(str(relative_path).encode("utf-8")).hexdigest()[:12]
    stem = "".join(char if char.isalnum() else "_" for char in relative_path.stem)
    return f"{stem[:48]}_{digest}"


def _ignore_invalid_frames(frames):
    """Salta paquetes con datos corruptos en vez de abortar la decodificación.
    Igual que faster_whisper.audio._ignore_invalid_frames: las capturas de
    streams SHOUTcast/Icecast suelen tener algún frame roto por un glitch de red
    del lado del grabador, y ffmpeg/faster-whisper lo saltan y siguen."""
    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def load_audio(path: Path, max_seconds: float | None = None) -> torch.Tensor:
    max_samples = round(max_seconds * SAMPLE_RATE) if max_seconds else None
    chunks: list[np.ndarray] = []
    samples = 0
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=SAMPLE_RATE
    )

    with av.open(str(path), metadata_errors="ignore") as container:
        if not container.streams.audio:
            raise RuntimeError("El archivo no contiene una pista de audio.")
        stream = container.streams.audio[0]
        for frame in _ignore_invalid_frames(container.decode(stream)):
            converted = resampler.resample(frame)
            for output_frame in converted if isinstance(converted, list) else [converted]:
                if output_frame is None:
                    continue
                values = output_frame.to_ndarray().reshape(-1)
                if max_samples is not None:
                    values = values[: max_samples - samples]
                if values.size:
                    chunks.append(values.copy())
                    samples += values.size
                if max_samples is not None and samples >= max_samples:
                    break
            if max_samples is not None and samples >= max_samples:
                break

    if not chunks:
        raise RuntimeError("No se pudieron decodificar muestras de audio.")
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    return torch.from_numpy(audio).unsqueeze(0)


def annotation_intervals(annotation: Any) -> list[Interval]:
    intervals = [
        Interval(float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    intervals.sort(key=lambda item: (item.start, item.end, item.speaker))
    return intervals


def interval_dicts(intervals: list[Interval]) -> list[dict[str, Any]]:
    return [
        {"start": round(item.start, 3), "end": round(item.end, 3), "speaker": item.speaker}
        for item in intervals
    ]


def choose_speaker(start: float, end: float, intervals: list[Interval]) -> str:
    best_speaker = "UNKNOWN"
    best_overlap = 0.0
    center = 0.5 * (start + end)
    nearest_distance = float("inf")
    nearest_speaker = "UNKNOWN"

    for interval in intervals:
        if interval.end < start - 1.0:
            continue
        if interval.start > end + 1.0 and best_overlap > 0:
            break
        overlap = max(0.0, min(end, interval.end) - max(start, interval.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = interval.speaker
        distance = 0.0 if interval.start <= center <= interval.end else min(
            abs(center - interval.start), abs(center - interval.end)
        )
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = interval.speaker

    if best_overlap > 0:
        return best_speaker
    return nearest_speaker if nearest_distance <= 0.75 else "UNKNOWN"


def transcript_units(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for segment in transcript.get("segments") or []:
        words = segment.get("words") or []
        if words:
            for word in words:
                if word.get("start") is None or word.get("end") is None:
                    continue
                units.append(
                    {
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                        "text": str(word.get("word", "")),
                        "probability": word.get("probability"),
                    }
                )
        elif str(segment.get("text", "")).strip():
            units.append(
                {
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                    "text": str(segment.get("text", "")),
                    "probability": None,
                }
            )
    return units


def align_transcript(
    transcript: dict[str, Any], exclusive: list[Interval]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aligned_units = []
    for unit in transcript_units(transcript):
        aligned_units.append(
            {**unit, "speaker": choose_speaker(unit["start"], unit["end"], exclusive)}
        )

    turns: list[dict[str, Any]] = []
    for unit in aligned_units:
        text = unit["text"]
        if not text.strip():
            continue
        if (
            turns
            and turns[-1]["speaker"] == unit["speaker"]
            and unit["start"] - turns[-1]["end"] <= 0.8
        ):
            turns[-1]["end"] = max(turns[-1]["end"], unit["end"])
            turns[-1]["text"] += text
            turns[-1]["unit_count"] += 1
        else:
            turns.append(
                {
                    "start": unit["start"],
                    "end": unit["end"],
                    "speaker": unit["speaker"],
                    "text": text,
                    "unit_count": 1,
                }
            )

    for turn in turns:
        turn["start"] = round(turn["start"], 3)
        turn["end"] = round(turn["end"], 3)
        turn["text"] = " ".join(turn["text"].split())
    return aligned_units, turns


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_outputs(
    output_base: Path,
    relative_audio: Path,
    model: str,
    elapsed: float,
    duration: float,
    diarization: Any,
    exclusive_annotation: Any,
    embeddings: np.ndarray,
    transcript: dict[str, Any] | None,
) -> dict[str, Any]:
    regular = annotation_intervals(diarization)
    exclusive = annotation_intervals(exclusive_annotation)
    labels = [str(label) for label in diarization.labels()]
    metadata = {
        "audio": str(relative_audio),
        "model": model,
        "created_at_utc": utc_now(),
        "duration_seconds": round(duration, 3),
        "processing_seconds": round(elapsed, 3),
        "speaker_scope": "local_to_audio_file",
        "speakers": labels,
        "num_speakers": len(labels),
        "diarization": interval_dicts(regular),
        "exclusive_diarization": interval_dicts(exclusive),
    }
    atomic_text(
        output_base.with_suffix(".diarization.json"),
        json.dumps(metadata, ensure_ascii=False, indent=2),
    )

    rttm_tmp = output_base.with_suffix(".rttm.tmp")
    rttm_path = output_base.with_suffix(".rttm")
    rttm_tmp.parent.mkdir(parents=True, exist_ok=True)
    with rttm_tmp.open("w", encoding="utf-8") as handle:
        diarization.write_rttm(handle)
    rttm_tmp.replace(rttm_path)

    embedding_path = output_base.with_suffix(".embeddings.npz")
    embedding_tmp = output_base.with_suffix(".embeddings.tmp.npz")
    np.savez_compressed(
        embedding_tmp,
        labels=np.asarray(labels, dtype=str),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    embedding_tmp.replace(embedding_path)

    turn_count = 0
    if transcript is not None:
        aligned_units, turns = align_transcript(transcript, exclusive)
        aligned = {
            "audio": str(relative_audio),
            "speaker_scope": "local_to_audio_file",
            "speakers": labels,
            "units": aligned_units,
            "turns": turns,
        }
        atomic_text(
            output_base.with_suffix(".speakers.json"),
            json.dumps(aligned, ensure_ascii=False, indent=2),
        )
        text_content = "\n".join(
            f"[{srt_timestamp(turn['start']).replace(',', '.')[:-4]} - "
            f"{srt_timestamp(turn['end']).replace(',', '.')[:-4]}] "
            f"{turn['speaker']}: {turn['text']}"
            for turn in turns
        )
        atomic_text(output_base.with_suffix(".speakers.txt"), text_content + "\n")
        srt_content = "\n\n".join(
            f"{index}\n{srt_timestamp(turn['start'])} --> {srt_timestamp(turn['end'])}\n"
            f"[{turn['speaker']}] {turn['text']}"
            for index, turn in enumerate(turns, 1)
        )
        atomic_text(output_base.with_suffix(".speakers.srt"), srt_content + "\n")
        turn_count = len(turns)

    return {
        "speakers": len(labels),
        "regular_segments": len(regular),
        "exclusive_segments": len(exclusive),
        "aligned_turns": turn_count,
    }


def read_file_list(path: Path, audio_root: Path) -> list[Path]:
    items = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        items.append(candidate if candidate.is_absolute() else audio_root / candidate)
    return items


def completed(output_base: Path) -> bool:
    suffixes = (
        ".diarization.json",
        ".rttm",
        ".embeddings.npz",
        ".speakers.json",
        ".speakers.txt",
        ".speakers.srt",
    )
    return all(output_base.with_suffix(suffix).exists() for suffix in suffixes)


def append_progress(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_pipeline(
    model: str,
    cache_dir: Path,
    device_arg: str,
    segmentation_batch_size: int,
    embedding_batch_size: int,
    *,
    allow_cpu: bool = False,
) -> tuple[torch.device, Any]:
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if device_arg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA no está disponible. Se detiene para evitar una ejecución accidental de varios días."
        )

    model_path = Path(model)
    is_local_model = model_path.exists()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or get_token()
    if not is_local_model and not token:
        raise RuntimeError(
            "Falta la autorización de Hugging Face. Ejecuta configurar_huggingface.ps1 "
            "después de aceptar las condiciones de pyannote Community-1."
        )

    print(f"DEVICE={device}")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}")
        # pyannote fuerza allow_tf32=False en cada llamada (fix_reproducibility),
        # así que no tiene sentido pedir "high" aquí: solo generaba un warning.
        torch.backends.cudnn.benchmark = True

    source = str(model_path.resolve()) if is_local_model else model
    print(f"MODEL={source}")
    pipeline = Pipeline.from_pretrained(
        source,
        token=token,
        cache_dir=cache_dir,
    )
    pipeline.to(device)
    if hasattr(pipeline, "segmentation_batch_size"):
        pipeline.segmentation_batch_size = segmentation_batch_size
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = embedding_batch_size
    return device, pipeline


def diarize_one(
    pipeline: Any,
    device: torch.device,
    audio_path: Path,
    audio_root: Path,
    transcript_root: Path,
    output_root: Path,
    model_name: str,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    test_seconds: float | None = None,
) -> dict[str, Any]:
    """Diariza un solo audio con un pipeline ya cargado y escribe sus salidas.
    No atrapa excepciones: el llamador decide si continúa con el resto del lote."""
    relative_audio = audio_path.resolve().relative_to(audio_root)
    output_base = output_root / relative_audio.with_suffix("")
    started = time.perf_counter()

    waveform = load_audio(audio_path, test_seconds)
    duration = waveform.shape[1] / SAMPLE_RATE
    uri = safe_uri(relative_audio)
    pipeline_input = {"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": uri}
    call_kwargs = {}
    for name, value in (
        ("num_speakers", num_speakers),
        ("min_speakers", min_speakers),
        ("max_speakers", max_speakers),
    ):
        if value is not None:
            call_kwargs[name] = value

    while True:
        try:
            result = pipeline(pipeline_input, **call_kwargs)
            break
        except torch.OutOfMemoryError:
            old_seg = int(getattr(pipeline, "segmentation_batch_size", 1))
            old_emb = int(getattr(pipeline, "embedding_batch_size", 1))
            if old_seg <= 1 and old_emb <= 1:
                raise
            pipeline.segmentation_batch_size = max(1, old_seg // 2)
            pipeline.embedding_batch_size = max(1, old_emb // 2)
            torch.cuda.empty_cache()
            print(
                "  OOM: reintento con lotes "
                f"{pipeline.segmentation_batch_size}/{pipeline.embedding_batch_size}",
                flush=True,
            )

    transcript_path = transcript_root / relative_audio.with_suffix(".json")
    transcript = None
    if transcript_path.exists() and test_seconds is None:
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - started
    stats = write_outputs(
        output_base,
        relative_audio,
        model_name,
        elapsed,
        duration,
        result.speaker_diarization,
        result.exclusive_speaker_diarization,
        result.speaker_embeddings,
        transcript,
    )
    del waveform, result
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "audio_seconds": round(duration, 3),
        "processing_seconds": round(elapsed, 3),
        **stats,
    }


def main() -> int:
    args = parse_args()
    audio_root = args.audio_root.resolve()
    transcript_root = args.transcript_root.resolve()
    output_root = args.output_root.resolve()
    cache_dir = args.cache_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device, pipeline = load_pipeline(
        args.model,
        cache_dir,
        args.device,
        args.segmentation_batch_size,
        args.embedding_batch_size,
        allow_cpu=args.allow_cpu,
    )

    files = (
        read_file_list(args.file_list.resolve(), audio_root)
        if args.file_list
        else sorted(audio_root.rglob("*.mp3"))
    )
    if args.limit:
        files = files[: args.limit]
    print(f"FILES={len(files)}")

    progress_path = output_root / "_progreso.csv"
    successes = 0
    errors = 0
    for index, audio_path in enumerate(files, 1):
        relative_audio = audio_path.resolve().relative_to(audio_root)
        output_base = output_root / relative_audio.with_suffix("")
        if completed(output_base) and not args.force:
            print(f"[{index}/{len(files)}] SKIP {relative_audio}", flush=True)
            continue

        print(f"[{index}/{len(files)}] START {relative_audio}", flush=True)
        started = time.perf_counter()
        try:
            stats = diarize_one(
                pipeline,
                device,
                audio_path,
                audio_root,
                transcript_root,
                output_root,
                args.model,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                test_seconds=args.test_seconds,
            )
            successes += 1
            append_progress(
                progress_path,
                {
                    "timestamp_utc": utc_now(),
                    "audio": str(relative_audio),
                    "status": "ok",
                    **stats,
                    "error": "",
                },
            )
            print(
                f"[{index}/{len(files)}] OK speakers={stats['speakers']} "
                f"time={stats['processing_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            elapsed = time.perf_counter() - started
            append_progress(
                progress_path,
                {
                    "timestamp_utc": utc_now(),
                    "audio": str(relative_audio),
                    "status": "error",
                    "audio_seconds": "",
                    "processing_seconds": round(elapsed, 3),
                    "speakers": "",
                    "regular_segments": "",
                    "exclusive_segments": "",
                    "aligned_turns": "",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"[{index}/{len(files)}] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()

    print(f"DONE successes={successes} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
