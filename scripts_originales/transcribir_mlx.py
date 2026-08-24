from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import av
import mlx_whisper


def timestamp(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def safe_text(value: str) -> str:
    return " ".join(value.strip().split())


def audio_duration_seconds(path: Path) -> float:
    with av.open(str(path), metadata_errors="replace") as container:
        stream = container.streams.audio[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
    raise RuntimeError(f"No se pudo determinar la duración de {path}")


def write_outputs(audio: Path, root: Path, out_root: Path, language: str, duration: float, segments: list) -> None:
    relative = audio.relative_to(root)
    target = out_root / relative.parent / relative.stem
    target.parent.mkdir(parents=True, exist_ok=True)

    text_path = target.with_suffix(".txt")
    srt_path = target.with_suffix(".srt")
    json_path = target.with_suffix(".json")

    text_lines = []
    srt_blocks = []
    json_segments = []
    for index, segment in enumerate(segments, start=1):
        text = safe_text(segment.get("text", ""))
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        text_lines.append(f"[{timestamp(start).replace(',', '.')}] {text}")
        srt_blocks.append(f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n")
        json_segments.append(
            {
                "id": segment.get("id", index - 1),
                "start": start,
                "end": end,
                "text": text,
                "avg_logprob": segment.get("avg_logprob"),
                "no_speech_prob": segment.get("no_speech_prob"),
                "compression_ratio": segment.get("compression_ratio"),
                "words": [
                    {
                        "start": word.get("start"),
                        "end": word.get("end"),
                        "word": word.get("word", ""),
                        "probability": word.get("probability"),
                    }
                    for word in (segment.get("words") or [])
                ],
            }
        )

    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "audio": str(relative),
                "language": language,
                "language_probability": None,
                "duration": duration,
                "duration_after_vad": None,
                "segments": json_segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe una carpeta de audios con Whisper large-v3 vía MLX (Apple Silicon).")
    parser.add_argument("--input", type=Path, default=Path("audios"))
    parser.add_argument(
        "--file-list",
        type=Path,
        help="Archivo UTF-8 con una ruta relativa al directorio de entrada por línea.",
    )
    parser.add_argument("--output", type=Path, default=Path("transcripciones"))
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx", help="Repo de Hugging Face con pesos MLX.")
    parser.add_argument("--language", default="es")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.input.resolve()
    out_root = args.output.resolve()
    if args.file_list:
        selected = [
            line.strip()
            for line in args.file_list.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        audio_files = [root / Path(relative) for relative in selected]
        missing = [path for path in audio_files if not path.exists()]
        if missing:
            print(f"Faltan {len(missing)} audios indicados en {args.file_list}", file=sys.stderr)
            return 2
    else:
        audio_files = sorted(root.rglob("*.mp3"))
    if not audio_files:
        print(f"No se encontraron MP3 en {root}", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "_progreso.csv"
    existing = {}
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", newline="") as fh:
            existing = {row["audio"]: row for row in csv.DictReader(fh)}

    pending = []
    for audio in audio_files:
        relative = str(audio.relative_to(root))
        done_json = (out_root / audio.relative_to(root)).with_suffix(".json")
        if not args.force and done_json.exists() and existing.get(relative, {}).get("status") == "ok":
            continue
        pending.append((audio, relative))

    print(f"Audios encontrados: {len(audio_files)}; pendientes: {len(pending)}")
    print(f"Motor: mlx-whisper | modelo: {args.model} | beam_size: {args.beam_size}")
    if not pending:
        print("No hay audios pendientes.")
        return 0

    decode_options = {}
    if args.beam_size and args.beam_size > 1:
        decode_options["beam_size"] = args.beam_size

    fields = ["audio", "status", "seconds", "duration", "language", "message"]
    with log_path.open("a", encoding="utf-8", newline="") as log_fh:
        writer = csv.DictWriter(log_fh, fieldnames=fields)
        if log_path.stat().st_size == 0:
            writer.writeheader()
        for number, (audio, relative) in enumerate(pending, start=1):
            started = time.perf_counter()
            print(f"[{number}/{len(pending)}] {relative}", flush=True)
            try:
                result = mlx_whisper.transcribe(
                    str(audio),
                    path_or_hf_repo=args.model,
                    language=args.language,
                    # Tupla, no un solo float: habilita el reintento a mayor temperatura cuando el decode a t=0 entra en loop de repetición.
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    condition_on_previous_text=False,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    word_timestamps=args.word_timestamps,
                    **decode_options,
                )
                duration = audio_duration_seconds(audio)
                write_outputs(audio, root, out_root, result.get("language", args.language), duration, result.get("segments") or [])
                elapsed = time.perf_counter() - started
                row = {
                    "audio": relative,
                    "status": "ok",
                    "seconds": f"{elapsed:.1f}",
                    "duration": f"{duration:.1f}",
                    "language": result.get("language", args.language),
                    "message": "",
                }
                print(f"  OK | {duration / 60:.1f} min de audio | {elapsed / 60:.1f} min de proceso", flush=True)
            except Exception as exc:  # keep the batch running if one file is damaged
                elapsed = time.perf_counter() - started
                row = {
                    "audio": relative,
                    "status": "error",
                    "seconds": f"{elapsed:.1f}",
                    "duration": "",
                    "language": "",
                    "message": repr(exc),
                }
                print(f"  ERROR | {exc}", file=sys.stderr, flush=True)
            writer.writerow(row)
            log_fh.flush()

    print(f"Finalizado. Resultados en: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
