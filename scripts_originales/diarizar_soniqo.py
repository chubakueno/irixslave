from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import av

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diarizar import Interval, align_transcript, atomic_text, interval_dicts, srt_timestamp, utc_now  # noqa: E402

SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diariza audios con el motor community1 de Soniqo (CoreML/Neural Engine) y los alinea con Whisper."
    )
    parser.add_argument("--audio-root", type=Path, default=Path("audios"))
    parser.add_argument("--transcript-root", type=Path, default=Path("transcripciones"))
    parser.add_argument("--output-root", type=Path, default=Path("diarizaciones"))
    parser.add_argument("--file-list", type=Path)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def audio_to_wav(path: Path, dest: Path) -> float:
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    samples = 0
    with wave.open(str(dest), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(SAMPLE_RATE)
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise RuntimeError("El archivo no contiene una pista de audio.")
            stream = container.streams.audio[0]
            for frame in container.decode(stream):
                converted = resampler.resample(frame)
                for output_frame in converted if isinstance(converted, list) else [converted]:
                    if output_frame is None:
                        continue
                    values = output_frame.to_ndarray().reshape(-1)
                    if values.size:
                        wav_out.writeframes(values.tobytes())
                        samples += values.size
    return samples / SAMPLE_RATE


def run_speech_diarize(
    wav_path: Path,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict[str, Any]:
    command = ["speech", "diarize", str(wav_path), "--engine", "community1", "--json"]
    if num_speakers is not None:
        command.extend(["--num-speakers", str(num_speakers)])
    if min_speakers is not None:
        command.extend(["--min-speakers", str(min_speakers)])
    if max_speakers is not None:
        command.extend(["--max-speakers", str(max_speakers)])
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"speech diarize terminó con código {completed.returncode}: {completed.stderr[-2000:]}")
    stdout = completed.stdout
    start = stdout.find("{")
    if start == -1:
        raise RuntimeError(f"No se encontró JSON en la salida de speech diarize: {stdout[-2000:]}")
    return json.loads(stdout[start:])


def write_outputs(
    output_base: Path,
    relative_audio: Path,
    model: str,
    elapsed: float,
    duration: float,
    intervals: list[Interval],
    transcript: dict[str, Any] | None,
) -> dict[str, Any]:
    labels = sorted({interval.speaker for interval in intervals})
    metadata = {
        "audio": str(relative_audio),
        "model": model,
        "created_at_utc": utc_now(),
        "duration_seconds": round(duration, 3),
        "processing_seconds": round(elapsed, 3),
        "speaker_scope": "local_to_audio_file",
        "speakers": labels,
        "num_speakers": len(labels),
        "diarization": interval_dicts(intervals),
        "exclusive_diarization": interval_dicts(intervals),
    }
    atomic_text(output_base.with_suffix(".diarization.json"), json.dumps(metadata, ensure_ascii=False, indent=2))

    turn_count = 0
    if transcript is not None:
        aligned_units, turns = align_transcript(transcript, intervals)
        aligned = {
            "audio": str(relative_audio),
            "speaker_scope": "local_to_audio_file",
            "speakers": labels,
            "units": aligned_units,
            "turns": turns,
        }
        atomic_text(output_base.with_suffix(".speakers.json"), json.dumps(aligned, ensure_ascii=False, indent=2))
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

    return {"speakers": len(labels), "aligned_turns": turn_count}


def completed(output_base: Path) -> bool:
    suffixes = (".diarization.json", ".speakers.json", ".speakers.txt", ".speakers.srt")
    return all(output_base.with_suffix(suffix).exists() for suffix in suffixes)


def read_file_list(path: Path, audio_root: Path) -> list[Path]:
    items = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        items.append(candidate if candidate.is_absolute() else audio_root / candidate)
    return items


def main() -> int:
    args = parse_args()
    audio_root = args.audio_root.resolve()
    transcript_root = args.transcript_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    files = (
        read_file_list(args.file_list.resolve(), audio_root)
        if args.file_list
        else sorted(audio_root.rglob("*.mp3"))
    )
    print(f"FILES={len(files)}")

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
            with tempfile.TemporaryDirectory(prefix="diarizar_soniqo_") as tmp:
                wav_path = Path(tmp) / "audio.wav"
                duration = audio_to_wav(audio_path, wav_path)
                result = run_speech_diarize(wav_path, args.num_speakers, args.min_speakers, args.max_speakers)

            intervals = [
                Interval(float(seg["start"]), float(seg["end"]), f"SPEAKER_{int(seg['speaker']):02d}")
                for seg in (result.get("segments") or [])
            ]

            transcript_path = transcript_root / relative_audio.with_suffix(".json")
            transcript = json.loads(transcript_path.read_text(encoding="utf-8")) if transcript_path.exists() else None

            elapsed = time.perf_counter() - started
            stats = write_outputs(
                output_base,
                relative_audio,
                "soniqo-community1",
                elapsed,
                duration,
                intervals,
                transcript,
            )
            successes += 1
            print(f"[{index}/{len(files)}] OK speakers={stats['speakers']} time={elapsed:.1f}s", flush=True)
        except Exception as exc:
            errors += 1
            print(f"[{index}/{len(files)}] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"DONE successes={successes} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
