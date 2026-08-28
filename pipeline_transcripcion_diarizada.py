from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".wma",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio con Faster-Whisper large-v3, lo diariza con "
            "Pyannote Community-1 y alinea cada palabra con un hablante local."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Audio o carpeta de audios.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resultados"),
        help="Carpeta de salida. Predeterminado: ./resultados",
    )
    parser.add_argument("--language", default="es")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--transcription-engine",
        choices=("faster-whisper", "mlx"),
        default="faster-whisper",
        help=(
            "faster-whisper (CTranslate2, CUDA/CPU) o mlx (Metal, solo Apple Silicon). "
            "Con --transcription-engine mlx, --whisper-model debe ser un repo de HF con pesos MLX "
            "en formato weights.npz o weights.safetensors (ej. mlx-community/whisper-large-v3-mlx; "
            "NO uses variantes que empaquetan model.safetensors, como -fp16 u -8bit, no son compatibles "
            "con el loader de mlx_whisper 0.4.3)."
        ),
    )
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument(
        "--pyannote-model",
        default="pyannote/speaker-diarization-community-1",
    )
    parser.add_argument(
        "--diarization-engine",
        choices=("pyannote", "soniqo"),
        default="pyannote",
        help=(
            "pyannote (pyannote-audio/PyTorch, CUDA/CPU) o soniqo (CLI 'speech diarize --engine community1' "
            "vía CoreML/Neural Engine, solo Apple Silicon; requiere 'brew install speech')."
        ),
    )
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--compute-type",
        help=(
            "Tipo de cómputo de Faster-Whisper (ej. int8_float16, float16, int8). "
            "Predeterminado: int8_float16 en CUDA, int8 en CPU."
        ),
    )
    parser.add_argument(
        "--segmentation-batch-size",
        type=int,
        default=6,
        help="Lote de la segmentación de Pyannote (solo --diarization-engine pyannote).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Lote de los embeddings de Pyannote (solo --diarization-engine pyannote).",
    )
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-transcription", action="store_true")
    parser.add_argument("--skip-diarization", action="store_true")
    return parser.parse_args()


def discover_audio(source: Path) -> tuple[Path, list[Path]]:
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Formato no admitido: {source.suffix}")
        return source.parent, [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No se encontraron audios compatibles en {source}")
    return source, files


def reject_output_collisions(audio_root: Path, files: list[Path]) -> None:
    seen: dict[Path, Path] = {}
    for audio in files:
        output_key = audio.relative_to(audio_root).with_suffix("")
        previous = seen.get(output_key)
        if previous is not None:
            raise ValueError(
                "Dos entradas producirían el mismo nombre de salida: "
                f"{previous} y {audio}"
            )
        seen[output_key] = audio


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def nvidia_library_dirs() -> list[Path]:
    """Carpetas de las libs nativas de los paquetes pip nvidia-cublas-cu12,
    nvidia-cudnn-cu12 y nvidia-cuda-nvrtc-cu12 (mismo venv que este proceso).
    ctranslate2 (motor de Faster-Whisper) las necesita para correr en CUDA y
    no las trae empaquetadas en su wheel, ni en Windows ni en Linux; solo
    cambia dónde busca el sistema operativo: PATH (DLLs) en Windows,
    LD_LIBRARY_PATH (.so) en Linux."""
    site_packages = Path(sysconfig.get_paths()["purelib"])
    nvidia_root = site_packages / "nvidia"
    subdir = "bin" if sys.platform == "win32" else "lib"
    candidates = (nvidia_root / pkg / subdir for pkg in ("cublas", "cudnn", "cuda_nvrtc"))
    return [path for path in candidates if path.is_dir()]


def runtime_environment(output_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYANNOTE_METRICS_ENABLED"] = "false"

    # Igual que worker_transcripcion.py: si el token de Hugging Face vino por
    # .env en vez de por "hf auth login", lo pasamos al subproceso de
    # diarizar.py (que lo busca en HF_TOKEN / HUGGING_FACE_HUB_TOKEN).
    dotenv_values = load_dotenv(Path(__file__).resolve().parent / ".env")
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if dotenv_values.get(key) and not env.get(key):
            env[key] = dotenv_values[key]
    matplotlib_cache = output_root / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(matplotlib_cache)

    lib_dirs = [str(path) for path in nvidia_library_dirs()]
    if lib_dirs:
        path_var = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
        env[path_var] = os.pathsep.join(lib_dirs + [env.get(path_var, "")])
    return env


def run_stage(command: list[str], env: dict[str, str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"{label} terminó con código {completed.returncode}")


def write_input_list(audio_root: Path, files: list[Path], destination: Path) -> None:
    relative_paths = []
    for audio in files:
        relative = audio.relative_to(audio_root)
        if "\n" in str(relative) or "\r" in str(relative):
            raise ValueError(f"Nombre de archivo no admitido: {relative!s}")
        relative_paths.append(str(relative))
    destination.write_text("\n".join(relative_paths) + "\n", encoding="utf-8")


def relative_output(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def write_result_index(
    output_root: Path,
    audio_root: Path,
    files: list[Path],
    whisper_model: str,
    pyannote_model: str,
) -> Path:
    transcript_root = output_root / "transcripciones"
    diarization_root = output_root / "diarizaciones"
    items = []
    for audio in files:
        relative = audio.relative_to(audio_root)
        transcript_json = (transcript_root / relative).with_suffix(".json")
        speakers_json = (diarization_root / relative).with_suffix(".speakers.json")
        items.append(
            {
                "audio": str(audio),
                "transcription_json": relative_output(transcript_json, output_root),
                "diarized_transcription_json": relative_output(speakers_json, output_root),
                "diarized_transcription_txt": relative_output(
                    (diarization_root / relative).with_suffix(".speakers.txt"),
                    output_root,
                ),
                "diarized_subtitles_srt": relative_output(
                    (diarization_root / relative).with_suffix(".speakers.srt"),
                    output_root,
                ),
                "speaker_scope": "local_to_audio_file",
                "complete": transcript_json.is_file() and speakers_json.is_file(),
            }
        )

    index_path = output_root / "indice_resultados.json"
    index_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "whisper_model": whisper_model,
                "pyannote_model": pyannote_model,
                "speaker_identity_note": (
                    "SPEAKER_XX es local a cada audio y no identifica automáticamente "
                    "a una persona real."
                ),
                "audio_count": len(items),
                "complete_count": sum(bool(item["complete"]) for item in items),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return index_path


def main() -> int:
    args = parse_args()
    package_root = Path(__file__).resolve().parent
    scripts_root = package_root / "scripts_originales"
    transcriber = (
        scripts_root / "transcribir_mlx.py"
        if args.transcription_engine == "mlx"
        else scripts_root / "transcribir.py"
    )
    diarizer = (
        scripts_root / "diarizar_soniqo.py"
        if args.diarization_engine == "soniqo"
        else scripts_root / "diarizar.py"
    )
    for required in (transcriber, diarizer):
        if not required.is_file():
            raise FileNotFoundError(f"Falta un script del paquete: {required}")

    audio_root, audio_files = discover_audio(args.input)
    reject_output_collisions(audio_root, audio_files)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    transcript_root = output_root / "transcripciones"
    diarization_root = output_root / "diarizaciones"
    models_dir = (
        args.models_dir.resolve()
        if args.models_dir
        else package_root / "modelos"
    )
    models_dir.mkdir(parents=True, exist_ok=True)

    file_list = output_root / "_archivos_entrada.txt"
    write_input_list(audio_root, audio_files, file_list)
    env = runtime_environment(output_root)
    compute_type = args.compute_type or (
        "int8_float16" if args.device == "cuda" else "int8"
    )
    batch_size = args.batch_size if args.device == "cuda" else 0

    if not args.skip_transcription:
        if args.transcription_engine == "mlx":
            command = [
                sys.executable,
                str(transcriber),
                "--input",
                str(audio_root),
                "--file-list",
                str(file_list),
                "--output",
                str(transcript_root),
                "--model",
                args.whisper_model,
                "--language",
                args.language,
                "--beam-size",
                str(args.beam_size),
                "--word-timestamps",
            ]
            label = "Transcripción con Whisper (MLX)"
        else:
            command = [
                sys.executable,
                str(transcriber),
                "--input",
                str(audio_root),
                "--file-list",
                str(file_list),
                "--output",
                str(transcript_root),
                "--model",
                args.whisper_model,
                "--model-dir",
                str(models_dir / "whisper"),
                "--device",
                args.device,
                "--compute-type",
                compute_type,
                "--language",
                args.language,
                "--beam-size",
                str(args.beam_size),
                "--batch-size",
                str(batch_size),
                "--word-timestamps",
            ]
            label = "Transcripción con Faster-Whisper"
        if args.force:
            command.append("--force")
        run_stage(command, env, label)

    if not args.skip_diarization:
        if args.diarization_engine == "soniqo":
            command = [
                sys.executable,
                str(diarizer),
                "--audio-root",
                str(audio_root),
                "--transcript-root",
                str(transcript_root),
                "--output-root",
                str(diarization_root),
                "--file-list",
                str(file_list),
            ]
            label = "Diarización con Soniqo (community1, CoreML)"
        else:
            command = [
                sys.executable,
                str(diarizer),
                "--audio-root",
                str(audio_root),
                "--transcript-root",
                str(transcript_root),
                "--output-root",
                str(diarization_root),
                "--cache-dir",
                str(models_dir / "pyannote-cache"),
                "--model",
                args.pyannote_model,
                "--file-list",
                str(file_list),
                "--device",
                args.device,
                "--segmentation-batch-size",
                str(args.segmentation_batch_size),
                "--embedding-batch-size",
                str(args.embedding_batch_size),
            ]
            if args.device == "cpu":
                command.append("--allow-cpu")
            label = "Diarización y alineación con Pyannote"
        for option in ("num_speakers", "min_speakers", "max_speakers"):
            value = getattr(args, option)
            if value is not None:
                command.extend((f"--{option.replace('_', '-')}", str(value)))
        if args.force:
            command.append("--force")
        run_stage(command, env, label)

    index_path = write_result_index(
        output_root,
        audio_root,
        audio_files,
        args.whisper_model,
        args.pyannote_model,
    )
    print(f"\nProceso terminado. Índice: {index_path}")
    if len(audio_files) == 1:
        relative = audio_files[0].relative_to(audio_root)
        final_path = (diarization_root / relative).with_suffix(".speakers.json")
        print(f"Transcripción diarizada: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
