"""Barrido de parámetros de transcripción con Faster-Whisper (CTranslate2).

Mide, para cada combinación (compute_type, batch_size, beam_size), el tiempo de
transcribir un audio, el RTF, y proxies de calidad para detectar si el cambio de
parámetro alteró la salida: nº de segmentos, nº de caracteres, logprob medio y
similitud del texto contra la primera corrida (la línea base).

Uso:
    ./.venv/bin/python sweep_whisper.py <audio> [--seconds 300]
        [--compute-types "int8_float16,float16,int8"]
        [--batch-sizes "8,16,24,32"]
        [--beam-sizes "1,5"]

Cómo leer el resultado:
    Buscá el mejor RTF cuya `sim` contra la base siga siendo ~1.000 y cuyo
    `logprob` no empeore. `compute_type` y `beam_size` SÍ cambian la salida;
    `batch_size` normalmente no (solo velocidad/VRAM). Si bajás a int8 y la `sim`
    cae, esa cuantización te está costando calidad.
    El ganador va en COMPUTE_TYPE / WHISPER_BATCH_SIZE del .env (o en
    --compute-type / --batch-size del pipeline).
"""

from __future__ import annotations

import argparse
import difflib
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / "scripts_originales"))
from transcribir import load_model  # noqa: E402

DEFAULT_COMPUTE_TYPES = "int8_float16,float16,int8"
DEFAULT_BATCH_SIZES = "8,16,24,32"
DEFAULT_BEAM_SIZES = "1"


def audio_seconds(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        return float(stream.duration * stream.time_base)


def transcribe_options(language: str, beam_size: int, batch_size: int, word_timestamps: bool) -> dict:
    # Mismas opciones que scripts_originales/transcribir.py en producción.
    options = {
        "language": language,
        "task": "transcribe",
        "beam_size": beam_size,
        "temperature": 0,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "word_timestamps": word_timestamps,
        "vad_parameters": {"min_silence_duration_ms": 700},
    }
    if batch_size > 0:
        options["batch_size"] = batch_size
    return options


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--seconds", type=float, default=300.0, help="Recorta el audio a N segundos (crea un .wav temporal). 0 = audio completo.")
    parser.add_argument("--compute-types", default=DEFAULT_COMPUTE_TYPES)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--beam-sizes", default=DEFAULT_BEAM_SIZES)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="es")
    parser.add_argument("--model-dir", type=Path, default=PACKAGE_ROOT / "modelos" / "whisper")
    parser.add_argument("--no-word-timestamps", action="store_true", help="Más rápido, pero no es lo que corre producción.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA no disponible; este barrido solo tiene sentido en GPU.", file=sys.stderr)
        return 2

    compute_types = [c.strip() for c in args.compute_types.split(",") if c.strip()]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    beam_sizes = [int(b) for b in args.beam_sizes.split(",") if b.strip()]
    word_timestamps = not args.no_word_timestamps

    # Recorte opcional a N segundos con un wav temporal (evita transcribir horas).
    audio_path = args.audio
    tmp_path: Path | None = None
    if args.seconds and args.seconds > 0:
        import av

        tmp_path = PACKAGE_ROOT / "_sweep_whisper_clip.wav"
        with av.open(str(args.audio)) as src, av.open(str(tmp_path), "w") as dst:
            in_stream = src.streams.audio[0]
            out_stream = dst.add_stream("pcm_s16le", rate=16000, layout="mono")
            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
            limit = args.seconds
            for frame in src.decode(in_stream):
                if frame.time is not None and frame.time > limit:
                    break
                for out_frame in resampler.resample(frame):
                    for packet in out_stream.encode(out_frame):
                        dst.mux(packet)
            for packet in out_stream.encode(None):
                dst.mux(packet)
        audio_path = tmp_path

    duration = audio_seconds(audio_path)
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"audio   : {args.audio.name}  ({duration:.0f}s)")
    print(f"barrido : compute={compute_types}  batch={batch_sizes}  beam={beam_sizes}  word_ts={word_timestamps}")
    print()

    baseline_text: str | None = None

    def run(compute_type: str, transcriber, batch_size: int, beam_size: int, *, report: bool = True) -> None:
        nonlocal baseline_text
        options = transcribe_options(args.language, beam_size, batch_size, word_timestamps)
        if not report:
            list(transcriber.transcribe(str(audio_path), **options)[0])
            return
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            segments_iter, _info = transcriber.transcribe(str(audio_path), **options)
            segments = list(segments_iter)
            elapsed = time.perf_counter() - started
            text = " ".join(s.text.strip() for s in segments)
            n_chars = len(text)
            logprobs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
            mean_lp = statistics.mean(logprobs) if logprobs else float("nan")
            peak = torch.cuda.max_memory_allocated() / 1e9
            if baseline_text is None:
                baseline_text = text
                sim = 1.0
            else:
                sim = difflib.SequenceMatcher(None, baseline_text, text).ratio()
            print(
                f"{compute_type:13s} batch={batch_size:3d} beam={beam_size:d}   "
                f"{elapsed:6.2f}s  RTF={duration / elapsed:6.1f}x   "
                f"seg={len(segments):4d}  chars={n_chars:6d}  logprob={mean_lp:+.3f}  sim={sim:.3f}  "
                f"peakVRAM={peak:4.1f}GB",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - seguir con el resto del barrido
            print(f"{compute_type:13s} batch={batch_size:3d} beam={beam_size:d}   FALLO {type(exc).__name__}: {str(exc)[:150]}", flush=True)
        finally:
            torch.cuda.empty_cache()

    try:
        for compute_type in compute_types:
            print(f"--- cargando modelo con compute_type={compute_type} ---", flush=True)
            try:
                transcriber = load_model(args.model, "cuda", compute_type, args.model_dir, batch_size=max(batch_sizes))
            except Exception as exc:  # noqa: BLE001
                print(f"{compute_type:13s}   NO CARGA: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
                continue
            # calentamiento por modelo (autotune de cuDNN); no se reporta
            run(compute_type, transcriber, batch_sizes[0], beam_sizes[0], report=False)
            for beam_size in beam_sizes:
                for batch_size in batch_sizes:
                    run(compute_type, transcriber, batch_size, beam_size)
            del transcriber
            torch.cuda.empty_cache()
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

    print()
    print("Mejor RTF con sim ~1.000 y logprob que no empeore. compute_type/beam cambian la salida; batch_size no.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
