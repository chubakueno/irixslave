"""Barrido de tamaños de lote para la diarización con Pyannote Community-1.

Mide, para cada combinación (segmentation_batch_size, embedding_batch_size), el
tiempo de una corrida completa del pipeline sobre un audio, el RTF (segundos de
audio / segundos de proceso), el número de hablantes detectado y el pico de VRAM.

Uso:
    ./.venv/bin/python sweep_diarizacion.py <audio> [--seconds 300] [--combos "6x16,24x48,32x64"]

Cómo leer el resultado:
    Elegí el (seg, emb) con mejor RTF cuyo `n_spk` sea igual al de la fila base
    (la primera del barrido). Si a partir de cierto lote el `n_spk` cambia, ese
    lote está alterando el clustering: descartalo aunque sea más rápido.
    El ganador va en PYANNOTE_SEGMENTATION_BATCH_SIZE / PYANNOTE_EMBEDDING_BATCH_SIZE
    del .env (o en --segmentation-batch-size / --embedding-batch-size del pipeline).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

import torch
from pyannote.audio import Pipeline

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / "scripts_originales"))
from diarizar import SAMPLE_RATE, load_audio  # noqa: E402

MODEL_ID = "pyannote/speaker-diarization-community-1"
DEFAULT_COMBOS = "6x16,16x32,24x48,32x64,32x96,48x128"


def hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    env_path = PACKAGE_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")) and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def parse_combos(raw: str) -> list[tuple[int, int]]:
    combos: list[tuple[int, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        seg, _, emb = chunk.partition("x")
        combos.append((int(seg), int(emb)))
    if not combos:
        raise ValueError("--combos quedó vacío")
    return combos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="Audio de prueba (mp3, wav, m4a...).")
    parser.add_argument("--seconds", type=float, default=300.0, help="Cuántos segundos del audio usar. 0 = audio completo. Predeterminado: 300.")
    parser.add_argument("--combos", default=DEFAULT_COMBOS, help=f'Lista "segXemb" separada por comas. Predeterminado: "{DEFAULT_COMBOS}"')
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=PACKAGE_ROOT / "modelos" / "pyannote-cache")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA no disponible; este barrido solo tiene sentido en GPU.", file=sys.stderr)
        return 2

    combos = parse_combos(args.combos)
    token = hf_token()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    waveform = load_audio(args.audio, args.seconds if args.seconds > 0 else None)
    duration = waveform.shape[1] / SAMPLE_RATE
    pipeline_input = {"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": "sweep"}

    pipeline = Pipeline.from_pretrained(args.model, token=token, cache_dir=str(args.cache_dir))
    pipeline.to(device)

    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"audio   : {args.audio.name}  ({duration:.0f}s de {args.seconds:.0f}s pedidos)")
    print(f"combos  : {', '.join(f'{s}x{e}' for s, e in combos)}")
    print()

    def run(seg: int, emb: int) -> None:
        pipeline.segmentation_batch_size = seg
        pipeline.embedding_batch_size = emb
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            result = pipeline(pipeline_input)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            n_spk = len(result.speaker_diarization.labels())
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"seg={seg:3d} emb={emb:3d}   {elapsed:6.2f}s   "
                f"RTF={duration / elapsed:6.1f}x   n_spk={n_spk:2d}   peakVRAM={peak:4.1f}GB",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - queremos seguir con el resto del barrido
            print(f"seg={seg:3d} emb={emb:3d}   FALLO {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        finally:
            torch.cuda.empty_cache()

    # Corrida de calentamiento (autotune de cuDNN); no se reporta.
    print("(calentando...)", flush=True)
    run(*combos[0])
    print()

    for seg, emb in combos:
        run(seg, emb)

    print()
    print("Elegí el mejor RTF cuyo n_spk sea igual al de la primera fila.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
