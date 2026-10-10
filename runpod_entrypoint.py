"""Valida el entorno del Pod y reemplaza PID 1 por el worker."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> None:
    if not os.environ.get("RADIO_API_TOKEN"):
        raise SystemExit("Falta RADIO_API_TOKEN en las variables del Pod.")
    capabilities = os.environ.get("WORKER_CAPABILITIES", "transcription,diarization")
    if "--transcription-only" in sys.argv:
        capabilities = "transcription"
    elif "--diarization-only" in sys.argv:
        capabilities = "diarization"
    elif "--capabilities" in sys.argv:
        index = sys.argv.index("--capabilities")
        if index + 1 < len(sys.argv):
            capabilities = sys.argv[index + 1]
    for arg in sys.argv[1:]:
        if arg.startswith("--capabilities="):
            capabilities = arg.split("=", 1)[1]
    if "diarization" in {item.strip().lower() for item in capabilities.split(",")} and not (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ):
        raise SystemExit("Falta HF_TOKEN para descargar el modelo de diarización.")
    for key in ("HF_HOME", "MODELS_DIR", "RESULTS_DIR", "WORKER_TMP_DIR"):
        if os.environ.get(key):
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    if os.environ.get("DEVICE", "cuda") == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA no está disponible: este contenedor necesita un Pod con GPU NVIDIA.")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    worker = Path(__file__).with_name("worker_transcripcion.py")
    os.execv(sys.executable, [sys.executable, "-u", str(worker), *sys.argv[1:]])


if __name__ == "__main__":
    main()
