"""Crea un Pod que arranca el worker automáticamente (RunPod REST API v2)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://api.runpod.io/v2/pods"
WORKER_ENV_KEYS = (
    "RADIO_BASE_URL", "RADIO_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "WHISPER_MODEL", "PYANNOTE_MODEL", "TRANSCRIPTION_LANGUAGE",
    "POLL_INTERVAL_SECONDS", "HEARTBEAT_INTERVAL_SECONDS", "COMPUTE_TYPE",
    "WHISPER_BATCH_SIZE", "PYANNOTE_SEGMENTATION_BATCH_SIZE",
    "PYANNOTE_EMBEDDING_BATCH_SIZE",
)
SECRET_KEYS = {"RADIO_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
# La imagen usa CUDA 12.8; RunPod filtra por versión exacta del driver, así que se
# listan todas las que pueden ejecutarla.
ALLOWED_CUDA_VERSIONS = ["12.8", "12.9", "13.0", "13.1", "13.2"]


def api_error_detail(exc: HTTPError, env: dict[str, str]) -> str:
    """Recupera el motivo del rechazo sin imprimir credenciales reflejadas."""
    try:
        raw = exc.read(16384).decode("utf-8", errors="replace")
    except OSError:
        return "No se pudo leer el detalle del rechazo."
    for key in SECRET_KEYS | {"RUNPOD_API_KEY", "VAST_API_KEY", "REGISTRY_PASSWORD"}:
        secret = env.get(key)
        if secret:
            raw = raw.replace(secret, "***")
            raw = raw.replace(json.dumps(secret)[1:-1], "***")
    raw = re.sub(r"(?:dckr_pat_|hf_|rpa_)[A-Za-z0-9_-]+", "***", raw)
    try:
        body = json.loads(raw)
    except ValueError:
        body = None
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("message") or body.get("error") or body.get("title")
        raw = json.dumps(detail, ensure_ascii=False) if detail is not None else raw
    # Una respuesta HTML de un proxy también ayuda a distinguirla de la API.
    return " ".join(raw.split())[:2000] or "La API no devolvió un motivo."


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return {**values, **os.environ}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Debe ser mayor que cero.")
    return number


def build_payload(args: argparse.Namespace, env: dict[str, str]) -> dict:
    worker_env = {key: env[key] for key in WORKER_ENV_KEYS if env.get(key)}
    worker_env["WORKER_CAPABILITIES"] = args.capabilities
    payload = {
        "name": args.name,
        "image": args.image,
        "gpu": {"id": args.gpu, "count": args.gpu_count,
                "allowedCudaVersions": ALLOWED_CUDA_VERSIONS},
        "cloud": args.cloud,
        "disk": args.disk_gb,
        "startSsh": False,
        "env": worker_env,
        "mounts": {"persistent": {"size": args.volume_gb, "path": "/workspace"}},
    }
    if args.network_volume_id:
        if not args.data_center_id:
            raise ValueError("--network-volume-id requiere --data-center-id del volumen.")
        payload["mounts"] = {"network": [{"volumeId": args.network_volume_id, "path": "/workspace"}]}
    if args.data_center_id:
        payload["dataCenterIds"] = [args.data_center_id]
    if args.registry_id:
        payload["registry"] = args.registry_id
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Imagen ya publicada en un registro Docker.")
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 4090", help="ID del catálogo de RunPod.")
    parser.add_argument("--gpu-count", type=positive_int, default=1)
    parser.add_argument("--name", default="irixslave-transcripcion")
    parser.add_argument("--cloud", choices=("SECURE", "COMMUNITY"), default="SECURE")
    parser.add_argument("--disk-gb", type=positive_int, default=30)
    parser.add_argument("--volume-gb", type=positive_int, default=40)
    parser.add_argument("--network-volume-id")
    parser.add_argument("--data-center-id")
    parser.add_argument("--registry-id", help="Credencial de registro privado configurada en RunPod.")
    parser.add_argument("--capabilities", choices=("transcription", "diarization", "transcription,diarization"),
                        default="transcription,diarization")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    parser.add_argument("--create", action="store_true", help="Crea el Pod (facturable). Sin esta opción muestra la petición sin secretos.")
    args = parser.parse_args(argv)
    env = read_env(args.env_file)
    try:
        payload = build_payload(args, env)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.create:
        preview = {**payload, "env": {key: "***" if key in SECRET_KEYS else value
                                      for key, value in payload["env"].items()}}
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print("Vista previa. Agrega --create para crear y arrancar el Pod.")
        return 0
    required = ["RUNPOD_API_KEY", "RADIO_API_TOKEN"]
    if "diarization" in args.capabilities and not env.get("HUGGING_FACE_HUB_TOKEN"):
        required.append("HF_TOKEN")
    missing = [key for key in required if not env.get(key) or env[key].startswith("reemplaza_")]
    if missing:
        parser.error(f"Falta configurar: {', '.join(missing)}")
    request = Request(API_URL, data=json.dumps(payload).encode(), method="POST",
                      headers={"Authorization": f"Bearer {env['RUNPOD_API_KEY']}",
                               "Content-Type": "application/json", "Accept": "application/json",
                               "User-Agent": "irixslave-runpod/1.0"})
    try:
        # No reintentamos POST: una respuesta perdida podría crear dos Pods facturables.
        with urlopen(request, timeout=90) as response:
            pod = json.load(response)
    except HTTPError as exc:
        print(f"RunPod rechazó la creación (HTTP {exc.code}).", file=sys.stderr)
        print(f"Detalle: {api_error_detail(exc, env)}", file=sys.stderr)
        if exc.code in (401, 403):
            print("Comprueba la API key de RunPod y sus permisos para Pods. "
                  "RUNPOD_API_KEY exportada en el sistema tiene prioridad sobre .env.", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"No se pudo confirmar la creación ({type(exc).__name__}). Revisa la consola RunPod antes de repetir.", file=sys.stderr)
        return 1
    pod_id = pod.get("id") if isinstance(pod, dict) else None
    if not pod_id:
        print("RunPod respondió sin ID. Revisa la consola antes de repetir.", file=sys.stderr)
        return 1
    print(f"Pod creado: {pod_id}")
    print(f"https://www.runpod.io/console/pods/{pod_id}")
    print("El contenedor arrancará el worker en modo LIVE con modelos persistentes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
