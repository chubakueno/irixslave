"""Alquila una instancia Vast.ai que arranca el worker automáticamente (API REST v0)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_BASE = "https://console.vast.ai/api/v0"
WORKER_ARGS = ["--live", "--persistent-models"]
# Rechazos que garantizan que no se creó nada y la oferta ya no sirve: se prueba la siguiente.
OFFER_UNAVAILABLE = {400, 404, 409, 410}
WORKER_ENV_KEYS = (
    "RADIO_BASE_URL", "RADIO_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "WHISPER_MODEL", "PYANNOTE_MODEL", "TRANSCRIPTION_LANGUAGE",
    "POLL_INTERVAL_SECONDS", "HEARTBEAT_INTERVAL_SECONDS", "COMPUTE_TYPE",
    "WHISPER_BATCH_SIZE", "PYANNOTE_SEGMENTATION_BATCH_SIZE",
    "PYANNOTE_EMBEDDING_BATCH_SIZE",
)
SECRET_KEYS = {"RADIO_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}


def call_api(method: str, path: str, api_key: str, body: dict) -> dict:
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": "irixslave-vast/1.0"}
    if api_key:  # la búsqueda de ofertas es pública; alquilar exige la clave
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"{API_BASE}{path}", data=json.dumps(body).encode(), method=method, headers=headers)
    with urlopen(request, timeout=90) as response:
        return json.load(response)


def api_error_detail(exc: HTTPError, env: dict[str, str]) -> str:
    """Recupera el motivo del rechazo sin imprimir credenciales reflejadas."""
    try:
        raw = exc.read(16384).decode("utf-8", errors="replace")
    except OSError:
        return "No se pudo leer el detalle del rechazo."
    for key in SECRET_KEYS | {"VAST_API_KEY", "REGISTRY_PASSWORD"}:
        secret = env.get(key)
        if secret:
            raw = raw.replace(secret, "***")
            raw = raw.replace(json.dumps(secret)[1:-1], "***")
    raw = re.sub(r"(?:dckr_pat_|hf_)[A-Za-z0-9_-]+", "***", raw)
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


def gpu_preference(raw: str) -> list[str]:
    """Lista separada por comas, de más a menos deseable (p. ej. "RTX 4070S,RTX 4070")."""
    names = [name.strip().replace("_", " ") for name in raw.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("Indica al menos una GPU.")
    return names


def machine_ids(raw: str) -> set[int]:
    """IDs de máquina separados por comas (VAST_EXCLUDE_MACHINES)."""
    return {int(item) for item in raw.replace(" ", "").split(",") if item}


def rank_offers(offers: list[dict], gpus: list[str], excluded: set[int] | frozenset[int] = frozenset()) -> list[dict]:
    """Primero la GPU más deseable; dentro de cada modelo, la más barata."""
    return sorted((offer for offer in offers
                   if offer.get("gpu_name") in gpus and offer.get("machine_id") not in excluded),
                  key=lambda offer: (gpus.index(offer["gpu_name"]), offer.get("dph_total", 0)))


def build_query(args: argparse.Namespace) -> dict:
    query = {
        "verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False},
        "gpu_name": {"in": args.gpu},
        "num_gpus": {"eq": args.gpu_count},
        "cuda_max_good": {"gte": 12.8},  # la imagen usa CUDA 12.8
        "disk_space": {"gte": args.disk_gb},
        "reliability": {"gte": args.min_reliability},
        "inet_down": {"gte": args.min_download_mbps},
        "type": "ondemand", "order": [["dph_total", "asc"]], "limit": 50,
    }
    if args.max_price:
        query["dph_total"] = {"lte": args.max_price}
    if args.exclude_machine:
        query["machine_id"] = {"notin": sorted(args.exclude_machine)}
    return query


def build_instance(args: argparse.Namespace, env: dict[str, str]) -> dict:
    worker_env = {key: env[key] for key in WORKER_ENV_KEYS if env.get(key)}
    worker_env["WORKER_CAPABILITIES"] = args.capabilities
    instance = {
        "image": args.image,
        "label": args.label,
        "disk": args.disk_gb,
        # "args" ejecuta el ENTRYPOINT de la imagen; ssh/jupyter lo reemplazarían.
        "runtype": "args",
        "args": WORKER_ARGS,
        "env": worker_env,
    }
    if env.get("REGISTRY_USERNAME") and env.get("REGISTRY_PASSWORD"):
        instance["image_login"] = (f"-u {env['REGISTRY_USERNAME']} -p {env['REGISTRY_PASSWORD']} "
                                   f"{args.registry_host}")
    return instance


def describe(offer: dict) -> str:
    return (f"oferta {offer.get('id')} (máquina {offer.get('machine_id', '?')}): "
            f"{offer.get('num_gpus')}x {offer.get('gpu_name')} "
            f"${offer.get('dph_total', 0):.3f}/h, CUDA {offer.get('cuda_max_good')}, "
            f"fiabilidad {offer.get('reliability', 0):.3f}, {offer.get('geolocation', '?')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Imagen ya publicada en un registro Docker.")
    parser.add_argument("--gpu", type=gpu_preference, default="RTX 4070S,RTX 4070S Ti,RTX 4070 Ti,RTX 4070",
                        help="GPUs aceptables del catálogo Vast.ai, de más a menos deseable, separadas por comas.")
    parser.add_argument("--gpu-count", type=positive_int, default=1)
    parser.add_argument("--label", default="irixslave-transcripcion")
    parser.add_argument("--disk-gb", type=positive_int, default=70)
    parser.add_argument("--max-price", type=float, help="Precio máximo en USD/hora (total).")
    parser.add_argument("--min-reliability", type=float, default=0.95)
    parser.add_argument("--min-download-mbps", type=positive_int, default=200)
    parser.add_argument("--offer-id", type=int, help="Alquila esta oferta en vez de buscar la más barata.")
    parser.add_argument("--exclude-machine", type=positive_int, action="append", default=[],
                        help="No alquilar esta máquina (repetible). Se suma a VAST_EXCLUDE_MACHINES del .env.")
    parser.add_argument("--registry-host", default="docker.io",
                        help="Registro para REGISTRY_USERNAME/REGISTRY_PASSWORD (imagen privada).")
    parser.add_argument("--capabilities", choices=("transcription", "diarization", "transcription,diarization"),
                        default="transcription,diarization")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    parser.add_argument("--create", action="store_true",
                        help="Alquila la instancia (facturable). Sin esta opción solo muestra ofertas y la petición.")
    args = parser.parse_args(argv)
    env = read_env(args.env_file)
    try:
        args.exclude_machine = set(args.exclude_machine) | machine_ids(env.get("VAST_EXCLUDE_MACHINES", ""))
    except ValueError:
        parser.error("VAST_EXCLUDE_MACHINES debe ser una lista de IDs numéricos separados por comas.")
    instance = build_instance(args, env)
    api_key = env.get("VAST_API_KEY", "")
    if api_key.startswith("reemplaza_"):
        api_key = ""

    if not args.create:
        preview = {**instance, "env": {key: "***" if key in SECRET_KEYS else value
                                       for key, value in instance["env"].items()}}
        if "image_login" in preview:
            preview["image_login"] = "***"
        print(json.dumps(preview, ensure_ascii=False, indent=2))

    required = ["VAST_API_KEY", "RADIO_API_TOKEN"]
    if "diarization" in args.capabilities and not env.get("HUGGING_FACE_HUB_TOKEN"):
        required.append("HF_TOKEN")
    missing = [key for key in required if not env.get(key) or env[key].startswith("reemplaza_")]
    if args.create and missing:
        parser.error(f"Falta configurar: {', '.join(missing)}")

    if args.offer_id:
        offers = [{"id": args.offer_id}]
    else:
        try:
            # Buscar ofertas es de solo lectura: un fallo aquí no crea nada.
            found = call_api("POST", "/bundles/", api_key, build_query(args)).get("offers") or []
            offers = rank_offers(found, args.gpu, args.exclude_machine)
        except HTTPError as exc:
            print(f"Vast.ai rechazó la búsqueda (HTTP {exc.code}).", file=sys.stderr)
            print(f"Detalle: {api_error_detail(exc, env)}", file=sys.stderr)
            return 1
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"No se pudo consultar ofertas ({type(exc).__name__}).", file=sys.stderr)
            return 1
        if not offers:
            print("Ninguna oferta cumple los filtros. Prueba otra --gpu, --max-price o --min-reliability.",
                  file=sys.stderr)
            return 1

    if not args.create:
        print("Ofertas que cumplen los filtros, por preferencia de GPU y precio:")
        for offer in offers[:5]:
            print(f"  {describe(offer)}")
        print("Vista previa. Agrega --create para alquilar la primera disponible."
              + ("" if api_key else " Necesitarás VAST_API_KEY."))
        return 0

    for offer in offers[:3]:
        try:
            # No reintentamos la misma oferta: una respuesta perdida podría crear dos instancias.
            result = call_api("PUT", f"/asks/{offer['id']}/", api_key, instance)
        except HTTPError as exc:
            detail = api_error_detail(exc, env)
            if exc.code in OFFER_UNAVAILABLE and not args.offer_id:
                print(f"Oferta {offer['id']} no disponible (HTTP {exc.code}): {detail}", file=sys.stderr)
                continue
            print(f"Vast.ai rechazó el alquiler (HTTP {exc.code}).", file=sys.stderr)
            print(f"Detalle: {detail}", file=sys.stderr)
            if exc.code in (401, 403):
                print("Comprueba la API key de Vast.ai. "
                      "VAST_API_KEY exportada en el sistema tiene prioridad sobre .env.", file=sys.stderr)
            return 1
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"No se pudo confirmar el alquiler ({type(exc).__name__}). "
                  "Revisa la consola Vast.ai antes de repetir.", file=sys.stderr)
            return 1
        instance_id = result.get("new_contract") if isinstance(result, dict) else None
        if not instance_id:
            print("Vast.ai respondió sin ID de instancia. Revisa la consola antes de repetir.", file=sys.stderr)
            return 1
        if "dph_total" in offer:
            print(f"Alquilada {describe(offer)}")
        print(f"Instancia creada: {instance_id}")
        print("https://cloud.vast.ai/instances/")
        print("El contenedor arrancará el worker en modo LIVE con modelos persistentes.")
        return 0
    print("Ninguna de las ofertas seguía disponible. Vuelve a intentarlo.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
