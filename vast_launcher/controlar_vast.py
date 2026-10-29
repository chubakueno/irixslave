"""Controlador externo: destruye UNA instancia tras una cola vacía sostenida."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from lanzar_vast import API_BASE, read_env


class NoRedirect(HTTPRedirectHandler):
    # No reenviar Authorization a un destino indicado por una redirección.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def api(method, url, token, worker_id=None):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json",
               "User-Agent": "irixslave-controller/1.0"}
    if worker_id:
        headers["X-Worker-Id"] = worker_id
    request = Request(url, method=method, headers=headers)
    with build_opener(NoRedirect()).open(request, timeout=30) as response:
        return json.load(response)


def queue_empty(status, capabilities):
    if not isinstance(status, dict) or set(status.get("capabilities", [])) != set(capabilities):
        raise ValueError("Capacidades inesperadas en Radio")
    for key in ("available_jobs", "active_leases"):
        value = status.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Campo inválido: {key}")
    if "next_available_at" not in status:
        raise ValueError("Falta next_available_at")
    return (status["available_jobs"] == 0 and status["active_leases"] == 0
            and status["next_available_at"] is None)


class IdleWindow:
    def __init__(self, seconds, max_gap):
        self.seconds, self.max_gap = seconds, max_gap
        self.since = self.previous = None

    def reset(self):
        self.since = self.previous = None

    def observe(self, empty, now):
        if not empty:
            self.reset()
            return False
        if self.previous is None or now - self.previous > self.max_gap or now < self.previous:
            self.since = now
        self.previous = now
        return now - self.since >= self.seconds


def identity(instance, args):
    if not isinstance(instance, dict):
        return False
    return (instance.get("id") == args.instance
            and instance.get("machine_id") == args.expect_machine
            and instance.get("label") == args.expect_label
            and instance.get("image_uuid") == args.expect_image)


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent.parent / ".env")
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--expect-machine", type=int, required=True)
    parser.add_argument("--expect-label", default="irixslave-transcripcion")
    parser.add_argument("--expect-image", default="chubakueno/irixslave:runpod-cu128")
    parser.add_argument("--capabilities", default="transcription,diarization")
    parser.add_argument("--idle-seconds", type=float, default=600)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true", help="Una consulta, sin destruir.")
    parser.add_argument("--destroy", action="store_true", help="Habilita destrucción; default: simulación.")
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v > 0 for v in (args.idle_seconds, args.poll_seconds)):
        parser.error("Los intervalos deben ser positivos y finitos")
    caps = args.capabilities.split(",")
    if not caps or any(c not in {"transcription", "diarization", "voice_recognition"} for c in caps):
        parser.error("Capacidades inválidas")
    env = read_env(args.env_file)
    for key in ("RADIO_API_TOKEN", "VAST_API_KEY"):
        if not env.get(key) or env[key].startswith("reemplaza_"):
            parser.error(f"Falta configurar {key}")
    base = env.get("RADIO_BASE_URL", "https://radio.datadaf.com").rstrip("/")
    if urlsplit(base).scheme != "https" or urlsplit(base).username:
        parser.error("RADIO_BASE_URL debe usar HTTPS sin credenciales en la URL")
    status_url = base + "/api/internal/jobs/status?" + urlencode({"capabilities": ",".join(caps)})
    instance_url = f"{API_BASE}/instances/{args.instance}/"
    # Un solo proceso por instancia. El directorio queda junto al .env, solo en la Mac.
    state_dir = args.env_file.resolve().parent / ".vast-controller"
    state_dir.mkdir(mode=0o700, exist_ok=True)
    with (state_dir / f"{args.instance}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("Ya hay un controlador para esta instancia")
            return 0
        attempted = state_dir / f"{args.instance}.delete-attempted"
        if attempted.exists():
            log("Ya hubo un intento de destrucción; revisar Vast antes de rearmar")
            return 0
        window = IdleWindow(args.idle_seconds, args.poll_seconds * 2 + 30)
        log(f"Instancia {args.instance}; inactividad {args.idle_seconds:g}s; "
            + ("DESTRUCCIÓN habilitada" if args.destroy and not args.once else "simulación"))
        while True:
            try:
                # time.time detecta también saltos por suspensión en macOS.
                started = time.time()
                status = api("GET", status_url, env["RADIO_API_TOKEN"],
                             f"vast-controller-mac-{args.instance}")
                empty = queue_empty(status, caps)
                data = api("GET", instance_url, env["VAST_API_KEY"])
                instance = data.get("instances")
                if not identity(instance, args):
                    log("Identidad de instancia distinta o ausente; controlador desarmado")
                    return 0
                if instance.get("actual_status") != "running":
                    window.reset()
                    log("Instancia no está running; contador reiniciado")
                else:
                    now = time.time()
                    if now - started > args.poll_seconds * 2:
                        window.reset()
                    ready = window.observe(empty, now)
                    elapsed = 0 if window.since is None else int(now - window.since)
                    log(f"Jobs={status['available_jobs']}; leases={status['active_leases']}; "
                        f"reintento={'sí' if status['next_available_at'] else 'no'}; vacío={elapsed}s")
                    if ready and args.destroy and not args.once:
                        # Última comprobación de la cola antes del DELETE. Sigue existiendo
                        # una carrera aceptada: un lease nuevo se recupera al vencer.
                        fresh = api("GET", status_url, env["RADIO_API_TOKEN"],
                                    f"vast-controller-mac-{args.instance}")
                        if not queue_empty(fresh, caps):
                            window.reset()
                            continue
                        # Persiste ANTES de mutar: ni un reinicio de macOS debe
                        # repetir un DELETE cuyo resultado pudo perderse.
                        with attempted.open("x") as marker:
                            marker.write(f"{args.instance}\n")
                            marker.flush()
                            os.fsync(marker.fileno())
                        log(f"Solicitando destrucción de {args.instance}")
                        try:
                            result = api("DELETE", instance_url, env["VAST_API_KEY"])
                            if result.get("success") is not True:
                                log("Destrucción no confirmada; desarmado. Revisar Vast manualmente")
                            else:
                                log(f"Instancia {args.instance} destruida")
                        except Exception:
                            # No repetir una mutación de resultado incierto automáticamente.
                            log("Resultado de destrucción incierto; desarmado. Revisar Vast manualmente")
                        return 0
            except HTTPError as exc:
                window.reset()
                if exc.code == 404 and exc.url.rstrip("/") == instance_url.rstrip("/"):
                    log("Instancia ya no existe; controlador finalizado")
                    return 0
                log(f"HTTP {exc.code}; contador reiniciado")
                if args.once:
                    return 1
            except Exception as exc:
                window.reset()
                # No imprimir cuerpos HTTP, URLs firmadas ni credenciales reflejadas.
                log(f"Error {type(exc).__name__}; contador reiniciado")
                if args.once:
                    return 1
            if args.once:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
