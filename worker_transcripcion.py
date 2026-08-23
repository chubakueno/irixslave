from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

PACKAGE_ROOT = Path(__file__).resolve().parent
ENV_PATH = PACKAGE_ROOT / ".env"

sys.path.insert(0, str(PACKAGE_ROOT))
from pipeline_transcripcion_diarizada import AUDIO_EXTENSIONS  # noqa: E402


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


def getenv(env_file: dict[str, str], key: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.environ.get(key) or env_file.get(key) or default
    if required and not value:
        raise SystemExit(
            f"Falta configurar {key}. Copia .env.example a .env y complétalo, "
            "o exporta la variable de entorno antes de correr el worker."
        )
    return value


@dataclass
class Config:
    base_url: str
    token: str
    worker_id: str
    poll_interval: float
    heartbeat_interval: float
    whisper_model: str
    pyannote_model: str
    device: str
    language: str
    models_dir: Path
    results_dir: Path


def load_config(poll_interval_override: float | None) -> Config:
    env_file = load_dotenv(ENV_PATH)
    poll_interval = poll_interval_override
    if poll_interval is None:
        poll_interval = float(getenv(env_file, "POLL_INTERVAL_SECONDS", "15"))
    return Config(
        base_url=getenv(env_file, "RADIO_BASE_URL", "https://radio.datadaf.com").rstrip("/"),
        token=getenv(env_file, "RADIO_API_TOKEN", required=True),
        worker_id=getenv(env_file, "WORKER_ID", "mac-mini-luis-01"),
        poll_interval=poll_interval,
        heartbeat_interval=float(getenv(env_file, "HEARTBEAT_INTERVAL_SECONDS", "60")),
        whisper_model=getenv(env_file, "WHISPER_MODEL", "large-v3"),
        pyannote_model=getenv(env_file, "PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"),
        device=getenv(env_file, "DEVICE", "cpu"),
        language=getenv(env_file, "TRANSCRIPTION_LANGUAGE", "es"),
        models_dir=PACKAGE_ROOT / "modelos",
        results_dir=PACKAGE_ROOT / "resultados_worker",
    )


def api_headers(cfg: Config, lease_id: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {cfg.token}",
        "X-Worker-Id": cfg.worker_id,
    }
    if lease_id:
        headers["X-Lease-Id"] = lease_id
    return headers


def lease_job(session: requests.Session, cfg: Config) -> dict[str, Any] | None:
    resp = session.post(
        f"{cfg.base_url}/api/internal/transcription/lease",
        headers=api_headers(cfg),
        timeout=30,
    )
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json()


def send_heartbeat(session: requests.Session, cfg: Config, job_id: str, lease_id: str) -> None:
    resp = session.post(
        f"{cfg.base_url}/api/internal/transcription/jobs/{job_id}/heartbeat",
        headers=api_headers(cfg, lease_id),
        timeout=15,
    )
    resp.raise_for_status()


def complete_job(session: requests.Session, cfg: Config, job_id: str, lease_id: str, payload: dict[str, Any]) -> None:
    resp = session.post(
        f"{cfg.base_url}/api/internal/transcription/jobs/{job_id}/complete",
        headers=api_headers(cfg, lease_id),
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()


def fail_job(session: requests.Session, cfg: Config, job_id: str, lease_id: str, error: str, retryable: bool) -> None:
    resp = session.post(
        f"{cfg.base_url}/api/internal/transcription/jobs/{job_id}/fail",
        headers=api_headers(cfg, lease_id),
        json={"error": error, "retryable": retryable},
        timeout=30,
    )
    resp.raise_for_status()


class HeartbeatThread(threading.Thread):
    def __init__(self, session: requests.Session, cfg: Config, job_id: str, lease_id: str, interval: float) -> None:
        super().__init__(daemon=True)
        self._session = session
        self._cfg = cfg
        self._job_id = job_id
        self._lease_id = lease_id
        self._interval = interval
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                send_heartbeat(self._session, self._cfg, self._job_id, self._lease_id)
                print(f"  [heartbeat OK {datetime.now().strftime('%H:%M:%S')}]")
            except Exception as exc:
                print(f"  [heartbeat ERROR: {exc}]", file=sys.stderr)

    def stop(self) -> None:
        self._stop.set()


def guess_suffix(url: str, content_type: str | None) -> str:
    path_suffix = Path(url.split("?")[0]).suffix.lower()
    if path_suffix in AUDIO_EXTENSIONS:
        return path_suffix
    if content_type:
        import mimetypes

        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext
    return ".mp3"


def download_audio(session: requests.Session, url: str, headers: dict[str, str], dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with session.get(url, headers=headers or {}, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        suffix = guess_suffix(url, resp.headers.get("Content-Type"))
        dest = dest_dir / f"audio{suffix}"
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    return dest


def run_local_pipeline(cfg: Config, audio_path: Path, out_dir: Path) -> None:
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "pipeline_transcripcion_diarizada.py"),
        "--input", str(audio_path),
        "--output", str(out_dir),
        "--language", cfg.language,
        "--device", cfg.device,
        "--whisper-model", cfg.whisper_model,
        "--pyannote-model", cfg.pyannote_model,
        "--models-dir", str(cfg.models_dir),
    ]
    completed = subprocess.run(command, cwd=PACKAGE_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"pipeline_transcripcion_diarizada.py terminó con código {completed.returncode}")


def load_results(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    transcript_matches = [
        p for p in (out_dir / "transcripciones").rglob("*.json") if p.name != "_progreso.csv"
    ]
    speakers_matches = list((out_dir / "diarizaciones").rglob("*.speakers.json"))
    if not transcript_matches or not speakers_matches:
        raise RuntimeError("El pipeline local no generó los archivos esperados (revisa los logs arriba).")
    transcript_data = json.loads(transcript_matches[0].read_text(encoding="utf-8"))
    speakers_data = json.loads(speakers_matches[0].read_text(encoding="utf-8"))
    return speakers_data, transcript_data


def build_payload(speakers_data: dict[str, Any], transcript_data: dict[str, Any], cfg: Config) -> dict[str, Any]:
    turns = speakers_data.get("turns") or []
    units = speakers_data.get("units") or []

    text = "\n".join(f"[{turn['speaker']}] {turn['text']}" for turn in turns).strip()
    words = [
        {
            "word": unit.get("text", ""),
            "start": unit.get("start"),
            "end": unit.get("end"),
            "probability": unit.get("probability"),
            "speaker": unit.get("speaker"),
        }
        for unit in units
    ]

    return {
        "text": text,
        "language": transcript_data.get("language") or cfg.language,
        "model": f"faster-whisper-{cfg.whisper_model}+{cfg.pyannote_model.split('/')[-1]}",
        "words": words,
    }


def handle_job(session: requests.Session, cfg: Config, job: dict[str, Any], live: bool) -> None:
    job_id = job.get("job_id") or job.get("id")
    lease_id = job.get("lease_id") or job.get("leaseId")
    audio = job.get("audio") or {}
    audio_url = audio.get("url")
    audio_headers = audio.get("headers") or {}

    if not job_id or not lease_id or not audio_url:
        raise RuntimeError(
            "Respuesta de lease incompleta o con un esquema inesperado:\n"
            + json.dumps(job, ensure_ascii=False, indent=2)[:2000]
        )

    print(f"\n=== Job {job_id} (lease {lease_id}) ===")
    heartbeat = HeartbeatThread(session, cfg, job_id, lease_id, cfg.heartbeat_interval)
    heartbeat.start()
    try:
        with tempfile.TemporaryDirectory(prefix=f"job_{job_id}_") as tmp:
            tmp_path = Path(tmp)
            print("  descargando audio...")
            audio_path = download_audio(session, audio_url, audio_headers, tmp_path / "in")
            size_mb = audio_path.stat().st_size / 1e6
            print(f"  audio: {audio_path.name} ({size_mb:.1f} MB)")

            print("  transcribiendo + diarizando localmente (puede tardar varios minutos en CPU)...")
            out_dir = tmp_path / "out"
            run_local_pipeline(cfg, audio_path, out_dir)

            speakers_data, transcript_data = load_results(out_dir)
            payload = build_payload(speakers_data, transcript_data, cfg)

        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        result_path = cfg.results_dir / f"{job_id}.json"
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  resultado guardado en: {result_path}")

        if not live:
            print("  modo DRY-RUN: no se sube nada a la API todavía.")
            print("  revisa el archivo de arriba y vuelve a correr con --live para subir resultados reales.")
            return

        complete_job(session, cfg, job_id, lease_id, payload)
        print("  subido correctamente (complete).")
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        if live:
            try:
                fail_job(session, cfg, job_id, lease_id, str(exc), retryable=True)
                print("  reportado como fallido (fail, retryable=true).")
            except Exception as fail_exc:
                print(f"  no se pudo reportar el fallo a la API: {fail_exc}", file=sys.stderr)
        raise
    finally:
        heartbeat.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Worker que hace polling de trabajos de transcripción diarizada en radio.datadaf.com"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Sube resultados reales via /complete y /fail. Sin esta bandera solo procesa y guarda localmente.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Procesa un solo job (o detecta que no hay ninguno pendiente) y termina.",
    )
    parser.add_argument("--poll-interval", type=float, help="Segundos entre intentos de lease cuando no hay trabajo.")
    args = parser.parse_args()

    cfg = load_config(args.poll_interval)
    session = requests.Session()

    print(f"Worker '{cfg.worker_id}' -> {cfg.base_url}  (modo: {'LIVE' if args.live else 'DRY-RUN'})")
    while True:
        try:
            job = lease_job(session, cfg)
        except requests.RequestException as exc:
            print(f"Error al pedir trabajo: {exc}", file=sys.stderr)
            job = None

        if job is None:
            print("Sin trabajos pendientes.")
            if args.once:
                return 0
            time.sleep(cfg.poll_interval)
            continue

        try:
            handle_job(session, cfg, job, live=args.live)
        except Exception:
            pass  # ya fue reportado/loggeado dentro de handle_job

        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
