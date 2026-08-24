from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

PACKAGE_ROOT = Path(__file__).resolve().parent
ENV_PATH = PACKAGE_ROOT / ".env"
IS_MACOS = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
STOP_KEY = b"q"

# Carpeta propia (en vez de %TEMP%) para los audios descargados por job. Así la
# exclusión de Windows Defender puede apuntar a esta única carpeta en vez de a
# todo %TEMP%, que comparten otros programas.
WORKER_TMP_DIR = PACKAGE_ROOT / ".worker_tmp"

sys.path.insert(0, str(PACKAGE_ROOT))
from pipeline_transcripcion_diarizada import AUDIO_EXTENSIONS, nvidia_library_dirs  # noqa: E402


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
    transcription_engine: str
    pyannote_model: str
    diarization_engine: str
    device: str
    language: str
    models_dir: Path
    results_dir: Path


def load_config(poll_interval_override: float | None) -> Config:
    env_file = load_dotenv(ENV_PATH)
    # diarizar.py busca el token de Hugging Face en el entorno real (HF_TOKEN /
    # HUGGING_FACE_HUB_TOKEN) antes de caer al caché de "hf auth login". Si vino
    # por .env en vez de por una variable ya exportada, lo exportamos aquí para
    # que tanto el subproceso de pipeline_transcripcion_diarizada.py como el
    # motor persistente (que importa diarizar.py en el mismo proceso) lo vean.
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = env_file.get(key)
        if value and not os.environ.get(key):
            os.environ[key] = value

    # --persistent-models importa diarizar.py/transcribir.py en este mismo
    # proceso (motor_persistente.py), no por subproceso, así que no pasa por
    # runtime_environment(). ctranslate2 necesita estas libs de cualquier
    # forma; hay que exportarlas ANTES de que main() importe motor_persistente
    # (que es cuando ctranslate2 las carga), por eso va aquí y no más abajo.
    lib_dirs = [str(path) for path in nvidia_library_dirs()]
    if lib_dirs:
        path_var = "PATH" if IS_WINDOWS else "LD_LIBRARY_PATH"
        os.environ[path_var] = os.pathsep.join(lib_dirs + [os.environ.get(path_var, "")])

    poll_interval = poll_interval_override
    if poll_interval is None:
        poll_interval = float(getenv(env_file, "POLL_INTERVAL_SECONDS", "15"))

    # Defaults por plataforma: en Apple Silicon usa los motores acelerados
    # (MLX/Soniqo); en Windows/Linux con NVIDIA cae al pipeline original
    # (faster-whisper/pyannote sobre CUDA). Todo overridable vía .env.
    if IS_MACOS:
        default_worker_id = "mac-mini-luis-01"
        default_transcription_engine = "mlx"
        default_whisper_model = "mlx-community/whisper-large-v3-mlx"
        default_diarization_engine = "soniqo"
        default_device = "cpu"
    else:
        default_worker_id = f"worker-{socket.gethostname()}"
        default_transcription_engine = "faster-whisper"
        default_whisper_model = "large-v3"
        default_diarization_engine = "pyannote"
        default_device = "cuda"

    return Config(
        base_url=getenv(env_file, "RADIO_BASE_URL", "https://radio.datadaf.com").rstrip("/"),
        token=getenv(env_file, "RADIO_API_TOKEN", required=True),
        worker_id=getenv(env_file, "WORKER_ID", default_worker_id),
        poll_interval=poll_interval,
        heartbeat_interval=float(getenv(env_file, "HEARTBEAT_INTERVAL_SECONDS", "30")),
        whisper_model=getenv(env_file, "WHISPER_MODEL", default_whisper_model),
        transcription_engine=getenv(env_file, "TRANSCRIPTION_ENGINE", default_transcription_engine),
        pyannote_model=getenv(env_file, "PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"),
        diarization_engine=getenv(env_file, "DIARIZATION_ENGINE", default_diarization_engine),
        device=getenv(env_file, "DEVICE", default_device),
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
                print(f"  [{self._job_id}] [heartbeat OK {datetime.now().strftime('%H:%M:%S')}]")
            except Exception as exc:
                print(f"  [{self._job_id}] [heartbeat ERROR: {exc}]", file=sys.stderr)

    def stop(self) -> None:
        self._stop.set()


class StopKeyWatcher(threading.Thread):
    """Escucha la tecla STOP_KEY sin bloquear stdin, para pedir una detención
    ordenada (termina el job en curso, no toma uno nuevo) sin usar Ctrl+C,
    que sigue matando el proceso de inmediato."""

    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self._stop_requested = stop_event
        self._stop_watching = threading.Event()

    def run(self) -> None:
        try:
            if IS_WINDOWS:
                self._watch_windows()
            else:
                self._watch_posix()
        except Exception:
            pass  # entorno sin consola interactiva (servicio, nohup, etc.): sin tecla, solo Ctrl+C

    def _trigger(self) -> None:
        if not self._stop_requested.is_set():
            print(
                f"\n  [tecla '{STOP_KEY.decode()}' detectada: se terminará el job actual "
                "y no se tomarán más. Ctrl+C sigue cortando de inmediato.]"
            )
        self._stop_requested.set()

    def _watch_windows(self) -> None:
        import msvcrt

        while not self._stop_watching.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                if key == STOP_KEY:
                    self._trigger()
            else:
                time.sleep(0.2)

    def _watch_posix(self) -> None:
        import select
        import termios
        import tty

        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop_watching.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready:
                    key = sys.stdin.read(1).lower().encode()
                    if key == STOP_KEY:
                        self._trigger()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def stop(self) -> None:
        self._stop_watching.set()


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


LOCAL_LOCK_MAX_ATTEMPTS = 4
LOCAL_LOCK_RETRY_DELAY = 5.0
LOCAL_LOCK_MAX_DELAY = 20.0


def is_local_file_lock_error(exc: BaseException) -> bool:
    """WinError 32: archivo bloqueado por otro proceso (típicamente el antivirus
    escaneando el mp3 recién descargado). Es transitorio y local a esta máquina,
    no un fallo real del job — no debería consumir un intento en el servidor.

    Se detecta por el texto porque algunas capas internas (p.ej. ctranslate2/av)
    re-lanzan el error de I/O como una excepción propia sin preservar el
    atributo `winerror`, aunque el mensaje original se conserve."""
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 32:
        return True
    text = str(exc)
    return "WinError 32" in text or "being used by another process" in text


def wait_until_readable(path: Path, attempts: int = 6, initial_delay: float = 0.3) -> None:
    """En Windows, un antivirus (p.ej. Defender) puede tener el archivo recién
    descargado bajo lock mientras lo escanea. Reintenta abrirlo antes de
    devolver el control, en vez de que el motor falle el job de inmediato."""
    delay = initial_delay
    for attempt in range(1, attempts + 1):
        try:
            with path.open("rb"):
                return
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay *= 2


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
    wait_until_readable(dest)
    return dest


def run_local_pipeline(cfg: Config, audio_path: Path, out_dir: Path) -> None:
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "pipeline_transcripcion_diarizada.py"),
        "--input", str(audio_path),
        "--output", str(out_dir),
        "--language", cfg.language,
        "--device", cfg.device,
        "--transcription-engine", cfg.transcription_engine,
        "--whisper-model", cfg.whisper_model,
        "--pyannote-model", cfg.pyannote_model,
        "--diarization-engine", cfg.diarization_engine,
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
    words = []
    for unit in units:
        word: dict[str, Any] = {
            "text": str(unit.get("text", "")).strip(),
            "start": unit.get("start"),
            "end": unit.get("end"),
            "type": "word",
            "speaker_id": unit.get("speaker"),
        }
        probability = unit.get("probability")
        if probability is not None and probability > 0:
            word["logprob"] = math.log(probability)
        words.append(word)

    return {
        "text": text,
        "language": transcript_data.get("language") or cfg.language,
        "model": (
            f"{cfg.transcription_engine}-{cfg.whisper_model.split('/')[-1]}"
            f"+{cfg.diarization_engine}-{cfg.pyannote_model.split('/')[-1]}"
        ),
        "words": words,
    }


@dataclass
class PreparedJob:
    """Un job ya leaseado y con el audio descargado, listo para procesar.
    Cada PreparedJob tiene su propia Session porque se lease/descarga en un
    hilo de pre-descarga mientras el job anterior aún se está procesando (y
    reportando su heartbeat) en el hilo principal; usar una única Session
    compartida entre esos dos hilos concurrentes no es seguro."""

    job: dict[str, Any]
    job_id: str
    lease_id: str
    session: requests.Session
    tmp_dir: tempfile.TemporaryDirectory
    audio_path: Path
    heartbeat: HeartbeatThread


def prepare_job(cfg: Config, live: bool) -> PreparedJob | None:
    """Lease + descarga de audio, sin transcribir/diarizar todavía. Se llama
    tanto para el job actual como, en un hilo aparte, para pre-descargar el
    siguiente mientras el actual se procesa (así se evitan los segundos
    muertos de descarga en los que no se usa ni CPU ni GPU)."""
    session = requests.Session()
    try:
        job = lease_job(session, cfg)
    except requests.RequestException as exc:
        print(f"Error al pedir trabajo: {exc}", file=sys.stderr)
        return None
    if job is None:
        return None

    job_id = job.get("job_id") or job.get("id")
    lease_id = job.get("lease_id") or job.get("leaseId")
    audio = job.get("audio") or {}
    audio_url = audio.get("url")
    audio_headers = audio.get("headers") or {}

    if not job_id or not lease_id or not audio_url:
        print(
            "Respuesta de lease incompleta o con un esquema inesperado:\n"
            + json.dumps(job, ensure_ascii=False, indent=2)[:2000],
            file=sys.stderr,
        )
        return None

    heartbeat = HeartbeatThread(session, cfg, job_id, lease_id, cfg.heartbeat_interval)
    heartbeat.start()

    WORKER_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.TemporaryDirectory(
        prefix=f"job_{job_id}_", dir=str(WORKER_TMP_DIR), ignore_cleanup_errors=True
    )
    try:
        print(f"  [{job_id}] descargando audio por adelantado...")
        audio_path = download_audio(session, audio_url, audio_headers, Path(tmp_dir.name) / "in")
        size_mb = audio_path.stat().st_size / 1e6
        print(f"  [{job_id}] audio listo: {audio_path.name} ({size_mb:.1f} MB)")
    except Exception as exc:
        heartbeat.stop()
        print(f"  [{job_id}] ERROR al descargar: {exc}", file=sys.stderr)
        traceback.print_exc()
        if live:
            try:
                fail_job(session, cfg, job_id, lease_id, str(exc), retryable=True)
                print(f"  [{job_id}] reportado como fallido (fail, retryable=true).")
            except Exception as fail_exc:
                print(f"  [{job_id}] no se pudo reportar el fallo a la API: {fail_exc}", file=sys.stderr)
        tmp_dir.cleanup()
        return None

    return PreparedJob(
        job=job,
        job_id=job_id,
        lease_id=lease_id,
        session=session,
        tmp_dir=tmp_dir,
        audio_path=audio_path,
        heartbeat=heartbeat,
    )


def process_prepared_job(cfg: Config, prepared: PreparedJob, live: bool, engine: Any | None = None) -> None:
    job_id = prepared.job_id
    lease_id = prepared.lease_id
    session = prepared.session

    print(f"\n=== Job {job_id} (lease {lease_id}) ===")
    try:
        tmp_path = Path(prepared.tmp_dir.name)
        print(f"  audio ya descargado: {prepared.audio_path.name}")

        print("  transcribiendo + diarizando localmente (puede tardar varios minutos en CPU)...")
        out_dir = tmp_path / "out"
        attempt = 1
        while True:
            try:
                if engine is not None:
                    speakers_data, transcript_data = engine.process(
                        prepared.audio_path, out_dir, language=cfg.language
                    )
                else:
                    run_local_pipeline(cfg, prepared.audio_path, out_dir)
                    speakers_data, transcript_data = load_results(out_dir)
                break
            except Exception as exc:
                if is_local_file_lock_error(exc):
                    print(
                        f"  archivo bloqueado localmente (WinError 32) en el intento {attempt}. "
                        "Traceback completo para diagnóstico:",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
                    if attempt < LOCAL_LOCK_MAX_ATTEMPTS:
                        delay = min(LOCAL_LOCK_RETRY_DELAY * attempt, LOCAL_LOCK_MAX_DELAY)
                        print(
                            f"  reintento local {attempt}/{LOCAL_LOCK_MAX_ATTEMPTS - 1} "
                            f"en {delay:.0f}s...",
                            file=sys.stderr,
                        )
                        time.sleep(delay)
                        attempt += 1
                        continue
                raise
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
        traceback.print_exc()
        if live:
            try:
                fail_job(session, cfg, job_id, lease_id, str(exc), retryable=True)
                print("  reportado como fallido (fail, retryable=true).")
            except Exception as fail_exc:
                print(f"  no se pudo reportar el fallo a la API: {fail_exc}", file=sys.stderr)
        raise
    finally:
        prepared.heartbeat.stop()
        prepared.tmp_dir.cleanup()
        prepared.session.close()


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
    parser.add_argument(
        "--persistent-models",
        action="store_true",
        help=(
            "Carga Whisper y Pyannote una sola vez al arrancar y los mantiene en memoria entre "
            "jobs (evita recargar el modelo en cada transcripción). Solo soporta "
            "transcription-engine=faster-whisper y diarization-engine=pyannote."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.poll_interval)

    engine = None
    if args.persistent_models:
        if cfg.transcription_engine != "faster-whisper" or cfg.diarization_engine != "pyannote":
            raise SystemExit(
                "--persistent-models solo soporta transcription-engine=faster-whisper y "
                f"diarization-engine=pyannote (actual: {cfg.transcription_engine}/{cfg.diarization_engine})."
            )
        from motor_persistente import PersistentEngine

        compute_type = "int8_float16" if cfg.device == "cuda" else "int8"
        batch_size = 12 if cfg.device == "cuda" else 0
        print("Cargando modelos en memoria (whisper + pyannote)...")
        engine = PersistentEngine(
            whisper_model=cfg.whisper_model,
            device=cfg.device,
            compute_type=compute_type,
            models_dir=cfg.models_dir,
            batch_size=batch_size,
            beam_size=1,
            pyannote_model=cfg.pyannote_model,
        )

    stop_requested = threading.Event()
    key_watcher: StopKeyWatcher | None = None
    if not args.once:
        key_watcher = StopKeyWatcher(stop_requested)
        key_watcher.start()
        print(f"(Presiona '{STOP_KEY.decode()}' para detener el worker tras el job actual.)")

    print(f"Worker '{cfg.worker_id}' -> {cfg.base_url}  (modo: {'LIVE' if args.live else 'DRY-RUN'})")
    try:
        current: PreparedJob | None = None
        while True:
            if current is None:
                if stop_requested.is_set():
                    print("Detención solicitada: no se tomarán más trabajos. Saliendo.")
                    return 0
                current = prepare_job(cfg, args.live)
                if current is None:
                    print("Sin trabajos pendientes.")
                    if args.once:
                        return 0
                    time.sleep(cfg.poll_interval)
                    continue

            # Mientras se procesa `current` (CPU/GPU), pre-descargamos el
            # siguiente job en un hilo aparte para que ya esté listo al
            # terminar (evita el hueco muerto de descarga entre jobs).
            prefetch_thread: threading.Thread | None = None
            prefetch_box: list[PreparedJob | None] = [None]
            if not args.once and not stop_requested.is_set():
                def _prefetch() -> None:
                    prefetch_box[0] = prepare_job(cfg, args.live)

                prefetch_thread = threading.Thread(target=_prefetch, daemon=True)
                prefetch_thread.start()

            try:
                process_prepared_job(cfg, current, live=args.live, engine=engine)
            except Exception:
                pass  # ya fue reportado/loggeado dentro de process_prepared_job

            if args.once:
                return 0

            if prefetch_thread is not None:
                prefetch_thread.join()
                current = prefetch_box[0]
            else:
                current = None
    finally:
        if key_watcher:
            key_watcher.stop()


if __name__ == "__main__":
    raise SystemExit(main())
