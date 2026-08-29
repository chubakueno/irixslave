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

_stdlib_print = print


def print(*args: Any, **kwargs: Any) -> None:  # noqa: A001 -- sella cada log del worker con la hora
    """Antepone `[HH:MM:SS]` a cada línea que imprime el worker, para que todos
    los logs (no solo el heartbeat) lleven hora. Si el texto tiene saltos de
    línea, sella cada una."""
    stamp = datetime.now().strftime("%H:%M:%S")
    sep = kwargs.get("sep", " ")
    text = sep.join(str(a) for a in args)
    stamped = "\n".join(f"[{stamp}] {line}" for line in text.split("\n"))
    _stdlib_print(stamped, **{k: v for k, v in kwargs.items() if k != "sep"})

# Salidas que este worker sabe producir. La API de jobs negocia por aquí: el
# worker anuncia sus `capabilities` al pedir trabajo y solo recibe jobs cuyos
# `requested_outputs` estén contenidos en ellas.
TRANSCRIPTION = "transcription"
DIARIZATION = "diarization"
ALL_CAPABILITIES = (TRANSCRIPTION, DIARIZATION)


def parse_capabilities(raw: str | None) -> list[str]:
    """Convierte "transcription,diarization" (env o CLI) en una lista validada,
    preservando el orden canónico de ALL_CAPABILITIES."""
    if not raw:
        return list(ALL_CAPABILITIES)
    wanted = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = wanted - set(ALL_CAPABILITIES)
    if unknown:
        raise SystemExit(
            f"Capacidad(es) no reconocida(s): {', '.join(sorted(unknown))}. "
            f"Válidas: {', '.join(ALL_CAPABILITIES)}."
        )
    if not wanted:
        raise SystemExit("La lista de capacidades quedó vacía.")
    return [item for item in ALL_CAPABILITIES if item in wanted]


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
    compute_type: str
    whisper_batch_size: int
    segmentation_batch_size: int
    embedding_batch_size: int
    capabilities: list[str]


def load_config(
    poll_interval_override: float | None,
    capabilities_override: list[str] | None = None,
) -> Config:
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

    capabilities = capabilities_override or parse_capabilities(
        getenv(env_file, "WORKER_CAPABILITIES", None)
    )

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
        # Vacío => se deriva por dispositivo (int8_float16 en CUDA, int8 en CPU).
        compute_type=getenv(env_file, "COMPUTE_TYPE", ""),
        whisper_batch_size=int(getenv(env_file, "WHISPER_BATCH_SIZE", "12")),
        segmentation_batch_size=int(getenv(env_file, "PYANNOTE_SEGMENTATION_BATCH_SIZE", "6")),
        embedding_batch_size=int(getenv(env_file, "PYANNOTE_EMBEDDING_BATCH_SIZE", "16")),
        capabilities=capabilities,
    )


def api_headers(cfg: Config, lease_id: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {cfg.token}",
        "X-Worker-Id": cfg.worker_id,
    }
    if lease_id:
        headers["X-Lease-Id"] = lease_id
    return headers


def job_action_url(cfg: Config, job_id: str, action: str) -> str:
    return f"{cfg.base_url}/api/internal/jobs/audio-processing/{job_id}/{action}"


def lease_job(session: requests.Session, cfg: Config) -> dict[str, Any] | None:
    resp = session.post(
        f"{cfg.base_url}/api/internal/jobs/lease",
        headers=api_headers(cfg),
        json={"capabilities": cfg.capabilities},
        timeout=30,
    )
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json()


def send_heartbeat(session: requests.Session, cfg: Config, job_id: str, lease_id: str) -> None:
    resp = session.post(
        job_action_url(cfg, job_id, "heartbeat"),
        headers=api_headers(cfg, lease_id),
        timeout=15,
    )
    resp.raise_for_status()


def complete_job(session: requests.Session, cfg: Config, job_id: str, lease_id: str, payload: dict[str, Any]) -> None:
    resp = session.post(
        job_action_url(cfg, job_id, "complete"),
        headers=api_headers(cfg, lease_id),
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()


def fail_job(session: requests.Session, cfg: Config, job_id: str, lease_id: str, error: str, retryable: bool) -> None:
    resp = session.post(
        job_action_url(cfg, job_id, "fail"),
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
                print(f"  [{self._job_id}] heartbeat OK")
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


def run_local_pipeline(
    cfg: Config,
    audio_path: Path,
    out_dir: Path,
    *,
    want_transcription: bool,
    want_diarization: bool,
) -> None:
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
        "--batch-size", str(cfg.whisper_batch_size),
        "--segmentation-batch-size", str(cfg.segmentation_batch_size),
        "--embedding-batch-size", str(cfg.embedding_batch_size),
    ]
    if not want_transcription:
        command.append("--skip-transcription")
    if not want_diarization:
        command.append("--skip-diarization")
    if cfg.compute_type:
        command += ["--compute-type", cfg.compute_type]
    completed = subprocess.run(command, cwd=PACKAGE_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"pipeline_transcripcion_diarizada.py terminó con código {completed.returncode}")


def _first_json(folder: Path, pattern: str) -> dict[str, Any] | None:
    matches = [p for p in folder.rglob(pattern) if p.name != "_progreso.csv"]
    if not matches:
        return None
    return json.loads(matches[0].read_text(encoding="utf-8"))


def load_results(
    out_dir: Path, *, want_transcription: bool, want_diarization: bool
) -> dict[str, Any]:
    """Lee los archivos que dejó el pipeline local. Solo exige los que
    corresponden a los `requested_outputs` del job."""
    transcript_data = _first_json(out_dir / "transcripciones", "*.json") if want_transcription else None
    diarization_data = _first_json(out_dir / "diarizaciones", "*.diarization.json") if want_diarization else None
    # El .speakers.json (transcripción alineada con hablante) solo existe cuando
    # se pidieron ambas salidas.
    speakers_data = (
        _first_json(out_dir / "diarizaciones", "*.speakers.json")
        if want_transcription and want_diarization
        else None
    )

    missing = []
    if want_transcription and transcript_data is None:
        missing.append("transcripciones/*.json")
    if want_diarization and diarization_data is None:
        missing.append("diarizaciones/*.diarization.json")
    if want_transcription and want_diarization and speakers_data is None:
        missing.append("diarizaciones/*.speakers.json")
    if missing:
        raise RuntimeError(
            "El pipeline local no generó: " + ", ".join(missing) + " (revisa los logs arriba)."
        )

    return {
        "transcript_data": transcript_data,
        "diarization_data": diarization_data,
        "speakers_data": speakers_data,
    }


def _word_from_unit(unit: dict[str, Any], *, with_speaker: bool) -> dict[str, Any]:
    word: dict[str, Any] = {
        "text": str(unit.get("text") or unit.get("word") or "").strip(),
        "start": unit.get("start"),
        "end": unit.get("end"),
        "type": "word",
    }
    if with_speaker:
        word["speaker_id"] = unit.get("speaker")
    probability = unit.get("probability")
    if probability is not None and probability > 0:
        word["logprob"] = math.log(probability)
    return word


def whisper_model_label(whisper_model: str) -> str:
    name = whisper_model.split("/")[-1]
    return name if name.lower().startswith("whisper") else f"whisper-{name}"


def pyannote_model_label(pyannote_model: str) -> str:
    name = pyannote_model.split("/")[-1]
    return name if name.lower().startswith("pyannote") else f"pyannote-{name}"


def build_transcription_output(
    cfg: Config,
    transcript_data: dict[str, Any],
    speakers_data: dict[str, Any] | None,
) -> dict[str, Any]:
    if speakers_data is not None:
        # Con diarización: texto por turno con etiqueta y palabras con speaker_id.
        turns = speakers_data.get("turns") or []
        units = speakers_data.get("units") or []
        text = "\n".join(f"[{turn['speaker']}] {turn['text']}" for turn in turns).strip()
        words = [_word_from_unit(u, with_speaker=True) for u in units]
    else:
        # Solo transcripción: un segmento de Whisper por línea, sin hablante.
        segments = transcript_data.get("segments") or []
        text = "\n".join(
            " ".join(str(seg.get("text", "")).split()) for seg in segments
        ).strip()
        words = [
            _word_from_unit(w, with_speaker=False)
            for seg in segments
            for w in (seg.get("words") or [])
            if w.get("start") is not None and w.get("end") is not None
        ]
    return {
        "text": text,
        "language": transcript_data.get("language") or cfg.language,
        "model": whisper_model_label(cfg.whisper_model),
        "words": words,
    }


def build_diarization_output(
    cfg: Config,
    diarization_data: dict[str, Any],
    speakers_data: dict[str, Any] | None,
) -> dict[str, Any]:
    processing_seconds = diarization_data.get("processing_seconds")
    metadata: dict[str, Any] = {}
    if speakers_data is not None:
        # Con transcripción: los "segmentos" son los turnos alineados (con texto).
        turns = speakers_data.get("turns") or []
        segments = [
            {
                "speaker_id": turn["speaker"],
                "start": turn["start"],
                "end": turn["end"],
                "text": turn["text"],
            }
            for turn in turns
        ]
        speaker_count = len(diarization_data.get("speakers") or speakers_data.get("speakers") or [])
    else:
        # Solo diarización: intervalos exclusivos (un hablante por instante), sin texto.
        segments = [
            {"speaker_id": item["speaker"], "start": item["start"], "end": item["end"]}
            for item in (diarization_data.get("exclusive_diarization") or [])
        ]
        speaker_count = diarization_data.get("num_speakers")
        if speaker_count is None:
            speaker_count = len(diarization_data.get("speakers") or [])

    metadata["speaker_count"] = speaker_count
    if processing_seconds is not None:
        metadata["processing_time_ms"] = round(float(processing_seconds) * 1000)

    return {
        "model": pyannote_model_label(diarization_data.get("model") or cfg.pyannote_model),
        "segments": segments,
        "metadata": metadata,
    }


def build_payload(
    cfg: Config,
    *,
    want_transcription: bool,
    want_diarization: bool,
    transcript_data: dict[str, Any] | None,
    diarization_data: dict[str, Any] | None,
    speakers_data: dict[str, Any] | None,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    if want_transcription:
        outputs["transcription"] = build_transcription_output(cfg, transcript_data or {}, speakers_data)
    if want_diarization:
        outputs["diarization"] = build_diarization_output(cfg, diarization_data or {}, speakers_data)
    return {"outputs": outputs}


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
    requested_outputs: list[str]
    session: requests.Session
    tmp_dir: tempfile.TemporaryDirectory
    audio_path: Path
    heartbeat: HeartbeatThread


def resolve_requested_outputs(job: dict[str, Any], cfg: Config) -> list[str]:
    """Determina qué salidas pide el job. Usa `requested_outputs` si viene; si no,
    lo deriva de `options` (`transcribe`/`diarize`). Devuelve la lista en el orden
    canónico y falla si pide algo fuera de las capacidades del worker."""
    raw = job.get("requested_outputs")
    if raw:
        wanted = {str(item).strip().lower() for item in raw}
    else:
        options = job.get("options") or {}
        wanted = set()
        if options.get("transcribe"):
            wanted.add(TRANSCRIPTION)
        if options.get("diarize"):
            wanted.add(DIARIZATION)
    unknown = wanted - set(ALL_CAPABILITIES)
    if unknown:
        raise ValueError(f"requested_outputs desconocido(s): {', '.join(sorted(unknown))}")
    if not wanted:
        raise ValueError("el job no pide ninguna salida (requested_outputs vacío)")
    outside = wanted - set(cfg.capabilities)
    if outside:
        raise ValueError(
            f"el job pide {', '.join(sorted(outside))} pero este worker solo anuncia "
            f"{', '.join(cfg.capabilities)}"
        )
    return [item for item in ALL_CAPABILITIES if item in wanted]


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

    try:
        requested_outputs = resolve_requested_outputs(job, cfg)
    except ValueError as exc:
        print(f"  [{job_id}] job rechazado: {exc}", file=sys.stderr)
        if live:
            try:
                fail_job(session, cfg, job_id, lease_id, str(exc), retryable=False)
            except Exception as fail_exc:
                print(f"  [{job_id}] no se pudo reportar el rechazo: {fail_exc}", file=sys.stderr)
        return None
    print(f"  [{job_id}] salidas pedidas: {', '.join(requested_outputs)}")

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
        requested_outputs=requested_outputs,
        session=session,
        tmp_dir=tmp_dir,
        audio_path=audio_path,
        heartbeat=heartbeat,
    )


def _finish_job(cfg: Config, prepared: PreparedJob, payload: dict[str, Any]) -> None:
    """Sube el resultado de un job a /complete y libera sus recursos (heartbeat,
    carpeta temporal, sesión HTTP). Lo llama el hilo `advance` del loop, no el
    hilo principal: el POST puede tardar (payload con miles de `words`)."""
    job_id, lease_id, session = prepared.job_id, prepared.lease_id, prepared.session
    try:
        complete_job(session, cfg, job_id, lease_id, payload)
        print(f"  [{job_id}] subido correctamente (complete).", flush=True)
    except Exception as exc:
        print(f"  [{job_id}] ERROR al subir el resultado: {exc}", file=sys.stderr)
        traceback.print_exc()
        try:
            fail_job(session, cfg, job_id, lease_id, str(exc), retryable=True)
            print(f"  [{job_id}] reportado como fallido (fail, retryable=true).", file=sys.stderr)
        except Exception as fail_exc:
            print(f"  [{job_id}] no se pudo reportar el fallo a la API: {fail_exc}", file=sys.stderr)
    finally:
        prepared.heartbeat.stop()
        prepared.tmp_dir.cleanup()
        prepared.session.close()


def process_prepared_job(
    cfg: Config, prepared: PreparedJob, live: bool, engine: Any | None = None
) -> dict[str, Any] | None:
    """Corre las etapas del job (parte que usa GPU/CPU, bloqueante) y devuelve el
    payload listo para subir. En dry-run limpia los recursos del job y devuelve
    None. La subida (y la limpieza, en modo live) las hace el llamador vía
    _finish_job. Si el cómputo falla, reporta /fail, limpia y relanza."""
    job_id = prepared.job_id
    lease_id = prepared.lease_id
    session = prepared.session

    want_transcription = TRANSCRIPTION in prepared.requested_outputs
    want_diarization = DIARIZATION in prepared.requested_outputs

    print(f"\n=== Job {job_id} (lease {lease_id}) — {', '.join(prepared.requested_outputs)} ===")
    try:
        tmp_path = Path(prepared.tmp_dir.name)
        print(f"  audio ya descargado: {prepared.audio_path.name}")

        etapas = " + ".join(prepared.requested_outputs)
        print(f"  procesando localmente ({etapas}) (puede tardar varios minutos en CPU)...")
        out_dir = tmp_path / "out"
        attempt = 1
        while True:
            try:
                if engine is not None:
                    results = engine.process(
                        prepared.audio_path,
                        out_dir,
                        language=cfg.language,
                        want_transcription=want_transcription,
                        want_diarization=want_diarization,
                    )
                else:
                    run_local_pipeline(
                        cfg,
                        prepared.audio_path,
                        out_dir,
                        want_transcription=want_transcription,
                        want_diarization=want_diarization,
                    )
                    results = load_results(
                        out_dir,
                        want_transcription=want_transcription,
                        want_diarization=want_diarization,
                    )
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
        payload = build_payload(
            cfg,
            want_transcription=want_transcription,
            want_diarization=want_diarization,
            **results,
        )

        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        result_path = cfg.results_dir / f"{job_id}.json"
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  resultado guardado en: {result_path}")

        if not live:
            print("  modo DRY-RUN: no se sube nada a la API todavía.")
            print("  revisa el archivo de arriba y vuelve a correr con --live para subir resultados reales.")
            prepared.heartbeat.stop()
            prepared.tmp_dir.cleanup()
            prepared.session.close()
            return None

        return payload
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        if live:
            try:
                fail_job(session, cfg, job_id, lease_id, str(exc), retryable=True)
                print("  reportado como fallido (fail, retryable=true).")
            except Exception as fail_exc:
                print(f"  no se pudo reportar el fallo a la API: {fail_exc}", file=sys.stderr)
        prepared.heartbeat.stop()
        prepared.tmp_dir.cleanup()
        prepared.session.close()
        raise


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
    capability_group = parser.add_mutually_exclusive_group()
    capability_group.add_argument(
        "--transcription-only",
        action="store_true",
        help="Solo toma jobs de transcripción (anuncia capabilities=[transcription]).",
    )
    capability_group.add_argument(
        "--diarization-only",
        action="store_true",
        help="Solo toma jobs de diarización (anuncia capabilities=[diarization]).",
    )
    capability_group.add_argument(
        "--capabilities",
        help=(
            "Lista separada por comas de las salidas que este worker acepta "
            "(transcription, diarization). Default: ambas, o WORKER_CAPABILITIES del .env."
        ),
    )
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

    capabilities_override: list[str] | None = None
    if args.transcription_only:
        capabilities_override = [TRANSCRIPTION]
    elif args.diarization_only:
        capabilities_override = [DIARIZATION]
    elif args.capabilities:
        capabilities_override = parse_capabilities(args.capabilities)

    cfg = load_config(args.poll_interval, capabilities_override)
    print(f"Capacidades del worker: {', '.join(cfg.capabilities)}")

    engine = None
    if args.persistent_models:
        if cfg.transcription_engine != "faster-whisper" or cfg.diarization_engine != "pyannote":
            raise SystemExit(
                "--persistent-models solo soporta transcription-engine=faster-whisper y "
                f"diarization-engine=pyannote (actual: {cfg.transcription_engine}/{cfg.diarization_engine})."
            )
        from motor_persistente import PersistentEngine

        compute_type = cfg.compute_type or (
            "int8_float16" if cfg.device == "cuda" else "int8"
        )
        batch_size = cfg.whisper_batch_size if cfg.device == "cuda" else 0
        # Solo carga el modelo de las capacidades anunciadas: un worker
        # --transcription-only no paga la VRAM ni el arranque de Pyannote, y
        # viceversa.
        load_whisper = TRANSCRIPTION in cfg.capabilities
        load_pyannote = DIARIZATION in cfg.capabilities
        cuales = " + ".join(
            name for name, on in (("whisper", load_whisper), ("pyannote", load_pyannote)) if on
        )
        print(f"Cargando modelos en memoria ({cuales})...")
        engine = PersistentEngine(
            whisper_model=cfg.whisper_model,
            device=cfg.device,
            compute_type=compute_type,
            models_dir=cfg.models_dir,
            batch_size=batch_size,
            beam_size=1,
            pyannote_model=cfg.pyannote_model,
            segmentation_batch_size=cfg.segmentation_batch_size,
            embedding_batch_size=cfg.embedding_batch_size,
            load_whisper=load_whisper,
            load_pyannote=load_pyannote,
        )

    stop_requested = threading.Event()
    key_watcher: StopKeyWatcher | None = None
    if not args.once:
        key_watcher = StopKeyWatcher(stop_requested)
        key_watcher.start()
        print(f"(Presiona '{STOP_KEY.decode()}' para detener el worker tras el job actual.)")

    # Un único hilo de fondo, `advance`, hace TODO lo que va entre jobs y no usa
    # GPU, en este orden: (1) sube a /complete el resultado del job recién
    # terminado y libera sus recursos, (2) pide el siguiente lease, (3)
    # pre-descarga su MP3. Así, mientras el hilo principal procesa el job N+1 en
    # la GPU, `advance` cierra el N y trae el N+2. Solo 2 hilos, y el lease del
    # N+2 nunca ocurre antes de que la subida del N haya terminado (van seguidos
    # en el mismo hilo, y el principal hace join antes de avanzar). Por eso el
    # worker sostiene como mucho 2 leases a la vez: el que procesa + el que
    # `advance` esté tocando en ese instante (subiendo O leaseando, nunca ambos).
    advance_box: list[PreparedJob | None] = [None]

    def start_advance(pending_upload: tuple[PreparedJob, dict[str, Any]] | None) -> threading.Thread | None:
        """Lanza el hilo de fondo. `pending_upload` es (prepared, payload) del job
        recién procesado, o None cuando no hay job previo que subir (arranque, o
        tras un rato sin trabajo)."""
        fetch = not args.once and not stop_requested.is_set()
        if pending_upload is None and not fetch:
            return None
        advance_box[0] = None

        def _run() -> None:
            if pending_upload is not None:
                _finish_job(cfg, pending_upload[0], pending_upload[1])
            if fetch:
                try:
                    advance_box[0] = prepare_job(cfg, args.live)
                except Exception as exc:  # que un fallo raro del prefetch no mate el hilo en silencio
                    print(f"  prefetch del siguiente job falló: {exc}", file=sys.stderr)
                    traceback.print_exc()

        thread = threading.Thread(target=_run, name="advance", daemon=True)
        thread.start()
        return thread

    print(f"Worker '{cfg.worker_id}' -> {cfg.base_url}  (modo: {'LIVE' if args.live else 'DRY-RUN'})")
    advance: threading.Thread | None = None
    try:
        current: PreparedJob | None = None
        while True:
            if current is None:
                if advance is not None:
                    advance.join()  # deja terminar la subida del último job
                    current = advance_box[0]
                    advance = None
                if current is None:
                    if stop_requested.is_set():
                        print("Detención solicitada: no quedan trabajos en curso. Saliendo.")
                        return 0
                    current = prepare_job(cfg, args.live)
                if current is None:
                    print("Sin trabajos pendientes.")
                    if args.once:
                        return 0
                    time.sleep(cfg.poll_interval)
                    continue

            # Asegura que `advance` esté trayendo el sucesor de `current` mientras
            # lo procesamos. En la primera vuelta no hay nada que subir todavía.
            if advance is None:
                advance = start_advance(pending_upload=None)

            try:
                payload = process_prepared_job(cfg, current, live=args.live, engine=engine)
            except Exception:
                payload = None  # ya reportado y limpiado dentro de process_prepared_job

            following: PreparedJob | None = None
            if advance is not None:
                advance.join()
                following = advance_box[0]
                advance = None

            if payload is not None and args.live:
                if args.once:
                    _finish_job(cfg, current, payload)
                else:
                    advance = start_advance(pending_upload=(current, payload))

            if args.once:
                if advance is not None:
                    advance.join()
                return 0

            current = following
    finally:
        if key_watcher:
            key_watcher.stop()
        # Ctrl+C / excepción: da un margen acotado a la subida en vuelo, sin colgar.
        if advance is not None:
            advance.join(timeout=60)


if __name__ == "__main__":
    raise SystemExit(main())
