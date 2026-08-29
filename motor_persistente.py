from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = PACKAGE_ROOT / "scripts_originales"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import diarizar  # noqa: E402
import transcribir  # noqa: E402

_stdlib_print = print


def print(*args: Any, **kwargs: Any) -> None:  # noqa: A001 -- sella cada log con la hora, igual que el worker
    stamp = datetime.now().strftime("%H:%M:%S")
    sep = kwargs.get("sep", " ")
    text = sep.join(str(a) for a in args)
    stamped = "\n".join(f"[{stamp}] {line}" for line in text.split("\n"))
    _stdlib_print(stamped, **{k: v for k, v in kwargs.items() if k != "sep"})


class PersistentEngine:
    """Carga Faster-Whisper y Pyannote una sola vez y los reutiliza entre jobs,
    evitando el costo de arranque (carga de modelo) en cada transcripción.

    Solo soporta la combinación por defecto (faster-whisper + pyannote); mlx/soniqo
    siguen corriendo vía subproceso como hasta ahora.

    `load_whisper` / `load_pyannote` permiten cargar solo el modelo necesario
    cuando el worker anuncia una sola capacidad (--transcription-only /
    --diarization-only).
    """

    def __init__(
        self,
        *,
        whisper_model: str,
        device: str,
        compute_type: str,
        models_dir: Path,
        batch_size: int,
        beam_size: int,
        pyannote_model: str,
        segmentation_batch_size: int = 6,
        embedding_batch_size: int = 16,
        allow_cpu_diarization: bool = False,
        load_whisper: bool = True,
        load_pyannote: bool = True,
    ) -> None:
        if not load_whisper and not load_pyannote:
            raise ValueError("PersistentEngine necesita cargar al menos un modelo.")

        self.device = device
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.pyannote_model = pyannote_model
        self.transcriber = None
        self.pipeline = None
        self.diarization_device = None

        if load_whisper:
            print("  [motor persistente] cargando Faster-Whisper...", flush=True)
            self.transcriber = transcribir.load_model(
                whisper_model, device, compute_type, models_dir / "whisper", batch_size
            )

        if load_pyannote:
            print("  [motor persistente] cargando Pyannote...", flush=True)
            self.diarization_device, self.pipeline = diarizar.load_pipeline(
                pyannote_model,
                models_dir / "pyannote-cache",
                device,
                segmentation_batch_size,
                embedding_batch_size,
                allow_cpu=allow_cpu_diarization,
            )
        print("  [motor persistente] modelos listos.", flush=True)

    def process(
        self,
        audio_path: Path,
        out_dir: Path,
        *,
        language: str,
        want_transcription: bool = True,
        want_diarization: bool = True,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict[str, Any]:
        """Corre las etapas pedidas y devuelve los datos ya leídos de disco, en el
        mismo formato que worker_transcripcion.load_results:
        {transcript_data, diarization_data, speakers_data} (None lo no pedido)."""
        if want_transcription and self.transcriber is None:
            raise RuntimeError("Se pidió transcripción pero el motor no cargó Whisper.")
        if want_diarization and self.pipeline is None:
            raise RuntimeError("Se pidió diarización pero el motor no cargó Pyannote.")

        audio_root = audio_path.parent
        transcript_root = out_dir / "transcripciones"
        diarization_root = out_dir / "diarizaciones"
        relative = audio_path.relative_to(audio_root)

        if want_transcription:
            print("  [motor persistente] transcribiendo...", flush=True)
            transcribir.transcribe_one(
                self.transcriber,
                audio_path,
                audio_root,
                transcript_root,
                language=language,
                beam_size=self.beam_size,
                batch_size=self.batch_size,
                word_timestamps=True,
            )

        if want_diarization:
            print("  [motor persistente] diarizando...", flush=True)
            # diarizar.diarize_one alinea con el transcript si lo encuentra en
            # transcript_root; en jobs de solo-diarización no existe y produce
            # únicamente el .diarization.json.
            diarizar.diarize_one(
                self.pipeline,
                self.diarization_device,
                audio_path,
                audio_root,
                transcript_root,
                diarization_root,
                self.pyannote_model,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        def _load(path: Path) -> dict[str, Any] | None:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

        transcript_data = (
            _load((transcript_root / relative).with_suffix(".json")) if want_transcription else None
        )
        diarization_data = (
            _load((diarization_root / relative).with_suffix(".diarization.json"))
            if want_diarization
            else None
        )
        speakers_data = (
            _load((diarization_root / relative).with_suffix(".speakers.json"))
            if want_transcription and want_diarization
            else None
        )

        return {
            "transcript_data": transcript_data,
            "diarization_data": diarization_data,
            "speakers_data": speakers_data,
        }
