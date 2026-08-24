from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = PACKAGE_ROOT / "scripts_originales"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import diarizar  # noqa: E402
import transcribir  # noqa: E402


class PersistentEngine:
    """Carga Faster-Whisper y Pyannote una sola vez y los reutiliza entre jobs,
    evitando el costo de arranque (carga de modelo) en cada transcripción.

    Solo soporta la combinación por defecto (faster-whisper + pyannote); mlx/soniqo
    siguen corriendo vía subproceso como hasta ahora.
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
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.pyannote_model = pyannote_model

        print("  [motor persistente] cargando Faster-Whisper...", flush=True)
        self.transcriber = transcribir.load_model(
            whisper_model, device, compute_type, models_dir / "whisper", batch_size
        )

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
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        audio_root = audio_path.parent
        transcript_root = out_dir / "transcripciones"
        diarization_root = out_dir / "diarizaciones"

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

        relative = audio_path.relative_to(audio_root)
        transcript_path = (transcript_root / relative).with_suffix(".json")
        speakers_path = (diarization_root / relative).with_suffix(".speakers.json")
        transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
        speakers_data = json.loads(speakers_path.read_text(encoding="utf-8"))
        return speakers_data, transcript_data
