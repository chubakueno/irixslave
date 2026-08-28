from __future__ import annotations

"""Harness de validacion: corre un clasificador de audio (AST) en ventanas
deslizantes sobre audios reales y saca scores por ventana para Speech/Music/
Singing. No decide nada por si solo -- el objetivo es generar un CSV que se
pueda escuchar y contrastar a mano antes de fijar umbrales de descarte.

Uso:
    .venv\\Scripts\\python.exe evaluar_clasificador_audio.py --input .worker_tmp --output resultados_clasificador
"""

import argparse
import csv
from pathlib import Path

import av
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
TARGET_SR = 16000
WINDOW_SECONDS = 10.0
HOP_SECONDS = 5.0
LABELS_OF_INTEREST = ("Speech", "Music", "Singing")


def load_audio_mono16k(path: Path) -> np.ndarray:
    container = av.open(str(path), metadata_errors="replace")
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
    chunks = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray())
    for resampled in resampler.resample(None):
        chunks.append(resampled.to_ndarray())
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32) / 32768.0
    return pcm


def find_label_indices(id2label: dict[int, str]) -> dict[str, int]:
    lower_map = {v.lower(): k for k, v in id2label.items()}
    found = {}
    for label in LABELS_OF_INTEREST:
        idx = lower_map.get(label.lower())
        if idx is None:
            raise RuntimeError(f"No se encontro la clase '{label}' en el modelo (revisar id2label).")
        found[label] = idx
    return found


def classify_file(
    path: Path,
    audio_id: str,
    feature_extractor: ASTFeatureExtractor,
    model: ASTForAudioClassification,
    label_idx: dict[str, int],
    device: str,
) -> list[dict]:
    pcm = load_audio_mono16k(path)
    total_seconds = len(pcm) / TARGET_SR
    window_samples = int(WINDOW_SECONDS * TARGET_SR)
    hop_samples = int(HOP_SECONDS * TARGET_SR)

    rows = []
    start_sample = 0
    while start_sample < len(pcm):
        end_sample = start_sample + window_samples
        chunk = pcm[start_sample:end_sample]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        inputs = feature_extractor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()

        window_start = start_sample / TARGET_SR
        window_end = min(end_sample, len(pcm)) / TARGET_SR
        rows.append(
            {
                "audio": audio_id,
                "start": round(window_start, 2),
                "end": round(window_end, 2),
                "p_speech": round(float(probs[label_idx["Speech"]]), 4),
                "p_music": round(float(probs[label_idx["Music"]]), 4),
                "p_singing": round(float(probs[label_idx["Singing"]]), 4),
            }
        )
        if end_sample >= len(pcm):
            break
        start_sample += hop_samples

    print(f"  {path}: {total_seconds:.1f}s de audio, {len(rows)} ventanas")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Corre AST sobre audios reales en ventanas deslizantes.")
    parser.add_argument("--input", type=Path, default=Path(".worker_tmp"), help="Carpeta raiz con audios.")
    parser.add_argument("--glob", default="*.mp3", help="Patron (recursivo) para encontrar audios dentro de --input.")
    parser.add_argument("--output", type=Path, default=Path("resultados_clasificador"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    audio_files = sorted(args.input.rglob(args.glob))
    if not audio_files:
        print(f"No se encontraron audios bajo {args.input} (patron: {args.glob})")
        return 2

    print(f"Cargando {MODEL_ID}...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(MODEL_ID).to(args.device).eval()
    label_idx = find_label_indices(model.config.id2label)
    print(f"Indices de clase: {label_idx}")

    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    print(f"Procesando {len(audio_files)} audios...")
    for audio_path in audio_files:
        try:
            relative = audio_path.relative_to(args.input)
        except ValueError:
            relative = audio_path
        # Layout del worker (.worker_tmp/<job>/in/audio.mp3): usa el nombre del job.
        # Layout de muestra_datadaf (<radio>/<capture_id>.mp3): usa <radio>/<capture_id>.
        audio_id = relative.parent.parent.name if audio_path.name == "audio.mp3" else str(relative.with_suffix(""))
        all_rows.extend(classify_file(audio_path, audio_id, feature_extractor, model, label_idx, args.device))

    out_csv = args.output / "ast_scores.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["audio", "start", "end", "p_speech", "p_music", "p_singing"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nListo: {out_csv} ({len(all_rows)} filas)")

    # Resumen orientativo (NO es la regla final, solo para elegir que tramos escuchar primero)
    candidatos = [r for r in all_rows if r["p_speech"] < 0.20 and (r["p_music"] > 0.85 or r["p_singing"] > 0.70)]
    print(f"\nVentanas candidatas a DROP bajo el umbral provisorio (p_speech<0.20 y p_music>0.85 o p_singing>0.70): {len(candidatos)}")
    for r in candidatos[:30]:
        print(f"  {r['audio']}  [{r['start']:>7.1f}s - {r['end']:>7.1f}s]  speech={r['p_speech']:.2f} music={r['p_music']:.2f} singing={r['p_singing']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
