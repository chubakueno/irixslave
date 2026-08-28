from __future__ import annotations

"""Prototipo de fingerprinting acustico para detectar jingles/cuñas/ads que se
repiten dentro de una misma radio, ANTES de transcribir (a diferencia del
cruce de texto, que solo confirma la redundancia post-hoc).

Es una implementacion propia en Python del algoritmo de landmarks tipo
Shazam/Panako (pares de picos espectrales -> hash -> votacion por offset
temporal consistente), no un wrapper de Panako (que requiere Java) ni de
chromaprint (pensado para tracks completos, no para encontrar un fragmento
dentro de un audio largo). Es exploratorio: valida si el enfoque funciona
antes de decidir invertir en una libreria madura.

Uso:
    .venv\\Scripts\\python.exe prototipo_fingerprint_audio.py --input muestra_datadaf/Radio_Yaravi
"""

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

TARGET_SR = 8000
WIN_SIZE = 1024
HOP = 512
NEIGHBORHOOD = 20  # tamano de la ventana para buscar maximos locales (en frames/bins)
FAN_OUT = 5  # cuantos peaks siguientes se emparejan con cada peak ancla
MIN_DT_FRAMES = 1
MAX_DT_FRAMES = 200  # ~ 200*HOP/SR = 12.8s de ventana de emparejamiento
MIN_MATCHING_HASHES = 40  # umbral para considerar un match real (no ruido)
CLUSTER_GAP_SECONDS = 5.0  # separacion maxima entre hashes consecutivos para seguir en el mismo cluster
# Un hash que aparece k veces genera k^2 pares -- pero un hash "generico" (ruido
# comun a muchos archivos) y un hash de una promo MUY repetida tienen el mismo k,
# asi que descartar por umbral de ocurrencias (como se hizo antes) tira ambos por
# igual y pierde promos reales muy populares. En vez de eso, se acota cuantas
# comparaciones hace CADA ocurrencia (no se descarta el hash entero): un jingle
# real sigue acumulando votos de sus otros cientos de hashes aunque cada uno
# individualmente compare menos: el ruido, al no concentrarse en ningun offset,
# se sigue diluyendo igual de bien con muestreo que con comparacion exhaustiva.
MAX_PAIRS_PER_OCCURRENCE = 20


def load_audio_mono(path: Path, sr: int) -> np.ndarray:
    container = av.open(str(path), metadata_errors="replace")
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
    chunks = []
    for frame in container.decode(stream):
        for r in resampler.resample(frame):
            chunks.append(r.to_ndarray())
    for r in resampler.resample(None):
        chunks.append(r.to_ndarray())
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32)


def spectrogram(pcm: np.ndarray) -> np.ndarray:
    n_frames = 1 + (len(pcm) - WIN_SIZE) // HOP
    if n_frames <= 0:
        return np.zeros((0, WIN_SIZE // 2 + 1))
    window = np.hanning(WIN_SIZE)
    frames = np.lib.stride_tricks.sliding_window_view(pcm, WIN_SIZE)[::HOP][:n_frames]
    spec = np.abs(np.fft.rfft(frames * window, axis=1))
    with np.errstate(divide="ignore"):
        return 20 * np.log10(spec + 1e-6)


def find_peaks(spec: np.ndarray) -> list[tuple[int, int]]:
    if spec.shape[0] == 0:
        return []
    from scipy.ndimage import maximum_filter

    local_max = maximum_filter(spec, size=(NEIGHBORHOOD, NEIGHBORHOOD // 2)) == spec
    threshold = np.percentile(spec, 75)
    mask = local_max & (spec > threshold)
    times, freqs = np.nonzero(mask)
    return list(zip(times.tolist(), freqs.tolist()))


def landmarks_to_hashes(peaks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Devuelve lista de (hash, tiempo_ancla_en_frames)."""
    peaks_sorted = sorted(peaks)
    hashes = []
    n = len(peaks_sorted)
    for i in range(n):
        t1, f1 = peaks_sorted[i]
        count = 0
        for j in range(i + 1, n):
            t2, f2 = peaks_sorted[j]
            dt = t2 - t1
            if dt < MIN_DT_FRAMES:
                continue
            if dt > MAX_DT_FRAMES:
                break
            h = (f1 << 20) | (f2 << 10) | dt
            hashes.append((h, t1))
            count += 1
            if count >= FAN_OUT:
                break
    return hashes


@dataclass
class Match:
    file_a: str
    file_b: str
    offset_frames: int
    n_hashes: int
    a_start_s: float
    a_end_s: float
    b_start_s: float
    b_end_s: float


def frames_to_seconds(frames: int) -> float:
    return frames * HOP / TARGET_SR


def cluster_pairs(pairs: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Agrupa (ta, tb) que comparten el mismo offset por proximidad temporal.

    offset_votes ya garantiza que todos los pares de un bucket comparten
    ta-tb constante, pero eso no implica que sean un tramo continuo: dos
    coincidencias reales y separadas (o una real + un hash espurio suelto)
    pueden caer en el mismo offset por casualidad. Sin este paso, un solo
    outlier lejano estira el rango reportado (min..max) hasta hacerlo
    parecer un bloque continuo que no existe -- fue exactamente el bug que
    encontramos escuchando el "match grande" de 6.5 min que en realidad
    eran 2198 hashes en 52s + 1 hash espurio a 6 minutos de distancia.
    """
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    gap_frames = int(CLUSTER_GAP_SECONDS * TARGET_SR / HOP)
    clusters: list[list[tuple[int, int]]] = [[pairs_sorted[0]]]
    for p in pairs_sorted[1:]:
        if p[0] - clusters[-1][-1][0] > gap_frames:
            clusters.append([])
        clusters[-1].append(p)
    return clusters


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototipo de deteccion de jingles/ads repetidos via fingerprinting.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--min-matches", type=int, default=MIN_MATCHING_HASHES)
    args = parser.parse_args()

    audio_files = sorted(args.input.glob("*.mp3"))
    if not audio_files:
        print(f"No hay mp3 en {args.input}")
        return 2

    # hash -> list[(file_id, tiempo_ancla_frames)]
    index: dict[int, list[tuple[str, int]]] = defaultdict(list)
    durations: dict[str, float] = {}

    for path in audio_files:
        file_id = path.stem
        print(f"Procesando {file_id}...", flush=True)
        pcm = load_audio_mono(path, TARGET_SR)
        durations[file_id] = len(pcm) / TARGET_SR
        spec = spectrogram(pcm)
        peaks = find_peaks(spec)
        hashes = landmarks_to_hashes(peaks)
        print(f"  {len(peaks)} peaks, {len(hashes)} hashes")
        for h, t in hashes:
            index[h].append((file_id, t))

    print(f"\n{len(index)} hashes distintos en el indice. Buscando coincidencias con offset consistente...")

    # (file_a, file_b, offset_cuantizado) -> lista de (ta, tb)
    offset_votes: dict[tuple[str, str, int], list[tuple[int, int]]] = defaultdict(list)
    muestreados = 0
    for h, occurrences in index.items():
        n = len(occurrences)
        if n < 2:
            continue
        if n - 1 <= MAX_PAIRS_PER_OCCURRENCE:
            pairs_idx = [(i, j) for i in range(n) for j in range(n) if i != j]
        else:
            muestreados += 1
            rng = random.Random(h)  # semilla fija por hash: mismo resultado en cada corrida
            pairs_idx = []
            for i in range(n):
                for j in rng.sample(range(n - 1), MAX_PAIRS_PER_OCCURRENCE):
                    pairs_idx.append((i, j if j < i else j + 1))  # evita j==i sin sesgar la muestra
        for i, j in pairs_idx:
            fa, ta = occurrences[i]
            fb, tb = occurrences[j]
            if fa >= fb:
                continue  # evita duplicar y auto-comparar mismo archivo
            offset = ta - tb
            offset_votes[(fa, fb, offset)].append((ta, tb))

    print(f"{muestreados} hashes con mas de {MAX_PAIRS_PER_OCCURRENCE + 1} ocurrencias: se muestrearon sus comparaciones en vez de compararlas todas.")

    matches: list[Match] = []
    for (fa, fb, offset), pairs in offset_votes.items():
        if len(pairs) < args.min_matches:
            continue  # filtro barato: ningun cluster de este bucket puede superar el total
        for cluster in cluster_pairs(pairs):
            if len(cluster) < args.min_matches:
                continue
            tas = [p[0] for p in cluster]
            tbs = [p[1] for p in cluster]
            matches.append(
                Match(
                    file_a=fa,
                    file_b=fb,
                    offset_frames=offset,
                    n_hashes=len(cluster),
                    a_start_s=frames_to_seconds(min(tas)),
                    a_end_s=frames_to_seconds(max(tas)),
                    b_start_s=frames_to_seconds(min(tbs)),
                    b_end_s=frames_to_seconds(max(tbs)),
                )
            )

    matches.sort(key=lambda m: -m.n_hashes)
    print(f"\n{len(matches)} clusters con >= {args.min_matches} hashes (offset consistente + continuidad temporal):\n")
    for m in matches:
        print(
            f"  {m.file_a} [{m.a_start_s:6.1f}s-{m.a_end_s:6.1f}s]  <->  "
            f"{m.file_b} [{m.b_start_s:6.1f}s-{m.b_end_s:6.1f}s]  ({m.n_hashes} hashes)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
