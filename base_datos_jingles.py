from __future__ import annotations

"""Base de datos de jingles/cuñas: en vez de comparar cada capture contra
todos los demás (O(n^2), lo que hace prototipo_fingerprint_audio.py), se
mantiene un indice invertido de clips ya conocidos (hash -> [(jingle_id,
offset_dentro_del_clip)]) y cada capture nuevo se compara UNA vez contra
ese indice (O(hashes_del_capture)).

Dos modos:
  --construir : arma la base de datos a partir de un archivo de clusters ya
                calculado por prototipo_fingerprint_audio.py (agrupa
                ocurrencias del mismo clip via union-find, elige una
                referencia canonica por clip, extrae su fingerprint "limpio").
  --buscar    : dado un audio nuevo, lo compara contra la base de datos y
                reporta que jingles conocidos aparecen y en que tiempos.

Uso:
    .venv\\Scripts\\python.exe base_datos_jingles.py --construir \\
        --clusters resultados_clasificador/fingerprint_yaravi_v2.txt \\
        --audios muestra_datadaf/Radio_Yaravi \\
        --min-dur 10 \\
        --db resultados_clasificador/jingles_yaravi.pkl

    .venv\\Scripts\\python.exe base_datos_jingles.py --buscar \\
        --db resultados_clasificador/jingles_yaravi.pkl \\
        --audios muestra_datadaf/Radio_Yaravi
"""

import argparse
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import prototipo_fingerprint_audio as fp

CLUSTER_LINE = re.compile(
    r"^\s*(\S+)\s+\[\s*([\d.]+)s-\s*([\d.]+)s\]\s+<->\s+(\S+)\s+\[\s*([\d.]+)s-\s*([\d.]+)s\]\s+\((\d+) hashes\)"
)


@dataclass
class Jingle:
    jingle_id: int
    file_id: str
    start_s: float
    end_s: float
    n_occurrences: int
    hashes: list[int] = field(default_factory=list)


@dataclass
class JingleDB:
    jingles: dict[int, Jingle]
    index: dict[int, list[tuple[int, int]]]  # hash -> [(jingle_id, offset_frames_dentro_del_clip)]


def parse_clusters(path: Path) -> list[tuple[str, float, float, str, float, float, int]]:
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = CLUSTER_LINE.match(line)
        if m:
            fa, a0, a1, fb, b0, b1, n = m.groups()
            matches.append((fa, float(a0), float(a1), fb, float(b0), float(b1), int(n)))
    return matches


def group_occurrences(matches: list) -> list[list[tuple[str, float, float]]]:
    occ_id: dict[tuple, int] = {}

    def get_occ(file: str, s: float, e: float) -> int:
        key = (file, round(s, 1), round(e, 1))
        if key not in occ_id:
            occ_id[key] = len(occ_id)
        return occ_id[key]

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    occ_info: dict[int, tuple[str, float, float]] = {}
    for fa, a0, a1, fb, b0, b1, _n in matches:
        ia, ib = get_occ(fa, a0, a1), get_occ(fb, b0, b1)
        occ_info[ia] = (fa, a0, a1)
        occ_info[ib] = (fb, b0, b1)
        parent.setdefault(ia, ia)
        parent.setdefault(ib, ib)
        union(ia, ib)

    groups: dict[int, list[int]] = defaultdict(list)
    for oid in occ_info:
        groups[find(oid)].append(oid)

    return [[occ_info[o] for o in oids] for oids in groups.values()]


def build_db(clusters_path: Path, audios_dir: Path, min_dur: float) -> JingleDB:
    matches = parse_clusters(clusters_path)
    groups = group_occurrences(matches)
    print(f"{len(groups)} clips distintos en {clusters_path.name}")

    jingles: dict[int, Jingle] = {}
    index: dict[int, list[tuple[int, int]]] = defaultdict(list)
    pcm_cache: dict[str, "object"] = {}

    jid = 0
    for occs in groups:
        durs = sorted(e - s for _, s, e in occs)
        typical = durs[len(durs) // 2]
        if typical < min_dur:
            continue
        # referencia canonica: la ocurrencia mas larga (fingerprint mas completo)
        file_id, s, e = max(occs, key=lambda o: o[2] - o[1])
        if file_id not in pcm_cache:
            pcm_cache[file_id] = fp.load_audio_mono(audios_dir / f"{file_id}.mp3", fp.TARGET_SR)
        pcm = pcm_cache[file_id]
        s0 = max(0, int(s * fp.TARGET_SR))
        s1 = min(len(pcm), int(e * fp.TARGET_SR))
        clip_pcm = pcm[s0:s1]
        spec = fp.spectrogram(clip_pcm)
        peaks = fp.find_peaks(spec)
        hashes = fp.landmarks_to_hashes(peaks)  # (hash, offset_frames_dentro_del_clip)

        jingle = Jingle(jid, file_id, s, e, len(occs), [h for h, _ in hashes])
        jingles[jid] = jingle
        for h, offset in hashes:
            index[h].append((jid, offset))
        jid += 1

    print(f"{len(jingles)} jingles indexados (min_dur={min_dur}s), {len(index)} hashes distintos en el indice")
    return JingleDB(jingles=jingles, index=index)


def search(db: JingleDB, audio_path: Path, min_matches: int = 30) -> list[tuple[int, float, float, int]]:
    """Devuelve [(jingle_id, offset_en_el_capture_s, duracion_s, n_hashes)]."""
    pcm = fp.load_audio_mono(audio_path, fp.TARGET_SR)
    spec = fp.spectrogram(pcm)
    peaks = fp.find_peaks(spec)
    hashes = fp.landmarks_to_hashes(peaks)

    # (jingle_id, alineacion_frames) -> lista de tiempos de ancla en el capture
    votes: dict[tuple[int, int], list[int]] = defaultdict(list)
    for h, t_capture in hashes:
        for jid, t_jingle in db.index.get(h, []):
            alignment = t_capture - t_jingle
            votes[(jid, alignment)].append(t_capture)

    results = []
    for (jid, alignment), times in votes.items():
        if len(times) < min_matches:
            continue
        times_sorted = sorted(times)
        gap_frames = int(fp.CLUSTER_GAP_SECONDS * fp.TARGET_SR / fp.HOP)
        clusters = [[times_sorted[0]]]
        for t in times_sorted[1:]:
            if t - clusters[-1][-1] > gap_frames:
                clusters.append([])
            clusters[-1].append(t)
        for cluster in clusters:
            if len(cluster) < min_matches:
                continue
            start_s = fp.frames_to_seconds(min(cluster))
            end_s = fp.frames_to_seconds(max(cluster))
            results.append((jid, start_s, end_s - start_s, len(cluster)))

    results.sort(key=lambda r: r[1])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Base de datos de jingles: construir e indexar, o buscar contra ella.")
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--construir", action="store_true")
    sub.add_argument("--buscar", action="store_true")
    parser.add_argument("--clusters", type=Path, help="(--construir) archivo de salida de prototipo_fingerprint_audio.py")
    parser.add_argument("--audios", type=Path, required=True, help="Carpeta con los .mp3")
    parser.add_argument("--min-dur", type=float, default=10.0, help="(--construir) duracion minima para indexar un clip")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--min-matches", type=int, default=30, help="(--buscar) hashes minimos para reportar un match")
    args = parser.parse_args()

    if args.construir:
        db = build_db(args.clusters, args.audios, args.min_dur)
        args.db.parent.mkdir(parents=True, exist_ok=True)
        with args.db.open("wb") as fh:
            pickle.dump(db, fh)
        print(f"Base de datos guardada en {args.db}")
        return 0

    with args.db.open("rb") as fh:
        db: JingleDB = pickle.load(fh)
    print(f"Base de datos cargada: {len(db.jingles)} jingles, {len(db.index)} hashes")

    def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        merged: list[tuple[float, float]] = []
        for s, e in sorted(intervals):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    total_capture_s = 0.0
    total_matched_s = 0.0
    for audio_path in sorted(args.audios.glob("*.mp3")):
        dur = fp.load_audio_mono(audio_path, fp.TARGET_SR).shape[0] / fp.TARGET_SR
        total_capture_s += dur
        results = search(db, audio_path, args.min_matches)
        # varios jingles de la DB pueden ser fragmentos del mismo clip real y
        # matchear el mismo tramo del capture -- se deduplica por union de
        # intervalos antes de sumar cobertura (si no, un tramo que matchea
        # contra 14 jingles "distintos" se cuenta 14 veces).
        merged = merge_intervals([(s, s + d) for _jid, s, d, _n in results])
        matched_s = sum(e - s for s, e in merged)
        total_matched_s += matched_s
        print(
            f"\n{audio_path.stem} ({dur:.0f}s): {len(results)} matches crudos, "
            f"{len(merged)} tramos distintos, {matched_s:.1f}s cubiertos (dedup)"
        )
        for jid, start_s, dur_s, n in results:
            j = db.jingles[jid]
            print(f"    jingle#{jid} (ref: {j.file_id} @ {j.start_s:.1f}s, {j.n_occurrences} apariciones conocidas) "
                  f"en [{start_s:.1f}s-{start_s+dur_s:.1f}s] ({n} hashes)")

    print(f"\nTotal: {total_matched_s:.1f}s reconocidos de {total_capture_s:.1f}s ({100*total_matched_s/total_capture_s:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
