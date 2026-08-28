from __future__ import annotations

"""Version incremental del fingerprinting: procesa los audios uno por uno,
buscando cada uno contra un indice que va creciendo, en vez de comparar
todos-contra-todos de una (prototipo_fingerprint_audio.py). Evita el blowup
de memoria del enfoque exhaustivo porque nunca mantiene en RAM la tabla de
votos completa de N archivos a la vez -- solo el indice acumulado (acotado
por archivo-hash) y los votos de UNA busqueda puntual, que se descartan
apenas se procesan.

Uso:
    .venv\\Scripts\\python.exe fingerprint_incremental.py --input muestra_datadaf/Radio_Yaravi
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import prototipo_fingerprint_audio as fp

MAX_INDEX_OCCURRENCES = 100  # tope por hash en el indice permanente: una vez alcanzado,
                              # no se agregan mas ocurrencias de ese hash (deja de crecer,
                              # pero lo ya indexado se sigue pudiendo usar para buscar)


@dataclass
class Match:
    file_a: str
    file_b: str
    a_start_s: float
    a_end_s: float
    b_start_s: float
    b_end_s: float
    n_hashes: int


def buscar_y_agregar(
    file_id: str,
    hashes: list[tuple[int, int]],
    index: dict[int, list[tuple[str, int]]],
) -> list[Match]:
    """Busca los hashes de este archivo contra el indice acumulado HASTA AHORA,
    reporta matches, y despues agrega este archivo al indice para el futuro."""

    # (archivo_viejo, offset) -> lista de (t_nuevo, t_viejo) -- solo para ESTE archivo
    votes: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for h, t_new in hashes:
        for other_file, t_old in index.get(h, []):
            offset = t_new - t_old
            votes[(other_file, offset)].append((t_new, t_old))

    matches: list[Match] = []
    for (other_file, offset), pairs in votes.items():
        if len(pairs) < fp.MIN_MATCHING_HASHES:
            continue
        for cluster in fp.cluster_pairs(pairs):
            if len(cluster) < fp.MIN_MATCHING_HASHES:
                continue
            t_news = [p[0] for p in cluster]
            t_olds = [p[1] for p in cluster]
            matches.append(
                Match(
                    file_a=file_id,
                    file_b=other_file,
                    a_start_s=fp.frames_to_seconds(min(t_news)),
                    a_end_s=fp.frames_to_seconds(max(t_news)),
                    b_start_s=fp.frames_to_seconds(min(t_olds)),
                    b_end_s=fp.frames_to_seconds(max(t_olds)),
                    n_hashes=len(cluster),
                )
            )
    # votes se descarta aca (sale de scope al terminar la funcion) -- nunca se
    # acumula la tabla de votos de mas de un archivo a la vez en memoria.

    # ahora sí, agregar este archivo al indice permanente para los que vengan despues
    for h, t in hashes:
        bucket = index[h]
        if len(bucket) < MAX_INDEX_OCCURRENCES:
            bucket.append((file_id, t))

    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Fingerprinting incremental: un archivo a la vez contra un indice creciente.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--order", choices=("chronological", "filename"), default="chronological")
    args = parser.parse_args()

    audio_files = sorted(args.input.glob("*.mp3"))
    if not audio_files:
        print(f"No hay mp3 en {args.input}")
        return 2

    if args.order == "chronological":
        manifest_path = args.input / "manifest.json"
        if manifest_path.exists():
            started_at = {m["id"]: m["started_at"] for m in json.loads(manifest_path.read_text(encoding="utf-8"))}
            audio_files = sorted(audio_files, key=lambda p: started_at.get(p.stem, ""))
        else:
            print("(no hay manifest.json, se usa orden por nombre de archivo)")

    index: dict[int, list[tuple[str, int]]] = defaultdict(list)
    all_matches: list[Match] = []

    for n, path in enumerate(audio_files, start=1):
        file_id = path.stem
        pcm = fp.load_audio_mono(path, fp.TARGET_SR)
        spec = fp.spectrogram(pcm)
        peaks = fp.find_peaks(spec)
        hashes = fp.landmarks_to_hashes(peaks)

        matches = buscar_y_agregar(file_id, hashes, index)
        all_matches.extend(matches)
        print(
            f"[{n}/{len(audio_files)}] {file_id}: {len(hashes)} hashes, "
            f"{len(matches)} matches contra el indice acumulado ({len(index)} hashes indexados)",
            flush=True,
        )
        for m in matches:
            print(f"    <-> {m.file_b} [{m.a_start_s:.1f}s-{m.a_end_s:.1f}s] = [{m.b_start_s:.1f}s-{m.b_end_s:.1f}s] ({m.n_hashes} hashes)")

    print(f"\nTotal: {len(all_matches)} matches encontrados sobre {len(audio_files)} archivos.")
    out_path = Path("resultados_clasificador/fingerprint_incremental.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([m.__dict__ for m in all_matches], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Guardado en {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
