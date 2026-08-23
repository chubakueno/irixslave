from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita calidad estructural y confianza de transcripciones.")
    parser.add_argument("--audio-root", type=Path, default=Path("audios"))
    parser.add_argument("--transcript-root", type=Path, default=Path("transcripciones"))
    parser.add_argument("--output", type=Path, default=Path("control_calidad"))
    args = parser.parse_args()

    audio_root = args.audio_root.resolve()
    transcript_root = args.transcript_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_root.rglob("*.mp3"))
    json_files = sorted(transcript_root.rglob("*.json"))
    audio_keys = {p.relative_to(audio_root).with_suffix(""): p for p in audio_files}
    json_keys = {p.relative_to(transcript_root).with_suffix(""): p for p in json_files}

    rows: list[dict] = []
    all_probabilities: list[float] = []
    languages: Counter[str] = Counter()
    parse_errors: list[str] = []

    for key in sorted(audio_keys):
        audio = audio_keys[key]
        json_path = json_keys.get(key)
        txt_path = transcript_root / key.with_suffix(".txt")
        srt_path = transcript_root / key.with_suffix(".srt")
        row = {
            "audio": str(key.with_suffix(".mp3")),
            "json_exists": bool(json_path),
            "txt_exists": txt_path.exists(),
            "srt_exists": srt_path.exists(),
            "json_parse_ok": False,
            "duration_seconds": None,
            "vad_seconds": None,
            "speech_coverage": None,
            "language": None,
            "language_probability": None,
            "segments": 0,
            "words": 0,
            "word_timestamps": False,
            "mean_word_probability": None,
            "median_word_probability": None,
            "low_confidence_words": 0,
            "low_confidence_rate": None,
            "very_low_confidence_words": 0,
            "segment_overlaps": 0,
            "max_segment_overlap_seconds": 0.0,
            "invalid_segment_times": 0,
            "invalid_word_times": 0,
            "flags": [],
        }
        if not json_path:
            row["flags"].append("missing_json")
            rows.append(row)
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            row["flags"].append("json_parse_error")
            parse_errors.append(f"{json_path}: {exc}")
            rows.append(row)
            continue

        row["json_parse_ok"] = True
        duration = float(data.get("duration") or 0)
        vad_seconds = float(data.get("duration_after_vad") or 0)
        row["duration_seconds"] = duration
        row["vad_seconds"] = vad_seconds
        row["speech_coverage"] = vad_seconds / duration if duration else None
        row["language"] = data.get("language")
        row["language_probability"] = data.get("language_probability")
        languages[str(data.get("language"))] += 1

        segments = data.get("segments") or []
        row["segments"] = len(segments)
        words = []
        invalid_segment_times = 0
        segment_overlaps = 0
        max_segment_overlap_seconds = 0.0
        invalid_word_times = 0
        previous_segment_end = 0.0
        has_word_field = False
        for segment in segments:
            start = float(segment.get("start") or 0)
            end = float(segment.get("end") or 0)
            if start < 0 or end < start or end > duration + 1:
                invalid_segment_times += 1
            overlap = previous_segment_end - start
            if overlap > 1e-3:
                segment_overlaps += 1
                max_segment_overlap_seconds = max(max_segment_overlap_seconds, overlap)
            previous_segment_end = max(previous_segment_end, end)
            if "words" in segment:
                has_word_field = True
            for word in segment.get("words") or []:
                words.append(word)
                word_start = float(word.get("start") or 0)
                word_end = float(word.get("end") or 0)
                if word_start < start - 1 or word_end < word_start or word_end > end + 1:
                    invalid_word_times += 1

        probabilities = [
            float(word["probability"])
            for word in words
            if word.get("probability") is not None
        ]
        all_probabilities.extend(probabilities)
        row["words"] = len(words)
        row["word_timestamps"] = has_word_field and bool(words)
        row["segment_overlaps"] = segment_overlaps
        row["max_segment_overlap_seconds"] = max_segment_overlap_seconds
        row["invalid_segment_times"] = invalid_segment_times
        row["invalid_word_times"] = invalid_word_times
        if probabilities:
            row["mean_word_probability"] = statistics.fmean(probabilities)
            row["median_word_probability"] = statistics.median(probabilities)
            row["low_confidence_words"] = sum(p < 0.5 for p in probabilities)
            row["low_confidence_rate"] = row["low_confidence_words"] / len(probabilities)
            row["very_low_confidence_words"] = sum(p < 0.3 for p in probabilities)

        if not txt_path.exists():
            row["flags"].append("missing_txt")
        if not srt_path.exists():
            row["flags"].append("missing_srt")
        if not segments:
            row["flags"].append("empty_transcript")
        elif len(words) and len(words) < 50:
            row["flags"].append("near_empty_transcript")
        if segments and not has_word_field:
            row["flags"].append("missing_word_timestamps")
        if row["low_confidence_rate"] is not None and row["low_confidence_rate"] > 0.05:
            row["flags"].append("high_low_confidence_rate")
        if invalid_segment_times:
            row["flags"].append("invalid_segment_times")
        if invalid_word_times:
            row["flags"].append("invalid_word_times")
        rows.append(row)

    flag_counts = Counter(flag for row in rows for flag in row["flags"])
    scored_rows = [row for row in rows if row["words"]]
    summary = {
        "audio_files": len(audio_files),
        "json_files": len(json_files),
        "txt_files": len(list(transcript_root.rglob("*.txt"))),
        "srt_files": len(list(transcript_root.rglob("*.srt"))),
        "audio_without_json": len(set(audio_keys) - set(json_keys)),
        "json_without_audio": len(set(json_keys) - set(audio_keys)),
        "parse_errors": len(parse_errors),
        "languages": dict(languages),
        "files_with_scored_words": len(scored_rows),
        "total_scored_words": len(all_probabilities),
        "mean_word_probability": statistics.fmean(all_probabilities) if all_probabilities else None,
        "median_word_probability": statistics.median(all_probabilities) if all_probabilities else None,
        "p10_word_probability": percentile(all_probabilities, 0.10),
        "words_below_0_5_rate": sum(p < 0.5 for p in all_probabilities) / len(all_probabilities) if all_probabilities else None,
        "words_below_0_3_rate": sum(p < 0.3 for p in all_probabilities) / len(all_probabilities) if all_probabilities else None,
        "words_at_least_0_8_rate": sum(p >= 0.8 for p in all_probabilities) / len(all_probabilities) if all_probabilities else None,
        "files_with_segment_overlaps": sum(bool(row["segment_overlaps"]) for row in rows),
        "segment_overlap_events": sum(row["segment_overlaps"] for row in rows),
        "max_segment_overlap_seconds": max((row["max_segment_overlap_seconds"] for row in rows), default=0.0),
        "flag_counts": dict(flag_counts),
    }

    csv_path = output / "auditoria_archivos.csv"
    fields = [key for key in rows[0] if key != "flags"] + ["flags"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            export = dict(row)
            export["flags"] = ";".join(row["flags"])
            writer.writerow(export)

    summary_path = output / "resumen_calidad.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    selection_groups = {
        "candidatos_vacios.txt": [
            row for row in rows
            if "empty_transcript" in row["flags"] or "near_empty_transcript" in row["flags"]
        ],
        "candidatos_sin_tiempos_palabra.txt": [
            row for row in rows if "missing_word_timestamps" in row["flags"]
        ],
        "candidatos_baja_confianza.txt": [
            row for row in rows if "high_low_confidence_rate" in row["flags"]
        ],
        "candidatos_reprocesar_beam5.txt": [
            row for row in rows
            if "missing_word_timestamps" in row["flags"]
            or "high_low_confidence_rate" in row["flags"]
        ],
    }
    for filename, selected_rows in selection_groups.items():
        (output / filename).write_text(
            "\n".join(row["audio"] for row in selected_rows) + ("\n" if selected_rows else ""),
            encoding="utf-8-sig",
        )

    flagged = [row for row in rows if row["flags"]]
    flagged.sort(key=lambda row: (
        row["mean_word_probability"] if row["mean_word_probability"] is not None else -1,
        row["audio"],
    ))
    report_lines = [
        "# Auditoría de calidad de transcripciones",
        "",
        f"- Audios: {summary['audio_files']}",
        f"- JSON/TXT/SRT: {summary['json_files']}/{summary['txt_files']}/{summary['srt_files']}",
        f"- Palabras evaluadas: {summary['total_scored_words']}",
        f"- Confianza media: {(summary['mean_word_probability'] or 0):.2%}",
        f"- Palabras con confianza menor a 50%: {(summary['words_below_0_5_rate'] or 0):.2%}",
        f"- Palabras con confianza menor a 30%: {(summary['words_below_0_3_rate'] or 0):.2%}",
        f"- Solapamientos breves entre bloques: {summary['segment_overlap_events']} en {summary['files_with_segment_overlaps']} archivos; máximo {summary['max_segment_overlap_seconds']:.2f} s",
        "- Nota: la confianza del modelo no equivale a una tasa de acierto medida contra una transcripción humana.",
        "",
        "## Archivos señalados",
        "",
        "| Archivo | Segmentos | Palabras | Confianza media | Banderas |",
        "|---|---:|---:|---:|---|",
    ]
    for row in flagged:
        confidence = "—" if row["mean_word_probability"] is None else f"{row['mean_word_probability']:.1%}"
        report_lines.append(
            f"| {row['audio']} | {row['segments']} | {row['words']} | {confidence} | {', '.join(row['flags'])} |"
        )
    report_path = output / "informe_calidad.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"REPORT={report_path}")
    print(f"CSV={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
