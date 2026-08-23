from __future__ import annotations

import argparse
import json
from pathlib import Path


def hhmmss(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera TXT y SRT con nombres a partir de speakers.json y speaker-map.json."
    )
    parser.add_argument("speaker_map", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    map_path = args.speaker_map.resolve()
    suffix = ".speaker-map.json"
    if not map_path.name.endswith(suffix):
        raise ValueError("El archivo debe terminar en .speaker-map.json")

    base = map_path.with_name(map_path.name[: -len(suffix)])
    speakers_path = base.with_suffix(".speakers.json")
    named_txt = base.with_suffix(".named.txt")
    named_srt = base.with_suffix(".named.srt")

    speaker_map = json.loads(map_path.read_text(encoding="utf-8-sig"))
    speakers = json.loads(speakers_path.read_text(encoding="utf-8-sig"))
    names = {
        item["speaker"]: item.get("canonical_name") or item.get("display_name") or item["speaker"]
        for item in speaker_map["assignments"]
    }
    corrections = speaker_map.get("quality_notes", [])

    txt_lines: list[str] = []
    srt_blocks: list[str] = []
    for index, turn in enumerate(speakers["turns"], 1):
        speaker = turn["speaker"]
        for correction in corrections:
            if (
                correction.get("observed") == speaker
                and abs(float(correction["start"]) - float(turn["start"])) < 0.02
                and abs(float(correction["end"]) - float(turn["end"])) < 0.02
            ):
                speaker = correction["expected"]
                break
        label = names.get(speaker, speaker)
        text = str(turn["text"]).strip()
        txt_lines.append(
            f"[{hhmmss(float(turn['start']))} - {hhmmss(float(turn['end']))}] {label}: {text}"
        )
        srt_blocks.append(
            "\n".join(
                (
                    str(index),
                    f"{srt_time(float(turn['start']))} --> {srt_time(float(turn['end']))}",
                    f"{label}: {text}",
                )
            )
        )

    named_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    named_srt.write_text("\n\n".join(srt_blocks) + "\n", encoding="utf-8")
    print(f"NAMED_TXT={named_txt}")
    print(f"NAMED_SRT={named_srt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
