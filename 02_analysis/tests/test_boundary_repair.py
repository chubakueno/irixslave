from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts_originales"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from boundary_repair import (  # noqa: E402
    RecoveredSpan,
    merge_recovered_spans,
    packed_block_boundaries,
    plan_adaptive_windows,
    plan_boundary_windows,
    recoverable_spans,
    suspicious_boundaries,
)
from faster_whisper.transcribe import Segment, Word  # noqa: E402


def word(text: str, start: float, end: float, probability: float = 0.9) -> Word:
    return Word(start=start, end=end, word=text, probability=probability)


def segment(segment_id: int, words: list[Word]) -> Segment:
    return Segment(
        id=segment_id,
        seek=0,
        start=words[0].start,
        end=words[-1].end,
        text="".join(item.word for item in words).strip(),
        tokens=[],
        avg_logprob=-0.1,
        compression_ratio=1.0,
        no_speech_prob=0.0,
        words=words,
        temperature=0.0,
    )


class BoundaryRepairTests(unittest.TestCase):
    def test_packed_boundaries_exclude_final_block(self) -> None:
        chunks = [
            {"start": 0, "end": 16_000 * 20},
            {"start": 16_000 * 21, "end": 16_000 * 31},
            {"start": 16_000 * 32, "end": 16_000 * 37},
        ]
        self.assertEqual(packed_block_boundaries(chunks), [31.0])

    def test_packed_boundaries_can_include_final_speech_tail(self) -> None:
        chunks = [
            {"start": 0, "end": 16_000 * 20},
            {"start": 16_000 * 21, "end": 16_000 * 31},
            {"start": 16_000 * 32, "end": 16_000 * 37},
        ]
        self.assertEqual(
            packed_block_boundaries(chunks, include_final=True), [31.0, 37.0]
        )

    def test_windows_are_clamped_to_audio(self) -> None:
        windows = plan_boundary_windows([5.0, 98.0], 100.0)
        self.assertEqual((windows[0].start, windows[0].end), (0.0, 11.0))
        self.assertEqual((windows[1].start, windows[1].end), (86.0, 100.0))

    def test_adaptive_window_brackets_visible_gap_anchors(self) -> None:
        existing = [
            word(" proyecto", 32.1, 32.9),
            word(" a", 37.1, 37.2),
        ]
        window = plan_adaptive_windows([36.6], 100.0, existing)[0]
        self.assertEqual((window.start, window.end), (30.6, 38.7))

    def test_adaptive_window_keeps_long_lookback_for_hidden_seam(self) -> None:
        existing = [word(" cruza", 59.8, 60.2)]
        window = plan_adaptive_windows([60.0], 100.0, existing)[0]
        self.assertEqual((window.start, window.end), (48.0, 61.7))

    def test_only_selects_boundary_with_observable_gap_and_right_anchor(self) -> None:
        existing = [
            word(" sano", 8.0, 9.7),
            word(" sigue", 10.1, 10.4),
            word(" proyecto", 32.1, 32.9),
            word(" emitir", 37.1, 37.6),
            word(" final", 59.7, 60.0),
            word(" continua", 60.2, 60.7),
        ]
        self.assertEqual(
            suspicious_boundaries([10.0, 36.6, 60.0], existing), [36.6, 60.0]
        )

    def test_selects_terminal_gap_without_right_anchor(self) -> None:
        existing = [word(" antes", 88.0, 94.0)]
        self.assertEqual(suspicious_boundaries([100.0], existing), [100.0])

    def test_recovers_missing_phrase_between_two_anchors(self) -> None:
        existing = [
            word(" proyecto", 32.10, 32.90),
            word(" a", 37.07, 37.18),
            word(" emitir", 37.18, 37.60),
        ]
        recovered = [
            word(" proyecto", 32.12, 32.88),
            word(" Nombre", 34.24, 34.70),
            word(" Apellido", 34.70, 35.25),
            word(" ha", 35.25, 35.48),
            word(" salido", 35.48, 36.32),
            word(" a", 37.08, 37.18),
            word(" emitir", 37.18, 37.59),
        ]

        spans = recoverable_spans(existing, recovered, boundary=36.60)

        self.assertEqual(len(spans), 1)
        self.assertEqual(
            "".join(item.word for item in spans[0].words).strip(),
            "Nombre Apellido ha salido",
        )

    def test_does_not_duplicate_words_already_present(self) -> None:
        existing = [word(" uno", 1.0, 1.2), word(" dos", 1.3, 1.5)]
        recovered = [word(" uno", 1.01, 1.19), word(" dos", 1.31, 1.49)]
        self.assertEqual(recoverable_spans(existing, recovered, boundary=1.5), [])

    def test_requires_anchors_on_both_sides(self) -> None:
        existing = [word(" antes", 8.0, 8.4)]
        recovered = [
            word(" antes", 8.0, 8.4),
            word(" posible", 9.0, 9.4),
            word(" omision", 9.4, 10.0),
        ]
        self.assertEqual(recoverable_spans(existing, recovered, boundary=10.0), [])

    def test_allows_strict_terminal_edge_recovery_when_explicit(self) -> None:
        existing = [word(" antes", 8.0, 8.4)]
        recovered = [
            word(" antes", 8.0, 8.4),
            word(" esfuerzo", 9.0, 9.4),
            word(" merece", 9.4, 10.0),
        ]
        spans = recoverable_spans(
            existing,
            recovered,
            boundary=10.0,
            min_missing_words=2,
            min_average_probability=0.7,
            allow_right_edge=True,
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].right_anchor, "__audio_end__")

    def test_allows_strict_left_edge_recovery_when_explicit(self) -> None:
        existing = [word(" despues", 10.5, 11.0)]
        recovered = [
            word(" frase", 8.5, 9.0),
            word(" larga", 9.0, 9.5),
            word(" omitida", 9.5, 10.0),
            word(" despues", 10.5, 11.0),
        ]
        spans = recoverable_spans(
            existing,
            recovered,
            boundary=10.0,
            min_average_probability=0.7,
            allow_left_edge=True,
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].left_anchor, "__window_start__")

    def test_merge_preserves_order_and_assigns_ids(self) -> None:
        before = segment(9, [word(" antes", 1.0, 1.5)])
        after = segment(10, [word(" despues", 4.0, 4.6)])
        span = RecoveredSpan(
            boundary=3.5,
            words=(word(" texto", 2.0, 2.4), word(" faltante", 2.4, 3.2)),
            left_anchor="antes",
            right_anchor="despues",
            average_probability=0.9,
        )

        merged = merge_recovered_spans([after, before], [span])

        self.assertEqual([item.id for item in merged], [1, 2, 3])
        self.assertEqual([item.text for item in merged], ["antes", "texto faltante", "despues"])


if __name__ == "__main__":
    unittest.main()
