from __future__ import annotations

import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts_originales"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audio_compat import decode_audio, install_faster_whisper_audio_compat


class AudioCompatTests(unittest.TestCase):
    def test_decode_wave_to_mono_16khz(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "tone.wav"
            sample_rate = 8_000
            duration = 0.1
            frames = [
                int(8_000 * math.sin(2 * math.pi * 440 * index / sample_rate))
                for index in range(round(sample_rate * duration))
            ]
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(b"".join(struct.pack("<h", frame) for frame in frames))

            decoded = decode_audio(source, sampling_rate=16_000)
            self.assertEqual(decoded.dtype.name, "float32")
            self.assertGreaterEqual(decoded.shape[0], 1_590)
            self.assertLessEqual(decoded.shape[0], 1_610)

    def test_faster_whisper_uses_compat_decoder(self) -> None:
        install_faster_whisper_audio_compat()
        import faster_whisper.transcribe as faster_transcribe

        self.assertEqual(faster_transcribe.decode_audio.__module__, "audio_compat")


if __name__ == "__main__":
    unittest.main()
