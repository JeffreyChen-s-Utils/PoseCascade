"""Synthesise tiny WAV byte buffers for the audio tests.

Pure ``stdlib wave`` + numpy — no audio device, no Qt, no soundfile
binding. Tests build clips in memory and skip disk I/O entirely except
when they explicitly need a file path.
"""
from __future__ import annotations

import struct
import wave
from io import BytesIO

import numpy as np


def build_sine_wav_bytes(
    *,
    duration_seconds: float = 1.0,
    frequency_hz: float = 440.0,
    sample_rate: int = 44100,
    channels: int = 1,
    sample_width: int = 2,
    amplitude: float = 0.5,
) -> bytes:
    """Synthesise an ``int16`` sine wave WAV.

    Defaults to a 1-second 440 Hz mono A4. Override for tests that need
    stereo, longer durations, or a different sample width (1, 2, or 4
    bytes — matches what :mod:`posecascade.audio.clip` understands).
    """
    sample_count = int(round(duration_seconds * sample_rate))
    t = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * frequency_hz * t) * amplitude
    samples = np.tile(mono.reshape(-1, 1), (1, channels))
    return _encode_wav(samples, sample_rate=sample_rate, sample_width=sample_width)


def _encode_wav(
    samples: np.ndarray, *, sample_rate: int, sample_width: int,
) -> bytes:
    """Write ``(N, channels)`` floats in ``[-1, 1]`` to an in-memory WAV."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(samples.shape[1])
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(_to_pcm_bytes(samples, sample_width))
    return buf.getvalue()


def _to_pcm_bytes(samples: np.ndarray, sample_width: int) -> bytes:
    interleaved = samples.reshape(-1)
    if sample_width == 1:
        encoded = np.clip(interleaved * 128.0 + 128.0, 0, 255).astype(np.uint8)
        return encoded.tobytes()
    if sample_width == 2:
        encoded = np.clip(interleaved * float(2**15 - 1), -32768, 32767).astype("<i2")
        return encoded.tobytes()
    if sample_width == 4:
        encoded = np.clip(
            interleaved * float(2**31 - 1),
            -float(2**31),
            float(2**31 - 1),
        ).astype("<i4")
        return encoded.tobytes()
    raise ValueError(f"unsupported sample_width: {sample_width}")


# Keep ``struct`` reachable for tests that hand-build odd-length WAVs.
__all__ = ["build_sine_wav_bytes", "struct"]
