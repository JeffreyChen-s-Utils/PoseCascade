"""WAV file loader + waveform peak summary.

PoseCascade only needs WAV — MMD's audio convention. The loader stays
in stdlib ``wave`` plus numpy so the whole module is import-clean
without any audio backend, which makes the timeline's waveform preview
a pure data step that runs in tests + headless CI.

``AudioClip.peaks`` is a downsampled ``(N, 2)`` array of ``(min, max)``
amplitudes per bin, scaled to the float range ``[-1, 1]``. The timeline
overlay reads this directly to draw the silhouette.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from posecascade.errors import MalformedAssetError

DEFAULT_PEAK_BIN_COUNT = 1024
_INT16_NORMALIZER = float(2**15)
_SUPPORTED_SAMPLE_WIDTHS = (1, 2, 4)
_SAMPLE_WIDTH_8BIT = 1
_SAMPLE_WIDTH_16BIT = 2


@dataclass(frozen=True)
class AudioClip:
    """A loaded WAV file + a downsampled waveform peak summary."""

    path: Path | None
    sample_rate: int
    channels: int
    duration_seconds: float
    peaks: NDArray[np.float32]   # (N, 2) — (min, max) per bin


def load_wav_file(path: Path, *, peak_bins: int = DEFAULT_PEAK_BIN_COUNT) -> AudioClip:
    """Load a WAV file from disk into an :class:`AudioClip`."""
    path = path.resolve()
    if not path.is_file():
        raise MalformedAssetError(f"WAV file not found: {path}")
    return _load_clip(path.read_bytes(), path=path, peak_bins=peak_bins)


def load_wav_bytes(
    data: bytes, *, peak_bins: int = DEFAULT_PEAK_BIN_COUNT,
) -> AudioClip:
    """Load a WAV byte buffer into an :class:`AudioClip` (no path)."""
    return _load_clip(data, path=None, peak_bins=peak_bins)


def _load_clip(
    data: bytes, *, path: Path | None, peak_bins: int,
) -> AudioClip:
    try:
        with wave.open(BytesIO(data), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except wave.Error as err:
        raise MalformedAssetError(f"unreadable WAV: {err}") from err
    if sample_width not in _SUPPORTED_SAMPLE_WIDTHS:
        raise MalformedAssetError(
            f"unsupported WAV sample width: {sample_width} bytes",
        )
    samples = _decode_samples(frames, sample_width, channels)
    duration = float(samples.shape[0]) / float(sample_rate or 1)
    peaks = _compute_peaks(samples, peak_bins)
    return AudioClip(
        path=path,
        sample_rate=int(sample_rate),
        channels=int(channels),
        duration_seconds=duration,
        peaks=peaks,
    )


def _decode_samples(
    frames: bytes, sample_width: int, channels: int,
) -> NDArray[np.float32]:
    """Decode raw PCM bytes into a float32 ``(N, channels)`` array in ``[-1, 1]``."""
    if sample_width == _SAMPLE_WIDTH_8BIT:
        # 8-bit WAV is unsigned; centre at zero.
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == _SAMPLE_WIDTH_16BIT:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / _INT16_NORMALIZER
    else:
        # 32-bit PCM — Python's `wave` module supports it but it's rare for
        # MMD audio. Treat as signed int32.
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / float(2**31)
    return samples.reshape(-1, channels) if channels > 1 else samples.reshape(-1, 1)


def _compute_peaks(
    samples: NDArray[np.float32], bin_count: int,
) -> NDArray[np.float32]:
    """Return ``(bin_count, 2)`` ``(min, max)`` peaks per bin (averaged across channels)."""
    if samples.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    mono = samples.mean(axis=1)
    if bin_count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    # Round up so the resulting bin count never exceeds the requested
    # ``bin_count`` (the previous floor-and-truncate left a stray
    # tail-bin when ``len(mono) % bin_count != 0``).
    samples_per_bin = max(1, -(-mono.shape[0] // bin_count))
    usable = (mono.shape[0] // samples_per_bin) * samples_per_bin
    if usable == 0:
        return np.zeros((0, 2), dtype=np.float32)
    reshaped = mono[:usable].reshape(-1, samples_per_bin)
    mins = reshaped.min(axis=1)
    maxs = reshaped.max(axis=1)
    return np.stack([mins, maxs], axis=1).astype(np.float32, copy=False)
