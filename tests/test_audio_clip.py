"""Tests for the WAV loader + waveform-peak summariser."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from posecascade.audio.clip import (
    DEFAULT_PEAK_BIN_COUNT,
    AudioClip,
    load_wav_bytes,
    load_wav_file,
)
from posecascade.errors import MalformedAssetError
from tests.fixtures.audio.build_wav import build_sine_wav_bytes


def test_mono_sine_loads_with_correct_metadata() -> None:
    data = build_sine_wav_bytes(duration_seconds=0.5, sample_rate=22050, channels=1)
    clip = load_wav_bytes(data)
    assert clip.sample_rate == 22050
    assert clip.channels == 1
    assert clip.duration_seconds == pytest.approx(0.5, abs=1.0e-3)
    assert clip.peaks.shape[1] == 2
    assert clip.peaks.shape[0] > 0


def test_stereo_clip_decodes_both_channels() -> None:
    data = build_sine_wav_bytes(channels=2)
    clip = load_wav_bytes(data)
    assert clip.channels == 2
    # Peaks span the full ±0.5 amplitude (we generated a 0.5-amp sine).
    assert clip.peaks[:, 1].max() == pytest.approx(0.5, abs=0.05)
    assert clip.peaks[:, 0].min() == pytest.approx(-0.5, abs=0.05)


def test_eight_bit_pcm_decodes() -> None:
    """8-bit WAV is unsigned-centred at 128; the loader must re-centre."""
    data = build_sine_wav_bytes(sample_width=1)
    clip = load_wav_bytes(data)
    assert clip.peaks[:, 0].min() < 0.0
    assert clip.peaks[:, 1].max() > 0.0


def test_thirty_two_bit_pcm_decodes() -> None:
    data = build_sine_wav_bytes(sample_width=4)
    clip = load_wav_bytes(data)
    assert clip.duration_seconds > 0


def test_default_peak_bin_count_used() -> None:
    """Bin count is "≤ DEFAULT" — the exact value drifts slightly when the
    sample length isn't an integer multiple of the bin width."""
    data = build_sine_wav_bytes(duration_seconds=2.0)
    clip = load_wav_bytes(data)
    assert 0 < clip.peaks.shape[0] <= DEFAULT_PEAK_BIN_COUNT


def test_custom_peak_bin_count_used() -> None:
    """A bin count that exactly divides the sample length lands on the nose."""
    data = build_sine_wav_bytes(duration_seconds=64.0 / 44100.0)
    clip = load_wav_bytes(data, peak_bins=64)
    assert clip.peaks.shape[0] == 64


def test_zero_peak_bins_returns_empty() -> None:
    data = build_sine_wav_bytes()
    clip = load_wav_bytes(data, peak_bins=0)
    assert clip.peaks.shape == (0, 2)


def test_load_wav_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    path.write_bytes(build_sine_wav_bytes())
    clip = load_wav_file(path)
    assert isinstance(clip, AudioClip)
    assert clip.path == path.resolve()


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MalformedAssetError, match="not found"):
        load_wav_file(tmp_path / "nope.wav")


def test_invalid_bytes_raise() -> None:
    with pytest.raises(MalformedAssetError, match="WAV"):
        load_wav_bytes(b"not_a_wav_file")


def test_peak_summary_is_monotone_in_amplitude() -> None:
    """Quiet vs loud clips: peaks of the louder one must dominate."""
    quiet = load_wav_bytes(build_sine_wav_bytes(amplitude=0.1))
    loud = load_wav_bytes(build_sine_wav_bytes(amplitude=0.9))
    assert loud.peaks[:, 1].max() > quiet.peaks[:, 1].max()
    assert loud.peaks[:, 0].min() < quiet.peaks[:, 0].min()


# Keep np importable through this module for downstream test reads.
__all__ = ["np"]
