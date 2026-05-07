"""Audio playback + waveform-preview utilities.

The clip layer (:mod:`posecascade.audio.clip`) is pure stdlib + numpy and
runs without any Qt or sound device — useful for tests + offline
preview generation. The player layer (:mod:`posecascade.audio.player`)
wraps Qt's :class:`QMediaPlayer` for actual playback, with a wall-clock
fallback when the Qt audio backend fails or is unavailable.
"""

from posecascade.audio.clip import AudioClip, load_wav_bytes, load_wav_file
from posecascade.audio.player import AudioPlayer

__all__ = [
    "AudioClip",
    "AudioPlayer",
    "load_wav_bytes",
    "load_wav_file",
]
