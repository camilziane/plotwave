from __future__ import annotations

import numpy as np

CHORDS: dict[str, list[float]] = {
    "Bm": [123.47, 146.83, 185.00],
    "G": [98.00, 123.47, 146.83],
    "D": [146.83, 185.00, 220.00],
    "A": [110.00, 138.59, 164.81],
    "F#m": [92.50, 110.00, 138.59],
    "E": [82.41, 103.83, 123.47],
}


def synth_tone(
    freq: float,
    *,
    sr: int,
    seconds: float,
    amp: float = 0.18,
) -> np.ndarray:
    time = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    tone = (
        np.sin(2 * np.pi * freq * time)
        + 0.32 * np.sin(2 * np.pi * 2 * freq * time)
        + 0.12 * np.sin(2 * np.pi * 3 * freq * time)
    )
    attack = max(1, int(0.03 * sr))
    release = max(1, int(0.12 * sr))
    env = np.ones_like(time)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    env[-release:] = np.linspace(1.0, 0.0, release)
    return amp * tone * env


def synth_chord(
    freqs: list[float],
    *,
    sr: int,
    seconds: float,
    amp: float = 0.24,
) -> np.ndarray:
    layers = [
        synth_tone(freq, sr=sr, seconds=seconds, amp=amp / len(freqs))
        for freq in freqs
    ]
    chord = np.sum(layers, axis=0)
    peak = np.max(np.abs(chord))
    if peak > 0:
        chord = 0.85 * chord / peak
    return chord


def build_progression(
    names: list[str],
    *,
    sr: int,
    chord_seconds: float = 2.0,
    gap_seconds: float = 1.0,
    amp: float = 0.24,
    start_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str]]]:
    parts: list[np.ndarray] = []
    segments: list[dict[str, float | str]] = []
    cursor = start_offset

    for index, name in enumerate(names):
        chord = synth_chord(CHORDS[name], sr=sr, seconds=chord_seconds, amp=amp)
        parts.append(chord)
        segments.append(
            {
                "name": name,
                "start": cursor,
                "end": cursor + chord_seconds,
                "index": float(index),
            }
        )
        cursor += chord_seconds
        if index < len(names) - 1:
            parts.append(np.zeros(int(sr * gap_seconds)))
            cursor += gap_seconds

    wav = np.concatenate(parts)
    time = start_offset + np.arange(len(wav), dtype=float) / sr
    return time, wav, segments


def smooth_envelope(wav: np.ndarray, window: int = 3_000) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(np.abs(wav), kernel, mode="same")


def segment_items(
    segments: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    items: list[dict[str, float | str]] = []
    for segment in segments:
        name = str(segment["name"])
        items.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "label": name,
            }
        )
    return items
