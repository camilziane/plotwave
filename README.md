# plotwave
<p align="left">
  <img src="https://raw.githubusercontent.com/camilziane/plotwave/main/assets/logo_name.png" alt="plotwave logo" width="200">
</p>

<p align="left">
  Interactive Plotly waveforms with synchronized audio playback
</p>

<p align="left">
  <a href="https://camilziane.github.io/plotwave/">Live interactive demo</a>
</p>

**Click anywhere in the waveform to hear the audio while inspecting it visually.  
Overlay multiple audio tracks or add additional signals (labels, predictions, segmentation, scores) on top of the waveform.**

Designed for **Jupyter notebooks**, `plotwave` can also be exported to **HTML**, making it easy to share interactive audio visualizations or log them in tools like **MLflow** for experiment analysis.

## Why plotwave

- Start from a path with `plotwave.audio_file(...)` and get an interactive waveform without manually decoding audio first.
- Click anywhere in the waveform to audition the exact moment you are inspecting.
- Click a segment label to loop that labeled region and compare it against the waveform.
- Overlay predictions, envelopes, scores, or ground-truth labels on the same timeline.
- Use the same interaction model in notebooks and exported HTML.

## Install

```bash
uv add plotwave
```

or

```bash
pip install plotwave
```

## Direct audio file example

```python
import plotwave

plotwave.audio_file("voice.mp3", name="voice").plot(
    layout={"title": {"text": "Direct file-backed audio"}, "height": 460},
)
```

`plotwave.audio_file(...)` uses `soundfile` internally and can open any local audio format that your `soundfile` backend supports, including WAV and MP3. It is the quickest way to go from a local audio file to an interactive player + waveform view.
Use `soundfile.read(...)` only when you want to derive an extra trace from the samples, like an envelope or model scores.

## Audio + curve

```python
import numpy as np
import soundfile as sf
import plotwave

wav, sr = sf.read("voice.mp3", always_2d=False)
env = np.abs(wav)

plotwave.plot(
    [
        plotwave.audio_file("voice.mp3", name="audio", color="#2563eb"),
        plotwave.series(env, sr=sr, name="envelope", color="#f97316", fill="tozeroy"),
    ],
    layout={"title": {"text": "Audio + envelope"}, "height": 520},
)
```

For `series(...)`, use:

- `sr=...` when your values are evenly sampled
- `time=...` when you already have an explicit time axis

## Segments and label looping

```python
import plotwave

plotwave.plot(
    [
        plotwave.audio_file("song.mp3", name="audio"),
        plotwave.segments(
            [
                {"start": 0.0, "end": 0.7, "label": "Bm"},
                {"start": 1.0, "end": 1.6, "label": "G"},
            ],
            name="Pred",
            lane="top",
            color_map={"Bm": "#2563eb", "G": "#16a34a"},
        ),
    ]
)
```

`segments(...)` adds:

- clickable label boxes that loop the matching audio span
- colored background blocks
- hoverable segment names
- top/bottom lanes for comparisons like prediction vs ground truth

## Export

```python
plot = plotwave.plot(wav, sr=sr)
plot.save("wave.html")
html = plot.html()
```

By default, notebook output, `html()`, and `save()` use compressed audio. You can disable that and control the bitrate explicitly:

```python
plot = plotwave.audio_file("voice.mp3").plot(compress_audio=False)
html = plot.html(compress_audio=True, bitrate="48k")
```

## API

Public API:

- `plotwave.plot(...)`
- `plotwave.audio(...)`
- `plotwave.audio_file(...)`
- `plotwave.series(...)`
- `plotwave.segments(...)`
- `plotwave.Plot`

All trace types also have `trace.plot(...)` as a shortcut for `plotwave.plot(trace, ...)`.

See [examples/getting_started.ipynb](https://github.com/camilziane/plotwave/blob/main/examples/getting_started.ipynb) for a full walkthrough.

Developer workflow: [DEVELOPERS.md](https://github.com/camilziane/plotwave/blob/main/DEVELOPERS.md)

To refresh the GitHub Pages demo locally:

```bash
uv run python scripts/build_pages_demo.py
```
