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


<p align="left">
  <img src="https://raw.githubusercontent.com/camilziane/plotwave/main/assets/example.png" alt="plotwave example output" width="1000">
</p>
<p align="left">
  Example of aplotwave output
</p>

## Install

```bash
uv add plotwave
```

or

```bash
pip install plotwave
```

## Smallest useful example

```python
import soundfile as sf
import plotwave

wav, sr = sf.read("wave.wav", always_2d=False)

plotwave.plot(wav, sr=sr, name="voice")
```

`soundfile` is only used here as a convenient audio loader. `plotwave` itself only depends on `numpy`.

## Audio + curve

```python
env = np.abs(wav)

plotwave.plot(
    [
        plotwave.audio(wav, sr, name="audio", color="#2563eb"),
        plotwave.series(env, name="envelope", color="#f97316", fill="tozeroy"),
    ],
    layout={"title": {"text": "Audio + envelope"}, "height": 520},
)
```

If `series(..., time=None)` is plotted next to audio, `plotwave` infers its time axis automatically when the audio timing is unambiguous.

## Segments

```python
plotwave.plot(
    [
        plotwave.audio(wav, sr, name="audio"),
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

- colored background blocks
- label boxes
- hoverable segment names
- top/bottom lanes for comparisons like prediction vs ground truth

## Export

```python
plot = plotwave.plot(wav, sr=sr)
plot.save("wave.html")
html = plot.html()
```

## API

Public API:

- `plotwave.plot(...)`
- `plotwave.audio(...)`
- `plotwave.series(...)`
- `plotwave.segments(...)`
- `plotwave.Plot`

See [examples/getting_started.ipynb](https://github.com/camilziane/plotwave/blob/main/examples/getting_started.ipynb) for a full walkthrough.

Developer workflow: [DEVELOPERS.md](https://github.com/camilziane/plotwave/blob/main/DEVELOPERS.md)

To refresh the GitHub Pages demo locally:

```bash
uv run python scripts/build_pages_demo.py
```
