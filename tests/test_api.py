from __future__ import annotations

import base64
import pathlib

import numpy as np
import pytest

import plotwave


def test_public_api_exports_expected_symbols() -> None:
    expected = {
        "AudioTrace",
        "Plot",
        "SegmentsTrace",
        "SeriesTrace",
        "audio",
        "audio_trace_plot",
        "plot",
        "segments",
        "series",
    }

    assert set(plotwave.__all__) == expected
    for name in expected:
        assert hasattr(plotwave, name)


def test_plot_audio_simple_passes_trace_kwargs() -> None:
    wav = np.sin(np.linspace(0, 4 * np.pi, 1_000))
    plot = plotwave.plot(
        wav,
        sr=16_000,
        name="voice",
        line={"width": 2},
        opacity=0.4,
    )

    html = plot.html()

    assert '"name": "voice"' in html
    assert '"opacity": 0.4' in html
    assert '"width": 2' in html
    assert "Total Timespan:" in html
    assert '>voice</option>' in html
    assert "data:audio/mp3;base64," in html
    assert 'id="' in html
    assert ">Play</span>" in html
    assert '"range": [-1.499' in html or '"range": [-1.5' in html
    assert '1.499' in html or '1.5' in html


def test_plot_bitrate_resamples_embedded_audio() -> None:
    wav = np.sin(np.linspace(0, 8 * np.pi, 1_600))
    plot = plotwave.plot(wav, sr=16_000, bitrate="32k")

    prepared, _, _ = plot._resolved()
    audio_info = next(trace.audio_info for trace in prepared if trace.audio_info is not None)
    assert audio_info is not None
    assert audio_info["sample_rate"] == 8000
    assert audio_info["format"] == "mp3"

    audio_bytes = base64.b64decode(audio_info["b64_data"])
    assert audio_bytes[0] == 0xFF
    assert audio_bytes[1] & 0xE0 == 0xE0


def test_color_kw_sets_visible_scatter_line_color() -> None:
    time = np.linspace(0, 1, 5)

    html = plotwave.plot(
        [
            plotwave.audio(np.sin(2 * np.pi * time), 5, name="audio", color="#ff0000"),
            plotwave.series(np.cos(2 * np.pi * time), time=time, name="series", color="#00ff00"),
        ]
    ).html()

    assert '"line": {"color": "#ff0000"}' in html or '"color": "#ff0000"' in html
    assert '"line": {"color": "#00ff00"}' in html or '"color": "#00ff00"' in html


def test_plot_series_simple_with_time() -> None:
    time = np.linspace(0, 1, 500)
    y = np.cos(2 * np.pi * 3 * time)

    html = plotwave.plot(y, time=time, fill="tozeroy").html()

    assert '"fill": "tozeroy"' in html
    assert '"x": [0.0' in html


def test_series_with_sr_builds_time_axis_explicitly() -> None:
    wav = np.array([0.0, 1.0, 0.0, -1.0], dtype=float)
    env = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

    html = plotwave.plot(
        [
            plotwave.audio(wav, 4, name="wave"),
            plotwave.series(env, sr=4, name="env"),
        ]
    ).html()

    assert '"x": [0.0, 0.25, 0.5, 0.75]' in html


def test_multitrace_layout_and_config_are_propagated() -> None:
    time = np.linspace(0, 1, 500)
    wav = np.sin(2 * np.pi * 220 * time)
    env = np.abs(wav)

    plot = plotwave.plot(
        [
            plotwave.audio(wav, 16_000, name="wave", line={"width": 1.5}),
            plotwave.series(env, time=time, name="env", opacity=0.3),
        ],
        layout={"title": {"text": "Overlay"}, "height": 420},
        config={"scrollZoom": True},
    )

    html = plot.html()

    assert '"text": "Overlay"' in html
    assert '"scrollZoom": true' in html
    assert '"height": 420' in html
    assert "const globalTime = previousInfo.start_time + currentAudio.currentTime;" in html
    assert 'class="plotwave-time-label"' in html
    assert "Current Time:" in html
    assert "Total Timespan:" in html
    assert "const updateCurrentTimeLabel = (time) => {" in html
    assert "const setToggleState = (isPlaying) => {" in html
    assert 'const togglePlayback = () => {' in html
    assert 'const startPlayback = () => {' in html
    assert 'const stopPlayback = () => {' in html
    assert "let playbackAnchorAudioTime = 0;" in html
    assert "const setPlaybackAnchor = (audioTime = currentAudio.currentTime) => {" in html
    assert "const cursorFrameIntervalMs = 33;" in html
    assert 'const plotArea = plotDiv.querySelector(".nsewdrag");' in html
    assert "const eventToGlobalTime = (clientX) => {" in html
    assert "audioElement.ontimeupdate = () => {" in html
    assert 'const playResult = currentAudio.play();' in html
    assert 'rootElement.addEventListener("keydown", (event) => {' in html
    assert "width: 94px;" in html
    assert "height: 36px;" in html
    assert "display: inline-flex;" in html
    assert "flex-direction: column;" in html
    assert "flex-wrap: wrap;" in html
    assert "flex: 1 1 auto;" in html
    assert "justify-content: space-between;" in html
    assert "line-height: 1.2;" in html
    assert 'const resizePlot = () => Plotly.Plots.resize(plotDiv);' in html
    assert 'const resizeObserver = new ResizeObserver(() => resizePlot());' in html
    assert 'if (event.code === "Space") {' in html
    assert 'if (event.key === "j" || event.key === "J") {' in html
    assert 'if (event.key === "l" || event.key === "L") {' in html


def test_series_without_time_or_sr_raises_when_plotted_with_audio() -> None:
    with pytest.raises(ValueError, match="explicit time or sr"):
        plotwave.plot(
            [
                plotwave.audio(np.ones(4), 4, name="a"),
                plotwave.series(np.linspace(0.0, 1.0, 4), name="env"),
            ]
        ).html()


def test_segments_helper_adds_shapes_and_annotations() -> None:
    time = np.linspace(0, 4, 400)
    y = np.sin(2 * np.pi * time)

    html = plotwave.plot(
        [
            plotwave.series(y, time=time, name="wave"),
            plotwave.segments(
                [
                    {"start": 0.0, "end": 1.5, "label": "Bm", "color": "#2563eb"},
                    {"start": 2.0, "end": 3.5, "label": "G", "color": "#16a34a"},
                ],
                name="Chord",
            ),
        ]
    ).html()

    assert "<b>Bm</b>" in html
    assert "<b>G</b>" in html
    assert '"yref": "paper"' in html
    assert "rgba(37, 99, 235, 0.08)" in html
    assert "Chord: Bm<extra></extra>" in html
    assert "Chord: G<extra></extra>" in html
    assert '"y0": 0.5' in html
    assert '"y1": 1.0' in html


def test_segments_hover_uses_reference_times_from_other_traces() -> None:
    time = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    html = plotwave.plot(
        [
            plotwave.series(np.ones_like(time), time=time, name="wave"),
            plotwave.segments(
                [{"start": 0.2, "end": 0.8, "label": "Bm"}],
                name="Pred",
            ),
        ]
    ).html()

    assert '"x": [0.25, 0.5, 0.75]' in html
    assert "Pred: Bm<extra></extra>" in html


def test_segments_helper_uses_color_map_when_item_color_is_missing() -> None:
    html = plotwave.plot(
        [
            plotwave.segments(
                [
                    {"start": 0.0, "end": 1.0, "label": "Bm"},
                    {"start": 1.5, "end": 2.5, "label": "G", "color": "#ff00aa"},
                ],
                color_map={"Bm": "#2563eb", "G": "#16a34a"},
            )
        ]
    ).html()

    assert "rgba(37, 99, 235, 0.08)" in html
    assert "rgba(255, 0, 170, 0.08)" in html


def test_segments_helper_supports_bottom_lane() -> None:
    html = plotwave.plot(
        [
            plotwave.segments(
                [
                    {"start": 0.0, "end": 1.0, "label": "truth", "color": "#16a34a"},
                ],
                lane="bottom",
            )
        ]
    ).html()

    assert '"y0": 0.0' in html
    assert '"y1": 0.5' in html


def test_save_writes_html_file(tmp_path: pathlib.Path) -> None:
    y = np.linspace(-1, 1, 100)
    plot = plotwave.plot(y)
    output = plot.save(tmp_path / "wave.html")

    assert output.exists()
    assert "<!DOCTYPE html>" in output.read_text(encoding="utf-8")


def test_repr_html_returns_iframe_markup() -> None:
    y = np.linspace(-1, 1, 64)

    markup = plotwave.plot(y)._repr_html_().lstrip()

    assert markup.startswith("<iframe")
    assert "srcdoc=" in markup


def test_invalid_inputs_raise_clean_errors() -> None:
    with pytest.raises(ValueError, match="positive"):
        plotwave.audio([0.0, 1.0], 0)

    with pytest.raises(ValueError, match="same length"):
        plotwave.series([0.0, 1.0], time=[0.0])

    with pytest.raises(ValueError, match="either time or sr"):
        plotwave.series([0.0, 1.0], time=[0.0, 1.0], sr=2)

    with pytest.raises(ValueError, match="positive"):
        plotwave.series([0.0, 1.0], sr=0)

    with pytest.raises(ValueError, match="cannot be empty"):
        plotwave.plot([])

    with pytest.raises(ValueError, match="bitrate must be a positive number"):
        plotwave.plot([0.0, 1.0], bitrate=0)

    with pytest.raises(ValueError, match="string like '64k'"):
        plotwave.plot([0.0, 1.0], bitrate="fast")


def test_legacy_audio_trace_plot_modes(tmp_path: pathlib.Path) -> None:
    time = np.linspace(0, 1, 200)
    wav = np.sin(2 * np.pi * 4 * time)

    output = plotwave.audio_trace_plot(
        [
            {"type": "audio", "x": time, "y": wav, "sr": 16_000, "name": "legacy"},
        ],
        output_mode="file",
        output_path=str(tmp_path / "legacy.html"),
    )

    assert output is not None
    assert pathlib.Path(output).exists()
