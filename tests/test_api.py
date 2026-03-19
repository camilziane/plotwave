from __future__ import annotations

import base64
import pathlib
import sys
import tomllib
import types
import wave

import lameenc
import numpy as np
import pytest

import plotwave
import plotwave._core as core


def _test_tone(*, sr: int = 16_000, duration: float = 0.25) -> np.ndarray:
    sample_count = max(1, int(sr * duration))
    time = np.arange(sample_count, dtype=np.float64) / sr
    return 0.4 * np.sin(2 * np.pi * 220 * time)


def _write_test_wav(
    path: pathlib.Path,
    *,
    sr: int = 16_000,
    duration: float = 0.25,
) -> pathlib.Path:
    pcm16 = (_test_tone(sr=sr, duration=duration) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm16.tobytes())
    return path


def _write_test_mp3(
    path: pathlib.Path,
    *,
    sr: int = 16_000,
    duration: float = 0.25,
) -> pathlib.Path:
    pcm16 = (_test_tone(sr=sr, duration=duration) * 32767).astype("<i2")
    encoder = lameenc.Encoder()
    encoder.set_channels(1)
    encoder.set_in_sample_rate(sr)
    encoder.set_out_sample_rate(sr)
    encoder.set_bit_rate(128)
    encoder.set_quality(2)
    path.write_bytes(bytes(encoder.encode(pcm16.tobytes()) + encoder.flush()))
    return path


def test_public_api_exports_expected_symbols() -> None:
    expected = {
        "AudioFileTrace",
        "AudioTrace",
        "Plot",
        "SegmentsTrace",
        "SeriesTrace",
        "audio",
        "audio_file",
        "audio_trace_plot",
        "plot",
        "segments",
        "series",
    }

    assert set(plotwave.__all__) == expected
    for name in expected:
        assert hasattr(plotwave, name)


def test_package_version_matches_pyproject() -> None:
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert plotwave.__version__ == metadata["project"]["version"]


def test_plot_audio_simple_passes_trace_kwargs() -> None:
    wav = np.sin(np.linspace(0, 4 * np.pi, 1_000))
    plot = plotwave.plot(
        wav,
        sr=16_000,
        points=3_000,
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


def test_plot_default_points_scale_with_duration() -> None:
    wav = np.sin(np.linspace(0, 4 * np.pi, 4_001))

    prepared, _, _ = plotwave.plot(wav, sr=4_000)._resolved(embed_audio=False)
    trace = prepared[0].plotly_trace

    assert len(trace["x"]) == 3_000
    assert len(trace["y"]) == 3_000


def test_plot_default_points_grow_beyond_3000_for_long_duration() -> None:
    wav = np.sin(np.linspace(0, 400 * np.pi, 6_401))

    prepared, _, _ = plotwave.plot(wav, sr=16)._resolved(embed_audio=False)

    assert len(prepared[0].plotly_trace["x"]) == 6_400


def test_audio_plot_range_uses_full_duration() -> None:
    wav = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)

    _, layout, _ = plotwave.plot(wav, sr=4, points=-1)._resolved(embed_audio=False)

    assert layout["xaxis"]["range"] == [0.0, 1.0]


def test_plot_points_override_still_wins_over_duration_default() -> None:
    wav = np.sin(np.linspace(0, 4 * np.pi, 33))

    prepared, _, _ = plotwave.plot(wav, sr=16, points=12)._resolved(embed_audio=False)

    assert len(prepared[0].plotly_trace["x"]) == 12


def test_plot_points_minus_one_displays_all_points() -> None:
    wav = np.sin(np.linspace(0, 4 * np.pi, 33))

    prepared, _, _ = plotwave.plot(wav, sr=16, points=-1)._resolved(embed_audio=False)

    assert len(prepared[0].plotly_trace["x"]) == 33


def test_plot_clip_normalizes_against_clipped_peak() -> None:
    wav = np.array([-100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0], dtype=np.float64)

    prepared, _, _ = plotwave.plot(
        wav,
        sr=16,
        points=-1,
        clip=0.25,
        norm=True,
    )._resolved(embed_audio=False)

    trace = prepared[0].plotly_trace

    assert trace["y"] == [-1.0, -1.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, 1.0, 1.0]


def test_plot_without_time_defaults_to_all_points() -> None:
    values = np.linspace(-1.0, 1.0, 4_001)

    prepared, _, _ = plotwave.plot(values)._resolved(embed_audio=False)

    assert len(prepared[0].plotly_trace["x"]) == 4_001
    assert len(prepared[0].plotly_trace["y"]) == 4_001


def test_audio_file_embeds_audio_for_portable_html(tmp_path: pathlib.Path) -> None:
    wav_path = _write_test_wav(tmp_path / "clip.wav")

    html = plotwave.plot([plotwave.audio_file(wav_path)]).html()

    assert "data:audio/mp3;base64," in html
    assert '"duration": 0.25' in html


def test_audio_file_reads_mp3_for_portable_html(tmp_path: pathlib.Path) -> None:
    mp3_path = _write_test_mp3(tmp_path / "clip.mp3")

    trace = plotwave.audio_file(mp3_path)
    html = plotwave.plot([trace]).html()

    assert trace.audio_format == "mp3"
    assert trace.frames > 0
    assert trace.sr > 0
    assert "data:audio/mp3;base64," in html
    assert '"duration": ' in html


def test_audio_file_uses_decoded_frame_count_for_mp3_metadata(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mp3_path = tmp_path / "clip.mp3"
    mp3_path.write_bytes(b"fake mp3")

    class FakeInfo:
        frames = 408_115
        samplerate = 48_000
        format = "MP3"

    monkeypatch.setattr(core.sf, "info", lambda _: FakeInfo())
    monkeypatch.setattr(core, "_decoded_soundfile_frame_count", lambda _: 403_200)

    trace = plotwave.audio_file(mp3_path)

    assert trace.frames == 403_200
    assert trace.sr == 48_000
    assert trace.audio_format == "mp3"


def test_audio_file_raises_clear_error_when_soundfile_cannot_decode(
    tmp_path: pathlib.Path,
) -> None:
    bogus_path = tmp_path / "bogus.audio"
    bogus_path.write_text("not really audio", encoding="utf-8")

    with pytest.raises(RuntimeError, match="soundfile could not decode"):
        plotwave.audio_file(bogus_path)


def test_audio_file_display_sampling_avoids_full_decode(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = _write_test_wav(tmp_path / "long.wav", duration=3.0)
    trace = plotwave.audio_file(wav_path, step=8)
    called = False

    def fail(_: pathlib.Path) -> np.ndarray:
        nonlocal called
        called = True
        raise AssertionError("display trace should not fully decode the audio file")

    monkeypatch.setattr(core, "_read_soundfile_samples", fail)

    plotly_trace = trace._display_trace(points=40)

    assert not called
    assert len(plotly_trace["x"]) == 40
    assert len(plotly_trace["y"]) == 40


def test_audio_file_clip_normalizes_against_clipped_peak(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = _write_test_wav(tmp_path / "clip.wav")
    trace = plotwave.audio_file(wav_path, norm=True, clip=0.25)

    monkeypatch.setattr(
        core,
        "_soundfile_display_trace",
        lambda path, *, frames, sr, points, step: (
            np.arange(7, dtype=np.float64),
            np.array([-100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0], dtype=np.float64),
        ),
    )

    plotly_trace = trace._display_trace(points=7)

    assert plotly_trace["y"] == [-1.0, -1.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, 1.0, 1.0]


def test_trace_plot_matches_top_level_constructor(tmp_path: pathlib.Path) -> None:
    wav_path = _write_test_wav(tmp_path / "trace.wav")
    traces = [
        plotwave.audio(np.linspace(-1.0, 1.0, 32), 16_000, name="audio"),
        plotwave.audio_file(wav_path, name="file"),
        plotwave.series(np.linspace(0.0, 1.0, 32), sr=16, name="series"),
        plotwave.segments([{"start": 0.0, "end": 1.0, "label": "A"}], name="segment"),
    ]

    for trace in traces:
        via_method = trace.plot(
            layout={"title": {"text": "Trace Plot"}},
            config={"scrollZoom": True},
            points=64,
            display="none",
            compress_audio=False,
            bitrate="32k",
        )
        via_function = plotwave.plot(
            trace,
            layout={"title": {"text": "Trace Plot"}},
            config={"scrollZoom": True},
            points=64,
            display="none",
            compress_audio=False,
            bitrate="32k",
        )

        assert isinstance(via_method, plotwave.Plot)
        assert via_method.compress_audio is False
        assert via_method.bitrate == "32k"
        assert via_method._resolved(embed_audio=False) == via_function._resolved(embed_audio=False)


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
    assert "let activeLoop = null;" in html
    assert "const maybeWrapLoop = () => {" in html
    assert "const activateLoop = (segment) => {" in html
    assert "const segmentLoopForPoint = (globalTime, paperY) => {" in html
    assert "const updateHoverCursor = (clientX, clientY) => {" in html
    assert "const updateLoopVisuals = () => {" in html
    assert 'const inactiveBoxFill = "rgba(150, 150, 150, 0.82)";' in html
    assert "let playbackAnchorAudioTime = 0;" in html
    assert (
        "const setPlaybackAnchor = "
        "(audioTime = currentAudio.currentTime, force = false) => {"
    ) in html
    assert "if (drift < 0 && Math.abs(drift) < 0.12) return;" in html
    assert "let syncCursorPosition = null;" in html
    assert "const syncCursorOverlayBounds = () => {" in html
    assert 'cursorLine.style.transform = `translate3d(${clampedX}px, 0, 0)`;' in html
    assert 'const plotArea = plotDiv.querySelector(".nsewdrag");' in html
    assert ".plotwave-label-hover * {" in html
    assert "cursor: pointer !important;" in html
    assert "pointer-events: none;" in html
    assert "will-change: transform;" in html
    assert 'plotDiv.classList.toggle("plotwave-label-hover", Boolean(hoverSegment));' in html
    assert "const eventToGlobalTime = (clientX) => {" in html
    assert "audioElement.ontimeupdate = () => {" in html
    assert 'audioElement.preload = info.duration <= 20 ? "auto" : "metadata";' in html
    assert 'if (audioElement.preload === "auto") {' in html
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
    assert 'const cursorResizeObserver = new ResizeObserver(() => {' in html
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

    assert '"x": [0.2, 0.25, 0.5, 0.75, 0.8]' in html
    assert "Pred: Bm<extra></extra>" in html


def test_segments_hover_preserves_explicit_segment_endpoints() -> None:
    time = np.array([0.0, 2.12, 4.2], dtype=np.float64)

    prepared, layout, _ = plotwave.plot(
        [
            plotwave.series(np.ones_like(time), time=time, name="wave"),
            plotwave.segments(
                [{"start": 2.12, "end": 4.24, "label": "Asus2"}],
                name="Pred",
            ),
        ]
    )._resolved(embed_audio=False)

    hover_trace = next(
        trace
        for trace in prepared
        if trace.plotly_trace.get("hovertemplate") == "Pred: Asus2<extra></extra>"
    )

    assert hover_trace.plotly_trace["x"] == [2.12, 4.2, 4.24]
    assert layout["xaxis"]["range"][1] == pytest.approx(4.24)


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


def test_segment_label_click_exports_loop_metadata_for_audio() -> None:
    wav = np.sin(np.linspace(0, 6 * np.pi, 512))

    html = plotwave.plot(
        [
            plotwave.audio(wav, sr=128, name="clip"),
            plotwave.segments(
                [{"start": 0.5, "end": 1.5, "label": "Verse"}],
                lane="top",
            ),
        ]
    ).html()

    assert (
        'const segmentInfos = [{"start": 0.5, "end": 1.5, "label": "Verse", "lane": "top"}];'
        in html
    )
    assert "const loopEpsilonSeconds = 0.01;" in html
    assert "const findAudioIndexForRange = (startTime, endTime) => {" in html
    assert 'console.warn("plotwave could not loop segment outside audio bounds", segment);' in html
    assert "const eventToPaperY = (clientY) => {" in html
    assert "if (segmentLoop === activeLoop) {" in html
    assert "Plotly.relayout(plotDiv, loopVisualUpdates);" in html
    assert "endTime <= info.start_time + info.duration + loopEpsilonSeconds" in html
    assert "globalTime <= segment.end + loopEpsilonSeconds" in html


def test_save_writes_html_file(tmp_path: pathlib.Path) -> None:
    y = np.linspace(-1, 1, 100)
    plot = plotwave.plot(y)
    output = plot.save(tmp_path / "wave.html")

    assert output.exists()
    assert "<!DOCTYPE html>" in output.read_text(encoding="utf-8")


def test_html_and_save_override_audio_compression_for_array_audio(
    tmp_path: pathlib.Path,
) -> None:
    wav = np.sin(np.linspace(0, 6 * np.pi, 512))
    plot = plotwave.plot(wav, sr=16_000, compress_audio=False)

    default_html = plot.html()
    override_html = plot.html(compress_audio=True, bitrate="32k")
    output = plot.save(tmp_path / "wave.html", compress_audio=True, bitrate="32k")
    saved_html = output.read_text(encoding="utf-8")

    assert "data:audio/wav;base64," in default_html
    assert "data:audio/mp3;base64," in override_html
    assert "data:audio/mp3;base64," in saved_html


def test_html_and_save_preserve_original_file_format_when_uncompressed(
    tmp_path: pathlib.Path,
) -> None:
    mp3_path = _write_test_mp3(tmp_path / "clip.mp3")
    plot = plotwave.plot([plotwave.audio_file(mp3_path)])

    html = plot.html(compress_audio=False)
    output = plot.save(tmp_path / "clip.html", compress_audio=False)

    assert "data:audio/mp3;base64," in html
    assert "data:audio/mp3;base64," in output.read_text(encoding="utf-8")


def test_repr_html_returns_iframe_markup(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    y = np.linspace(-1, 1, 64)
    monkeypatch.chdir(tmp_path)

    markup = plotwave.plot(y)._repr_html_().lstrip()

    assert markup.startswith("<iframe")
    assert "srcdoc=" in markup


def test_show_uses_plot_repr_once_in_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    displayed: list[object] = []
    ipython_module = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")

    def fake_display(value: object) -> None:
        displayed.append(value)

    display_module.display = fake_display  # type: ignore[attr-defined]
    ipython_module.display = display_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", ipython_module)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    monkeypatch.setattr(core, "_is_notebook", lambda: True)

    plot = plotwave.plot(np.linspace(-1, 1, 64))

    assert plot.show() is None
    assert displayed == [plot]


def test_repr_html_resolves_plot_once(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = plotwave.Plot._resolved
    monkeypatch.chdir(tmp_path)

    def wrapped(
        self: plotwave.Plot,
        *,
        embed_audio: bool = True,
        compress_audio: bool | None = None,
        bitrate: object = core._BITRATE_UNSET,
    ) -> tuple[list[object], dict[str, object], dict[str, object]]:
        nonlocal calls
        calls += 1
        return original(
            self,
            embed_audio=embed_audio,
            compress_audio=compress_audio,
            bitrate=bitrate,
        )

    monkeypatch.setattr(plotwave.Plot, "_resolved", wrapped)

    plotwave.plot(np.linspace(-1, 1, 64))._repr_html_()

    assert calls == 1


def test_repr_html_uses_jupyter_asset_urls(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "asset")

    markup = plotwave.plot(np.linspace(-1, 1, 128), sr=16_000, bitrate="32k")._repr_html_()

    assert "plotwave-assets/" in markup
    assert any((tmp_path / "plotwave-assets").glob("*.mp3"))


def test_repr_html_reuses_cached_notebook_audio_asset(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "asset")
    wav = np.sin(np.linspace(0, 8 * np.pi, 1_600))

    first_markup = plotwave.plot(wav, sr=16_000, bitrate="32k")._repr_html_()
    second_markup = plotwave.plot(wav, sr=16_000, bitrate="32k")._repr_html_()

    assets = list((tmp_path / "plotwave-assets").glob("*.mp3"))

    assert len(assets) == 1
    assert assets[0].name in first_markup
    assert assets[0].name in second_markup


def test_repr_html_normalizes_notebook_audio_urls(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "asset")

    markup = plotwave.plot(np.linspace(-1, 1, 128), sr=16_000, bitrate="32k")._repr_html_()

    assert "notebookBaseUrl" in markup
    assert "plotwave-assets/" in markup
    assert "metadata" in markup


def test_repr_html_uses_inline_audio_when_forced(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "inline")

    markup = plotwave.plot(np.linspace(-1, 1, 128), sr=16_000, bitrate="32k")._repr_html_()

    assert "data:audio/mp3;base64," in markup
    assert any((tmp_path / "plotwave-assets").glob("*.mp3"))


def test_repr_html_uses_inline_audio_in_vscode(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VSCODE_PID", "12345")

    markup = plotwave.plot(np.linspace(-1, 1, 128), sr=16_000, bitrate="32k")._repr_html_()

    assert "data:audio/mp3;base64," in markup
    assert any((tmp_path / "plotwave-assets").glob("*.mp3"))


def test_repr_html_reuses_cached_notebook_audio_asset_when_inline(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "inline")
    wav = np.sin(np.linspace(0, 8 * np.pi, 1_600))

    first_markup = plotwave.plot(wav, sr=16_000, bitrate="32k")._repr_html_()
    second_markup = plotwave.plot(wav, sr=16_000, bitrate="32k")._repr_html_()

    assets = list((tmp_path / "plotwave-assets").glob("*.mp3"))

    assert len(assets) == 1
    assert "data:audio/mp3;base64," in first_markup
    assert "data:audio/mp3;base64," in second_markup


def test_repr_html_uses_cached_wav_asset_when_audio_compression_is_disabled(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "asset")

    markup = plotwave.plot(
        np.linspace(-1, 1, 128),
        sr=16_000,
        compress_audio=False,
        bitrate="fast",
    )._repr_html_()

    assert "plotwave-assets/" in markup
    assert ".wav" in markup
    assert "data:audio/wav;base64," not in markup
    assert any((tmp_path / "plotwave-assets").glob("*.wav"))


def test_repr_html_reuses_cached_wav_asset_when_inline_and_uncompressed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "inline")
    wav = np.sin(np.linspace(0, 8 * np.pi, 1_600))

    first_markup = plotwave.plot(wav, sr=16_000, compress_audio=False)._repr_html_()
    second_markup = plotwave.plot(wav, sr=16_000, compress_audio=False)._repr_html_()

    assets = list((tmp_path / "plotwave-assets").glob("*.wav"))

    assert len(assets) == 1
    assert "data:audio/wav;base64," in first_markup
    assert "data:audio/wav;base64," in second_markup


def test_audio_file_repr_html_uses_cached_mp3_asset_in_notebook(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "asset")
    _write_test_wav(tmp_path / "clip.wav")

    markup = plotwave.plot([plotwave.audio_file("clip.wav")])._repr_html_()

    assert "plotwave-assets/" in markup
    assert ".mp3" in markup
    assert "data:audio/mp3;base64," not in markup
    assert any((tmp_path / "plotwave-assets").glob("*.mp3"))


def test_audio_file_repr_html_uses_inline_cached_mp3_in_vscode(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "inline")
    _write_test_wav(tmp_path / "clip.wav")

    markup = plotwave.plot([plotwave.audio_file("clip.wav")])._repr_html_()

    assert "data:audio/mp3;base64," in markup
    assert any((tmp_path / "plotwave-assets").glob("*.mp3"))


def test_audio_file_repr_html_uses_cached_original_format_asset_when_uncompressed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "asset")
    source_path = _write_test_mp3(tmp_path / "clip.mp3")

    markup = plotwave.plot(
        [plotwave.audio_file("clip.mp3")],
        compress_audio=False,
    )._repr_html_()
    assets = list((tmp_path / "plotwave-assets").glob("*.mp3"))

    assert "plotwave-assets/" in markup
    assert ".mp3" in markup
    assert "data:audio/mp3;base64," not in markup
    assert len(assets) == 1
    assert assets[0].read_bytes() == source_path.read_bytes()


def test_audio_file_repr_html_reuses_cached_original_format_asset_when_inline(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLOTWAVE_NOTEBOOK_AUDIO", "inline")
    source_path = _write_test_mp3(tmp_path / "clip.mp3")

    first_markup = plotwave.plot(
        [plotwave.audio_file("clip.mp3")],
        compress_audio=False,
    )._repr_html_()
    second_markup = plotwave.plot(
        [plotwave.audio_file("clip.mp3")],
        compress_audio=False,
    )._repr_html_()

    assets = list((tmp_path / "plotwave-assets").glob("*.mp3"))

    assert len(assets) == 1
    assert assets[0].read_bytes() == source_path.read_bytes()
    assert "data:audio/mp3;base64," in first_markup
    assert "data:audio/mp3;base64," in second_markup


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

    with pytest.raises(ValueError, match="positive integer or -1"):
        plotwave.plot([0.0, 1.0], points=-2).html()


def test_bitrate_is_ignored_when_audio_compression_is_disabled() -> None:
    html = plotwave.plot(
        np.linspace(-1.0, 1.0, 128),
        sr=16_000,
        compress_audio=False,
        bitrate="fast",
    ).html()

    assert "data:audio/wav;base64," in html


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
