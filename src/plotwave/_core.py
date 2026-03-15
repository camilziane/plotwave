from __future__ import annotations

import base64
import html
import pathlib
import tempfile
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence, cast

import lameenc
import numpy as np
import numpy.typing as npt

from plotwave._render import PreparedTrace, build_html, deep_merge

TraceArgs = dict[str, Any]
ReservedTraceKeys = {"x", "y", "type"}
FloatArray = npt.NDArray[np.float64]
Bitrate = int | float | str | None
DEFAULT_EMBEDDED_MP3_BITRATE_BPS = 128_000
MP3_SAMPLE_RATES = (8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000)
DEFAULT_SEGMENT_COLORS = [
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#d62728",
    "#9467bd",
    "#8c564b",
]


def _validate_trace_kwargs(trace: TraceArgs) -> None:
    invalid = ReservedTraceKeys.intersection(trace)
    if invalid:
        names = ", ".join(sorted(invalid))
        raise ValueError(f"Reserved Plotly keys are managed by plotwave: {names}")


def _as_float_array(values: Sequence[float] | np.ndarray, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        axis = 0 if array.shape[0] <= array.shape[1] else 1
        return cast(FloatArray, np.asarray(array.mean(axis=axis), dtype=np.float64))
    raise ValueError(f"{name} must be 1D or 2D")


def _as_time_array(
    time: Sequence[float] | np.ndarray | None,
    *,
    length: int,
    sr: float | None = None,
) -> FloatArray:
    if time is None:
        if sr is None:
            return np.arange(length, dtype=float)
        return np.arange(length, dtype=float) / sr

    time_array = _as_float_array(time, name="time")
    if len(time_array) != length:
        raise ValueError("time must have the same length as the signal")
    return time_array


def _downsample(
    x: FloatArray,
    y: FloatArray,
    *,
    step: int | None,
    points: int,
) -> tuple[FloatArray, FloatArray]:
    if step is not None:
        if step <= 0:
            raise ValueError("step must be a positive integer")
        x = x[::step]
        y = y[::step]
    if len(x) > points:
        indices = np.linspace(0, len(x) - 1, points, dtype=int)
        x = x[indices]
        y = y[indices]
    return x, y


def _clip_signal(values: FloatArray, clip: float | None) -> FloatArray:
    if clip is None:
        return values
    if not 0.0 <= clip < 0.5:
        raise ValueError("clip must be between 0.0 and 0.5")
    low = float(np.quantile(values, clip))
    high = float(np.quantile(values, 1.0 - clip))
    return cast(FloatArray, np.clip(values, low, high))


def _normalize_signal(values: FloatArray) -> FloatArray:
    amplitude = float(np.max(np.abs(values)))
    if amplitude == 0.0:
        return values
    return values / amplitude


def _normalize_bitrate(bitrate: Bitrate) -> int | None:
    if bitrate is None:
        return None
    if isinstance(bitrate, str):
        normalized = bitrate.strip().lower()
        multiplier = 1
        if normalized.endswith("k"):
            normalized = normalized[:-1]
            multiplier = 1000
        try:
            bitrate_value = float(normalized)
        except ValueError as exc:
            raise ValueError("bitrate must be a positive number or a string like '64k'") from exc
        bitrate_bps = int(round(bitrate_value * multiplier))
    elif isinstance(bitrate, (int, float)):
        bitrate_bps = int(round(float(bitrate)))
    else:
        raise ValueError("bitrate must be a positive number or a string like '64k'")
    if bitrate_bps <= 0:
        raise ValueError("bitrate must be a positive number")
    return bitrate_bps


def _resample_audio(samples: FloatArray, *, source_sr: float, target_sr: float) -> FloatArray:
    if len(samples) <= 1 or np.isclose(source_sr, target_sr):
        return samples

    duration = len(samples) / source_sr
    target_length = max(1, int(round(duration * target_sr)))
    source_time = np.arange(len(samples), dtype=np.float64) / source_sr
    target_time = np.arange(target_length, dtype=np.float64) / target_sr
    resampled = np.interp(target_time, source_time, samples)
    return np.asarray(resampled, dtype=np.float64)


def _select_mp3_sample_rate(*, source_sr: float, bitrate: int | None) -> int:
    target_sr = int(round(source_sr))
    if bitrate is not None:
        target_sr = min(target_sr, max(1, int(round(bitrate / 16))))
    supported = [sample_rate for sample_rate in MP3_SAMPLE_RATES if sample_rate <= target_sr]
    if supported:
        return supported[-1]
    return MP3_SAMPLE_RATES[0]


def _encoded_audio_payload(
    samples: FloatArray,
    *,
    sr: float,
    bitrate: int | None = None,
) -> tuple[str, int, str]:
    encoded_sr = _select_mp3_sample_rate(source_sr=sr, bitrate=bitrate)
    encoded_samples = _resample_audio(samples, source_sr=sr, target_sr=float(encoded_sr))
    target_bitrate = bitrate if bitrate is not None else DEFAULT_EMBEDDED_MP3_BITRATE_BPS
    return (
        _encode_mp3_base64(encoded_samples, float(encoded_sr), bitrate=target_bitrate),
        encoded_sr,
        "mp3",
    )


def _encode_mp3_base64(samples: FloatArray, sr: float, *, bitrate: int) -> str:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype("<i2")
    encoder = lameenc.Encoder()
    encoder.set_channels(1)
    encoder.set_in_sample_rate(int(sr))
    encoder.set_out_sample_rate(int(sr))
    encoder.set_bit_rate(max(8, int(round(bitrate / 1000))))
    encoder.set_quality(2)
    payload = bytes(encoder.encode(pcm16.tobytes()) + encoder.flush())
    return base64.b64encode(payload).decode("ascii")


def _is_notebook() -> bool:
    try:
        import IPython
    except ImportError:
        return False
    get_ipython = cast(Callable[[], object | None] | None, getattr(IPython, "get_ipython", None))
    if get_ipython is None:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return bool(shell.__class__.__name__ == "ZMQInteractiveShell")


def _iframe_html(document: str, *, height: int) -> str:
    escaped = html.escape(document, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" scrolling="no" '
        f'style="border:none;width:100%;height:{height}px;overflow:hidden;display:block;"></iframe>'
    )


def _build_ipython_iframe(document: str, *, height: int) -> Any:
    from IPython.display import IFrame

    escaped = html.escape(document, quote=True)
    extras = [
        f'srcdoc="{escaped}"',
        'style="border:none; width:100%; overflow:hidden;"',
        'scrolling="no"',
    ]
    return IFrame(src="about:blank", width="100%", height=f"{height}px", extras=extras)


def _normalize_layout(layout: dict[str, Any] | None, bounds: dict[str, float]) -> dict[str, Any]:
    base = {
        "margin": {"t": 48, "r": 20, "b": 48, "l": 60},
        "hovermode": "x unified",
        "dragmode": "zoom",
        "clickmode": "event",
        "xaxis": {
            "title": {"text": "Time"},
            "range": [bounds["xmin"], bounds["xmax"]],
        },
        "yaxis": {
            "title": {"text": "Value"},
            "range": [bounds["ymin"], bounds["ymax"]],
        },
        "legend": {"x": 1, "y": 1, "xanchor": "right"},
    }
    return deep_merge(base, layout)


def _normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return deep_merge({"responsive": True}, config)


def _numeric_values(values: list[Any]) -> list[float]:
    numeric: list[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            numeric.append(float(value))
    return numeric


def _trace_bounds(traces: list[PreparedTrace]) -> dict[str, float]:
    x_values = _numeric_values([value for trace in traces for value in trace.plotly_trace["x"]])
    y_values = _numeric_values([value for trace in traces for value in trace.plotly_trace["y"]])
    has_audio = any(trace.audio_info is not None for trace in traces)
    xmin = float(min(x_values)) if x_values else 0.0
    xmax = float(max(x_values)) if x_values else 1.0
    raw_ymin = float(min(y_values)) if y_values else -1.0
    raw_ymax = float(max(y_values)) if y_values else 1.0
    ymin = raw_ymin
    ymax = raw_ymax
    if xmin == xmax:
        xmax = xmin + 1.0
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5
    elif has_audio:
        if ymax > 0.0:
            ymax += 0.5 * abs(ymax)
        if ymin < 0.0:
            ymin -= 0.5 * abs(ymin)
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "raw_ymin": raw_ymin,
        "raw_ymax": raw_ymax,
    }


def _segments_fallback_bounds(overlays: Sequence["SegmentsTrace"]) -> dict[str, float]:
    x_values = [float(value) for overlay in overlays for value in overlay.segment_edges()]
    xmin = float(min(x_values)) if x_values else 0.0
    xmax = float(max(x_values)) if x_values else 1.0
    if xmin == xmax:
        xmax = xmin + 1.0
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": -1.0,
        "ymax": 1.0,
        "raw_ymin": -1.0,
        "raw_ymax": 1.0,
    }


def _color_with_alpha(color: str, alpha: float) -> str:
    if color.startswith("#") and len(color) == 7:
        red = int(color[1:3], 16)
        green = int(color[3:5], 16)
        blue = int(color[5:7], 16)
        return f"rgba({red}, {green}, {blue}, {alpha})"
    if color.startswith("rgb(") and color.endswith(")"):
        values = color[4:-1]
        return f"rgba({values}, {alpha})"
    if color.startswith("rgba("):
        parts = [part.strip() for part in color[5:-1].split(",")]
        if len(parts) == 4:
            return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha})"
    return color


def _apply_scatter_color_defaults(trace: dict[str, Any]) -> dict[str, Any]:
    color = trace.get("color")
    if color is None:
        return trace

    line = trace.get("line")
    if not isinstance(line, dict):
        line = {}
    line.setdefault("color", color)
    trace["line"] = line

    mode = str(trace.get("mode", "lines"))
    if "markers" in mode:
        marker = trace.get("marker")
        if not isinstance(marker, dict):
            marker = {}
        marker.setdefault("color", color)
        trace["marker"] = marker

    return trace


@dataclass(slots=True)
class AudioTrace:
    wav: FloatArray
    sr: float
    time: FloatArray | None = None
    norm: bool = False
    clip: float | None = None
    step: int | None = None
    trace: TraceArgs = field(default_factory=dict)

    def prepared(self, *, points: int, bitrate: int | None = None) -> PreparedTrace:
        display_values = np.array(self.wav, copy=True)
        if self.norm:
            display_values = _normalize_signal(display_values)
        display_values = _clip_signal(display_values, self.clip)

        display_time = _as_time_array(self.time, length=len(display_values), sr=self.sr)
        display_time, display_values = _downsample(
            display_time,
            display_values,
            step=self.step,
            points=points,
        )

        plotly_trace = {
            "type": "scatter",
            "mode": self.trace.get("mode", "lines"),
            "x": display_time.tolist(),
            "y": display_values.tolist(),
        }
        plotly_trace.update(self.trace)
        plotly_trace = _apply_scatter_color_defaults(plotly_trace)

        time_full = _as_time_array(self.time, length=len(self.wav), sr=self.sr)
        b64_data, encoded_sr, audio_format = _encoded_audio_payload(
            self.wav,
            sr=self.sr,
            bitrate=bitrate,
        )
        audio_info = {
            "name": str(self.trace.get("name", "audio")),
            "b64_data": b64_data,
            "start_time": float(time_full[0]),
            "duration": float(len(self.wav) / self.sr),
            "sample_rate": encoded_sr,
            "format": audio_format,
        }
        return PreparedTrace(plotly_trace=plotly_trace, audio_info=audio_info)


@dataclass(slots=True)
class SeriesTrace:
    y: FloatArray
    time: FloatArray | None = None
    sr: float | None = None
    step: int | None = None
    trace: TraceArgs = field(default_factory=dict)

    def prepared(self, *, points: int) -> PreparedTrace:
        if self.time is not None or self.sr is not None:
            time = _as_time_array(self.time, length=len(self.y), sr=self.sr)
        else:
            time = np.arange(len(self.y), dtype=float)
        time, values = _downsample(time, self.y, step=self.step, points=points)
        plotly_trace = {
            "type": "scatter",
            "mode": self.trace.get("mode", "lines"),
            "x": time.tolist(),
            "y": values.tolist(),
        }
        plotly_trace.update(self.trace)
        plotly_trace = _apply_scatter_color_defaults(plotly_trace)
        return PreparedTrace(plotly_trace=plotly_trace)


@dataclass(slots=True)
class SegmentsTrace:
    items: list[dict[str, Any]]
    name: str = "Segment"
    lane: Literal["top", "bottom"] = "top"
    bg_alpha: float = 0.08
    box_alpha: float = 0.92
    textfont: dict[str, Any] = field(default_factory=dict)

    def segment_edges(self) -> list[float]:
        edges: list[float] = []
        for item in self.items:
            edges.extend([float(item["start"]), float(item["end"])])
        return edges

    def prepared(
        self,
        *,
        bounds: dict[str, float],
        reference_x: FloatArray | None = None,
    ) -> tuple[list[PreparedTrace], list[dict[str, Any]], list[dict[str, Any]]]:
        if self.lane == "top":
            band_y0 = 0.5
            band_y1 = 1.0
            box_y0 = 0.90
            box_y1 = 0.985
            label_y = 0.9425
            hover_y = bounds["ymin"] + 0.75 * (bounds["ymax"] - bounds["ymin"])
        else:
            band_y0 = 0.0
            band_y1 = 0.5
            box_y0 = 0.015
            box_y1 = 0.10
            label_y = 0.0575
            hover_y = bounds["ymin"] + 0.25 * (bounds["ymax"] - bounds["ymin"])
        hover_traces: list[PreparedTrace] = []
        shapes: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []

        for item in self.items:
            start = float(item["start"])
            end = float(item["end"])
            label = str(item["label"])
            color = str(item["color"])
            center = (start + end) / 2.0
            hover_x_array: FloatArray
            if reference_x is not None:
                segment_x = reference_x[(reference_x >= start) & (reference_x <= end)]
                if len(segment_x) == 0:
                    hover_x_array = np.linspace(start, end, 128, dtype=float)
                else:
                    hover_x_array = segment_x
            else:
                hover_x_array = np.linspace(start, end, 128, dtype=float)
            hover_x = hover_x_array.tolist()
            hover_y_values = [hover_y] * len(hover_x)

            hover_traces.append(
                PreparedTrace(
                    plotly_trace={
                        "type": "scatter",
                        "mode": "lines",
                        "x": hover_x,
                        "y": hover_y_values,
                        "name": self.name,
                        "showlegend": False,
                        "hovertemplate": f"{self.name}: {label}<extra></extra>",
                        "line": {"color": "rgba(0,0,0,0)", "width": 18},
                    }
                )
            )

            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": start,
                    "x1": end,
                    "y0": band_y0,
                    "y1": band_y1,
                    "fillcolor": _color_with_alpha(color, self.bg_alpha),
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": start,
                    "x1": end,
                    "y0": box_y0,
                    "y1": box_y1,
                    "fillcolor": _color_with_alpha(color, self.box_alpha),
                    "line": {"width": 0},
                    "layer": "above",
                }
            )
            annotations.append(
                {
                    "xref": "x",
                    "yref": "paper",
                    "x": center,
                    "y": label_y,
                    "text": f"<b>{label}</b>",
                    "showarrow": False,
                    "font": {"size": 12, "color": "white", **self.textfont},
                    "xanchor": "center",
                    "yanchor": "middle",
                }
            )
        return hover_traces, shapes, annotations


PlotItem = AudioTrace | SeriesTrace | SegmentsTrace


@dataclass(slots=True)
class Plot:
    data: list[PlotItem]
    layout: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    points: int = 3000
    display: str = "auto"
    bitrate: Bitrate = None

    def _prepared(self) -> tuple[list[PreparedTrace], list[SegmentsTrace]]:
        if self.points <= 0:
            raise ValueError("points must be a positive integer")
        normalized_bitrate = _normalize_bitrate(self.bitrate)
        prepared: list[PreparedTrace] = []
        overlays: list[SegmentsTrace] = []
        audio_items: list[AudioTrace] = []
        pending_series: list[SeriesTrace] = []
        for item in self.data:
            if isinstance(item, SegmentsTrace):
                overlays.append(item)
            elif isinstance(item, AudioTrace):
                audio_items.append(item)
                prepared.append(item.prepared(points=self.points, bitrate=normalized_bitrate))
            else:
                pending_series.append(item)

        if audio_items and any(item.time is None and item.sr is None for item in pending_series):
            raise ValueError(
                "SeriesTrace plotted with audio requires explicit time or sr."
            )

        for item in pending_series:
            prepared.append(item.prepared(points=self.points))
        if not prepared and not overlays:
            raise ValueError("plot requires at least one trace")
        return prepared, overlays

    def _resolved(self) -> tuple[list[PreparedTrace], dict[str, Any], dict[str, Any]]:
        prepared, overlays = self._prepared()
        base_bounds = _trace_bounds(prepared) if prepared else _segments_fallback_bounds(overlays)
        reference_x_values = _numeric_values(
            [value for trace in prepared for value in trace.plotly_trace["x"]]
        )
        reference_x = (
            np.asarray(sorted(set(reference_x_values)), dtype=np.float64)
            if reference_x_values
            else None
        )
        overlay_shapes: list[dict[str, Any]] = []
        overlay_annotations: list[dict[str, Any]] = []
        overlay_traces: list[PreparedTrace] = []
        for overlay in overlays:
            traces, shapes, annotations = overlay.prepared(
                bounds=base_bounds,
                reference_x=reference_x,
            )
            overlay_traces.extend(traces)
            overlay_shapes.extend(shapes)
            overlay_annotations.extend(annotations)

        all_traces = prepared + overlay_traces
        bounds = _trace_bounds(all_traces) if all_traces else base_bounds
        layout = _normalize_layout(self.layout, bounds)
        if overlay_shapes:
            layout["shapes"] = overlay_shapes + list(layout.get("shapes", []))
        if overlay_annotations:
            layout["annotations"] = overlay_annotations + list(layout.get("annotations", []))
        config = _normalize_config(self.config)
        return all_traces, layout, config

    def _document(self) -> str:
        prepared, layout, config = self._resolved()
        return build_html(prepared, layout, config, frame_height=self._frame_height(layout))

    def html(self) -> str:
        return self._document()

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        output = pathlib.Path(path)
        output.write_text(self._document(), encoding="utf-8")
        return output

    def show(self) -> "Plot":
        if self.display == "inline" or (self.display == "auto" and _is_notebook()):
            try:
                from IPython.display import display
            except ImportError as exc:
                raise RuntimeError("Inline display requires IPython") from exc
            display(self._ipython_iframe())  # type: ignore[no-untyped-call]
            return self

        if self.display == "none":
            return self

        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(self._document())
            temp_path = pathlib.Path(tmp.name)
        webbrowser.open(temp_path.as_uri())
        return self

    def _repr_html_(self) -> str:
        iframe = self._ipython_iframe()
        repr_html = getattr(iframe, "_repr_html_", None)
        if callable(repr_html):
            return cast(str, repr_html())
        return _iframe_html(self._document(), height=self._frame_height())

    def _frame_height(self, layout: dict[str, Any] | None = None) -> int:
        effective_layout = layout
        if effective_layout is None:
            _, effective_layout, _ = self._resolved()
        plot_height = int(effective_layout.get("height", 600))
        return plot_height + 45

    def _ipython_iframe(self) -> Any:
        return _build_ipython_iframe(self._document(), height=self._frame_height())


def audio(
    wav: Sequence[float] | np.ndarray,
    sr: float,
    *,
    time: Sequence[float] | np.ndarray | None = None,
    norm: bool = False,
    clip: float | None = None,
    step: int | None = None,
    **trace: Any,
) -> AudioTrace:
    if sr <= 0:
        raise ValueError("sr must be a positive number")
    _validate_trace_kwargs(trace)
    wav_array = _as_float_array(wav, name="wav")
    time_array = None if time is None else _as_time_array(time, length=len(wav_array))
    return AudioTrace(
        wav=wav_array,
        sr=float(sr),
        time=time_array,
        norm=norm,
        clip=clip,
        step=step,
        trace=dict(trace),
    )


def series(
    y: Sequence[float] | np.ndarray,
    *,
    time: Sequence[float] | np.ndarray | None = None,
    sr: float | None = None,
    step: int | None = None,
    **trace: Any,
) -> SeriesTrace:
    _validate_trace_kwargs(trace)
    if time is not None and sr is not None:
        raise ValueError("series accepts either time or sr, not both")
    if sr is not None and sr <= 0:
        raise ValueError("sr must be a positive number")
    values = _as_float_array(y, name="y")
    time_array = None if time is None else _as_time_array(time, length=len(values))
    return SeriesTrace(
        y=values,
        time=time_array,
        sr=None if sr is None else float(sr),
        step=step,
        trace=dict(trace),
    )


def _coerce_data(
    data: Sequence[float] | np.ndarray | PlotItem | Sequence[PlotItem],
    *,
    sr: float | None,
    time: Sequence[float] | np.ndarray | None,
    trace: TraceArgs,
) -> list[PlotItem]:
    if isinstance(data, (AudioTrace, SeriesTrace, SegmentsTrace)):
        if sr is not None or time is not None or trace:
            raise ValueError(
                "sr, time, and trace kwargs are only supported for raw single-trace data"
            )
        return [data]
    if isinstance(data, np.ndarray):
        if sr is not None:
            return [audio(data, sr, time=time, **trace)]
        return [series(data, time=time, **trace)]
    if (
        isinstance(data, Sequence)
        and data
        and all(isinstance(item, (AudioTrace, SeriesTrace, SegmentsTrace)) for item in data)
    ):
        if sr is not None or time is not None or trace:
            raise ValueError(
                "sr, time, and trace kwargs are only supported for raw single-trace data"
            )
        return list(cast(Sequence[PlotItem], data))
    if sr is not None:
        return [audio(data, sr, time=time, **trace)]  # type: ignore[arg-type]
    return [series(data, time=time, **trace)]  # type: ignore[arg-type]


def plot(
    data: Sequence[float] | np.ndarray | PlotItem | Sequence[PlotItem],
    *,
    sr: float | None = None,
    time: Sequence[float] | np.ndarray | None = None,
    layout: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    points: int = 3000,
    display: str = "auto",
    bitrate: Bitrate = None,
    **trace: Any,
) -> Plot:
    if display not in {"auto", "inline", "browser", "none"}:
        raise ValueError("display must be one of: auto, inline, browser, none")
    _normalize_bitrate(bitrate)
    items = _coerce_data(data, sr=sr, time=time, trace=dict(trace))
    return Plot(
        data=items,
        layout=layout,
        config=config,
        points=points,
        display=display,
        bitrate=bitrate,
    )


def audio_trace_plot(
    data: Sequence[dict[str, Any]],
    title: str | None = "Interactive Audio/Data Plot",
    max_points_display: int = 3000,
    output_mode: str = "file",
    output_path: str = "interactive_plot.html",
    iframe_height: str = "600px",
    audio_format: str = "mp3",
    audio_bitrate: str = "16k",
) -> str | Any | None:
    del audio_format
    if not data:
        return None

    items: list[PlotItem] = []
    for trace_spec in data:
        trace_kwargs: dict[str, Any] = {}
        for key in ("name", "color", "line", "fill", "opacity", "mode"):
            if key in trace_spec:
                trace_kwargs[key] = trace_spec[key]
        item_type = trace_spec.get("type", "numpy")
        remove_outliers = trace_spec.get("remove_outliers", False)
        step = trace_spec.get("decimate_by")
        if item_type == "audio":
            items.append(
                audio(
                    trace_spec["y"],
                    trace_spec["sr"],
                    time=trace_spec.get("x"),
                    norm=bool(trace_spec.get("minmax_normalization", False)),
                    clip=0.01 if remove_outliers else None,
                    step=step,
                    **trace_kwargs,
                )
            )
        else:
            items.append(
                series(
                    trace_spec["y"],
                    time=trace_spec.get("x"),
                    sr=trace_spec.get("sr"),
                    step=step,
                    **trace_kwargs,
                )
            )

    height = int(str(iframe_height).removesuffix("px"))
    plot_obj = plot(
        items,
        layout={"title": {"text": title or ""}, "height": height},
        points=max_points_display,
        bitrate=audio_bitrate,
    )

    if output_mode == "file":
        return str(plot_obj.save(output_path).resolve())
    if output_mode == "html_string":
        return plot_obj.html()
    if output_mode == "jupyter":
        try:
            return _build_ipython_iframe(plot_obj.html(), height=height)
        except ImportError as exc:
            raise RuntimeError("Jupyter output requires IPython") from exc
    return None


def segments(
    items: Sequence[dict[str, Any]],
    *,
    name: str = "Segment",
    lane: Literal["top", "bottom"] = "top",
    color_map: dict[str, str] | None = None,
    bg_alpha: float = 0.08,
    box_alpha: float = 0.92,
    textfont: dict[str, Any] | None = None,
) -> SegmentsTrace:
    if not items:
        raise ValueError("segments requires at least one segment")

    normalized: list[dict[str, Any]] = []
    resolved_color_map = dict(color_map or {})
    for index, item in enumerate(items):
        if "start" not in item or "end" not in item:
            raise ValueError("each segment requires 'start' and 'end'")
        start = float(item["start"])
        end = float(item["end"])
        if end <= start:
            raise ValueError("segment 'end' must be greater than 'start'")
        label = str(item.get("label", item.get("name", f"Segment {index + 1}")))
        if "color" in item:
            color = str(item["color"])
        elif label in resolved_color_map:
            color = str(resolved_color_map[label])
        else:
            color = DEFAULT_SEGMENT_COLORS[index % len(DEFAULT_SEGMENT_COLORS)]
        normalized.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "color": color,
            }
        )

    if not 0.0 <= bg_alpha <= 1.0:
        raise ValueError("bg_alpha must be between 0.0 and 1.0")
    if not 0.0 <= box_alpha <= 1.0:
        raise ValueError("box_alpha must be between 0.0 and 1.0")
    if lane not in {"top", "bottom"}:
        raise ValueError("lane must be 'top' or 'bottom'")

    return SegmentsTrace(
        items=normalized,
        name=name,
        lane=lane,
        bg_alpha=bg_alpha,
        box_alpha=box_alpha,
        textfont=dict(textfont or {}),
    )
