from __future__ import annotations

import base64
import hashlib
import html
import io
import os
import pathlib
import tempfile
import wave
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence, cast

import lameenc  # type: ignore[import-not-found]
import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]

from plotwave._render import PreparedTrace, build_html, deep_merge

TraceArgs = dict[str, Any]
ReservedTraceKeys = {"x", "y", "type"}
FloatArray = npt.NDArray[np.float64]
Bitrate = int | float | str | None
DEFAULT_EMBEDDED_MP3_BITRATE_BPS = 128_000
NOTEBOOK_ASSET_DIR = "plotwave-assets"
NOTEBOOK_AUDIO_MODE_ENV_VAR = "PLOTWAVE_NOTEBOOK_AUDIO"
MP3_SAMPLE_RATES = (8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000)
ALL_POINTS = -1
DEFAULT_BASE_POINTS = 3_000
DEFAULT_POINTS_PER_SECOND = 16
DEFAULT_SEGMENT_COLORS = [
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#d62728",
    "#9467bd",
    "#8c564b",
]
_AUDIO_FORMAT_ALIASES = {
    "aif": "aiff",
    "mpeg": "mp3",
    "mpeg3": "mp3",
    "oga": "ogg",
    "wave": "wav",
}
_BITRATE_UNSET = object()


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
    if points == ALL_POINTS:
        return x, y
    if points <= 0:
        raise ValueError("points must be a positive integer or -1")
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


def _prepare_audio_display_values(
    values: FloatArray,
    *,
    norm: bool,
    clip: float | None,
) -> FloatArray:
    display_values = _clip_signal(values, clip)
    if norm:
        display_values = _normalize_signal(display_values)
    return display_values


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

    ratio = source_sr / target_sr
    rounded_ratio = round(ratio)
    if rounded_ratio >= 1 and np.isclose(ratio, rounded_ratio):
        return np.asarray(samples[::rounded_ratio], dtype=np.float64)

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
    payload, encoded_sr, audio_format = _encoded_audio_bytes(samples, sr=sr, bitrate=bitrate)
    return (
        base64.b64encode(payload).decode("ascii"),
        encoded_sr,
        audio_format,
    )


def _encoded_audio_bytes(
    samples: FloatArray,
    *,
    sr: float,
    bitrate: int | None = None,
) -> tuple[bytes, int, str]:
    encoded_sr = _select_mp3_sample_rate(source_sr=sr, bitrate=bitrate)
    encoded_samples = _resample_audio(samples, source_sr=sr, target_sr=float(encoded_sr))
    target_bitrate = bitrate if bitrate is not None else DEFAULT_EMBEDDED_MP3_BITRATE_BPS
    return (
        _encode_mp3_bytes(encoded_samples, float(encoded_sr), bitrate=target_bitrate),
        encoded_sr,
        "mp3",
    )


def _encode_mp3_bytes(samples: FloatArray, sr: float, *, bitrate: int) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype("<i2")
    encoder = lameenc.Encoder()
    encoder.set_channels(1)
    encoder.set_in_sample_rate(int(sr))
    encoder.set_out_sample_rate(int(sr))
    encoder.set_bit_rate(max(8, int(round(bitrate / 1000))))
    encoder.set_quality(2)
    return bytes(encoder.encode(pcm16.tobytes()) + encoder.flush())


def _encode_mp3_base64(samples: FloatArray, sr: float, *, bitrate: int) -> str:
    payload = _encode_mp3_bytes(samples, sr, bitrate=bitrate)
    return base64.b64encode(payload).decode("ascii")


def _audio_format_from_suffix(suffix: str) -> str:
    normalized = suffix.strip().lower().lstrip(".")
    if not normalized:
        return "wav"
    return _AUDIO_FORMAT_ALIASES.get(normalized, normalized)


def _audio_format_from_path(path: pathlib.Path, *, fallback: str | None = None) -> str:
    if path.suffix:
        return _audio_format_from_suffix(path.suffix)
    if fallback:
        return _audio_format_from_suffix(fallback)
    return "wav"


def _array_audio_asset_hash(
    samples: FloatArray,
    *,
    source_sr: float,
    target_format: str,
    target_sr: int | None = None,
    bitrate: int | None = None,
) -> str:
    contiguous = np.ascontiguousarray(samples)
    digest = hashlib.sha256()
    digest.update(b"plotwave-notebook-array-audio-v2")
    digest.update(
        f"{source_sr:.12g}:{target_format}:{target_sr or 0}:{bitrate or 0}".encode("ascii")
    )
    digest.update(contiguous.view(np.uint8).tobytes())
    return digest.hexdigest()


def _audio_src_from_path(path: pathlib.Path) -> str:
    normalized = path.expanduser()
    if not normalized.is_absolute():
        return normalized.as_posix()
    try:
        return normalized.resolve().relative_to(pathlib.Path.cwd().resolve()).as_posix()
    except ValueError:
        return normalized.resolve().as_uri()


def _embedded_audio_info(
    base_info: dict[str, Any],
    *,
    payload: bytes,
    audio_format: str,
    sample_rate: int | None,
) -> dict[str, Any]:
    info = {
        **base_info,
        "b64_data": base64.b64encode(payload).decode("ascii"),
        "format": audio_format,
    }
    if sample_rate is not None:
        info["sample_rate"] = sample_rate
    return info


def _asset_audio_info(
    base_info: dict[str, Any],
    *,
    asset_path: pathlib.Path,
    audio_format: str,
    sample_rate: int | None,
    inline: bool,
) -> dict[str, Any]:
    if inline:
        return _embedded_audio_info(
            base_info,
            payload=asset_path.read_bytes(),
            audio_format=audio_format,
            sample_rate=sample_rate,
        )
    info = {
        **base_info,
        "src": _audio_src_from_path(asset_path),
        "format": audio_format,
    }
    if sample_rate is not None:
        info["sample_rate"] = sample_rate
    return info


def _file_audio_asset_hash(
    path: pathlib.Path,
    *,
    target_format: str,
    source_sr: float | None = None,
    target_sr: int | None = None,
    bitrate: int | None = None,
) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    digest = hashlib.sha256()
    digest.update(b"plotwave-notebook-file-audio-v2")
    digest.update(str(resolved).encode("utf-8"))
    digest.update(
        (
            f"{stat.st_size}:{stat.st_mtime_ns}:{source_sr or 0:.12g}:"
            f"{target_format}:{target_sr or 0}:{bitrate or 0}"
        ).encode("ascii")
    )
    return digest.hexdigest()

def _ensure_audio_asset(path: pathlib.Path, builder: Callable[[], bytes]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(builder())
    return path


def _raise_soundfile_error(
    path: pathlib.Path,
    *,
    action: str,
    exc: Exception,
) -> None:
    raise RuntimeError(
        f"plotwave.audio_file could not {action} {path}: "
        "soundfile could not decode this audio file with the libsndfile backend on this system"
    ) from exc


def _read_soundfile_samples(path: pathlib.Path) -> FloatArray:
    try:
        with sf.SoundFile(str(path)) as audio_file:
            payload = audio_file.read(dtype="float64", always_2d=True)
    except (OSError, RuntimeError, ValueError) as exc:
        _raise_soundfile_error(path, action="read", exc=exc)
    if payload.size == 0:
        raise ValueError("audio file cannot be empty")
    return np.asarray(payload.mean(axis=1), dtype=np.float64)


def _decoded_soundfile_frame_count(path: pathlib.Path) -> int:
    try:
        with sf.SoundFile(str(path)) as audio_file:
            total_frames = 0
            while True:
                payload = audio_file.read(65_536, dtype="float64", always_2d=True)
                if payload.size == 0:
                    break
                total_frames += int(payload.shape[0])
    except (OSError, RuntimeError, ValueError) as exc:
        _raise_soundfile_error(path, action="count frames for", exc=exc)
    if total_frames <= 0:
        raise ValueError("audio file cannot be empty")
    return total_frames


def _encode_wav_bytes(samples: FloatArray, *, sr: float) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype("<i2")
    sample_rate = max(1, int(round(sr)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _soundfile_display_trace(
    path: pathlib.Path,
    *,
    frames: int,
    sr: float,
    points: int,
    step: int | None,
) -> tuple[FloatArray, FloatArray]:
    effective_step = 1 if step is None else step
    if effective_step <= 0:
        raise ValueError("step must be a positive integer")
    virtual_length = max(1, (frames + effective_step - 1) // effective_step)
    sample_count = min(points, virtual_length)
    sample_positions = np.linspace(0, virtual_length - 1, sample_count, dtype=int) * effective_step
    sample_positions = np.clip(sample_positions, 0, frames - 1)
    display_time = sample_positions.astype(np.float64) / sr
    display_values = np.empty(sample_count, dtype=np.float64)
    try:
        with sf.SoundFile(str(path)) as audio_file:
            for index, position in enumerate(sample_positions):
                audio_file.seek(int(position))
                frame = audio_file.read(1, dtype="float64", always_2d=True)
                display_values[index] = 0.0 if frame.size == 0 else float(frame.mean())
    except (OSError, RuntimeError, ValueError) as exc:
        _raise_soundfile_error(path, action="sample", exc=exc)
    return display_time, display_values


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


def _is_vscode_environment() -> bool:
    if os.environ.get("TERM_PROGRAM", "").strip().lower() == "vscode":
        return True
    return any(
        os.environ.get(name)
        for name in ("VSCODE_PID", "VSCODE_CWD", "VSCODE_IPC_HOOK_CLI")
    )


def _notebook_audio_mode() -> Literal["asset", "inline"]:
    normalized = os.environ.get(NOTEBOOK_AUDIO_MODE_ENV_VAR, "").strip().lower()
    if normalized in {"inline", "embed", "embedded", "data"}:
        return "inline"
    if normalized in {"asset", "assets", "file", "files"}:
        return "asset"
    if _is_vscode_environment():
        return "inline"
    return "asset"


def _iframe_html(document: str, *, height: int) -> str:
    escaped = html.escape(document, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" scrolling="no" '
        f'style="border:none;width:100%;height:{height}px;overflow:hidden;display:block;"></iframe>'
    )


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


def _time_span_from_trace(item: PlotItem) -> tuple[float, float] | None:
    if isinstance(item, AudioTrace):
        if item.time is not None:
            return float(item.time[0]), float(item.time[-1])
        if len(item.wav) <= 1:
            return 0.0, 0.0
        return 0.0, float((len(item.wav) - 1) / item.sr)
    if isinstance(item, AudioFileTrace):
        if item.frames <= 1:
            return 0.0, 0.0
        return 0.0, float((item.frames - 1) / item.sr)
    if isinstance(item, SeriesTrace):
        if item.time is not None:
            return float(item.time[0]), float(item.time[-1])
        if item.sr is not None:
            if len(item.y) <= 1:
                return 0.0, 0.0
            return 0.0, float((len(item.y) - 1) / item.sr)
        return None
    if isinstance(item, SegmentsTrace):
        edges = item.segment_edges()
        if not edges:
            return None
        return min(edges), max(edges)
    return None


def _trace_bounds(traces: list[PreparedTrace]) -> dict[str, float]:
    x_values = _numeric_values([value for trace in traces for value in trace.plotly_trace["x"]])
    y_values = _numeric_values([value for trace in traces for value in trace.plotly_trace["y"]])
    audio_x_values: list[float] = []
    for trace in traces:
        if trace.audio_info is None:
            continue
        start_time = trace.audio_info.get("start_time")
        duration = trace.audio_info.get("duration")
        if not isinstance(start_time, (int, float, np.integer, np.floating)):
            continue
        if not isinstance(duration, (int, float, np.integer, np.floating)):
            continue
        start_value = float(start_time)
        duration_value = float(duration)
        if not np.isfinite(start_value) or not np.isfinite(duration_value):
            continue
        audio_x_values.extend([start_value, start_value + duration_value])
    has_audio = any(trace.audio_info is not None for trace in traces)
    all_x_values = x_values + audio_x_values
    xmin = float(min(all_x_values)) if all_x_values else 0.0
    xmax = float(max(all_x_values)) if all_x_values else 1.0
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
    """In-memory audio trace.

    Parameters
    ----------
    wav:
        Audio samples. A 2D input is averaged to mono for display and playback.
    sr:
        Sample rate in Hz.
    time:
        Optional explicit time axis for the samples. When omitted, time is
        inferred from `sr`.
    norm:
        Whether to normalize the displayed waveform to unit peak amplitude.
    clip:
        Optional symmetric quantile clipping applied only to the displayed waveform.
    step:
        Optional decimation factor applied before display downsampling.
    trace:
        Extra Plotly scatter arguments, such as `name`, `color`, `line`, or `opacity`.
    """
    wav: FloatArray
    sr: float
    time: FloatArray | None = None
    norm: bool = False
    clip: float | None = None
    step: int | None = None
    trace: TraceArgs = field(default_factory=dict)

    def _display_trace(self, *, points: int) -> dict[str, Any]:
        display_values = np.array(self.wav, copy=True)
        display_values = _prepare_audio_display_values(
            display_values,
            norm=self.norm,
            clip=self.clip,
        )

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
        return _apply_scatter_color_defaults(plotly_trace)

    def _audio_info(self) -> dict[str, Any]:
        time_full = _as_time_array(self.time, length=len(self.wav), sr=self.sr)
        return {
            "name": str(self.trace.get("name", "audio")),
            "start_time": float(time_full[0]),
            "duration": float(len(self.wav) / self.sr),
        }

    def plot(
        self,
        *,
        layout: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        points: int | None = None,
        display: str = "auto",
        compress_audio: bool = True,
        bitrate: Bitrate = None,
    ) -> "Plot":
        """Build a :class:`Plot` from this audio trace.

        Parameters
        ----------
        layout:
            Optional Plotly layout overrides.
        config:
            Optional Plotly config overrides.
        points:
            Maximum number of displayed points after downsampling. When omitted,
            it defaults to `max(3000, duration_seconds * 16)` for timed plots and
            all points for plots without a time-based notion. Use `-1` to force
            full-resolution display.
        display:
            Display mode used by :meth:`Plot.show`.
        compress_audio:
            Whether embedded or cached audio should be MP3-compressed.
        bitrate:
            Target bitrate for compressed audio, for example `"32k"` or `64000`.
            Ignored when `compress_audio` is `False`.
        """
        return plot(
            self,
            layout=layout,
            config=config,
            points=points,
            display=display,
            compress_audio=compress_audio,
            bitrate=bitrate,
        )

    def _embedded_audio(
        self,
        *,
        bitrate: int | None,
        compress_audio: bool,
    ) -> dict[str, Any]:
        if compress_audio:
            payload, encoded_sr, audio_format = _encoded_audio_bytes(
                self.wav,
                sr=self.sr,
                bitrate=bitrate,
            )
            return _embedded_audio_info(
                self._audio_info(),
                payload=payload,
                audio_format=audio_format,
                sample_rate=encoded_sr,
            )
        return _embedded_audio_info(
            self._audio_info(),
            payload=_encode_wav_bytes(self.wav, sr=self.sr),
            audio_format="wav",
            sample_rate=int(round(self.sr)),
        )

    def _notebook_asset(
        self,
        *,
        asset_dir: pathlib.Path,
        bitrate: int | None,
        compress_audio: bool,
    ) -> tuple[pathlib.Path, int | None, str]:
        if compress_audio:
            target_bitrate = bitrate if bitrate is not None else DEFAULT_EMBEDDED_MP3_BITRATE_BPS
            encoded_sr = _select_mp3_sample_rate(source_sr=self.sr, bitrate=target_bitrate)
            asset_key = _array_audio_asset_hash(
                self.wav,
                source_sr=self.sr,
                target_format="mp3",
                target_sr=encoded_sr,
                bitrate=target_bitrate,
            )
            asset_path = asset_dir / f"{asset_key}.mp3"
            _ensure_audio_asset(
                asset_path,
                lambda: _encoded_audio_bytes(
                    self.wav,
                    sr=self.sr,
                    bitrate=target_bitrate,
                )[0],
            )
            return asset_path, encoded_sr, "mp3"

        asset_key = _array_audio_asset_hash(
            self.wav,
            source_sr=self.sr,
            target_format="wav",
            target_sr=int(round(self.sr)),
        )
        asset_path = asset_dir / f"{asset_key}.wav"
        _ensure_audio_asset(asset_path, lambda: _encode_wav_bytes(self.wav, sr=self.sr))
        return asset_path, int(round(self.sr)), "wav"

    def prepared(
        self,
        *,
        points: int,
        bitrate: int | None = None,
        compress_audio: bool = True,
        embed_audio: bool = True,
    ) -> PreparedTrace:
        plotly_trace = self._display_trace(points=points)
        audio_info = self._audio_info()
        if embed_audio:
            audio_info = self._embedded_audio(
                bitrate=bitrate,
                compress_audio=compress_audio,
            )
        return PreparedTrace(plotly_trace=plotly_trace, audio_info=audio_info)

    def notebook_audio_asset(
        self,
        *,
        asset_dir: pathlib.Path,
        bitrate: int | None = None,
        compress_audio: bool = True,
        inline: bool = False,
    ) -> dict[str, Any]:
        asset_path, sample_rate, audio_format = self._notebook_asset(
            asset_dir=asset_dir,
            bitrate=bitrate,
            compress_audio=compress_audio,
        )
        return _asset_audio_info(
            self._audio_info(),
            asset_path=asset_path,
            audio_format=audio_format,
            sample_rate=sample_rate,
            inline=inline,
        )


@dataclass(slots=True)
class AudioFileTrace:
    """File-backed audio trace.

    Parameters
    ----------
    path:
        Path to the source audio file.
    sr:
        Sample rate in Hz reported by the audio file metadata.
    frames:
        Number of audio frames in the file.
    audio_format:
        File format used for uncompressed embedding or notebook caching.
    norm:
        Whether to normalize the displayed waveform to unit peak amplitude.
    clip:
        Optional symmetric quantile clipping applied only to the displayed waveform.
    step:
        Optional decimation factor applied before display downsampling.
    trace:
        Extra Plotly scatter arguments, such as `name`, `color`, `line`, or `opacity`.
    """
    path: pathlib.Path
    sr: float
    frames: int
    audio_format: str
    norm: bool = False
    clip: float | None = None
    step: int | None = None
    trace: TraceArgs = field(default_factory=dict)

    def _display_trace(self, *, points: int) -> dict[str, Any]:
        display_time, display_values = _soundfile_display_trace(
            self.path,
            frames=self.frames,
            sr=self.sr,
            points=points,
            step=self.step,
        )
        display_values = _prepare_audio_display_values(
            display_values,
            norm=self.norm,
            clip=self.clip,
        )

        plotly_trace = {
            "type": "scatter",
            "mode": self.trace.get("mode", "lines"),
            "x": display_time.tolist(),
            "y": display_values.tolist(),
        }
        plotly_trace.update(self.trace)
        return _apply_scatter_color_defaults(plotly_trace)

    def _audio_info(self) -> dict[str, Any]:
        return {
            "name": str(self.trace.get("name", self.path.stem or "audio")),
            "start_time": 0.0,
            "duration": float(self.frames / self.sr),
        }

    def plot(
        self,
        *,
        layout: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        points: int | None = None,
        display: str = "auto",
        compress_audio: bool = True,
        bitrate: Bitrate = None,
    ) -> "Plot":
        """Build a :class:`Plot` from this file-backed audio trace.

        Parameters
        ----------
        layout:
            Optional Plotly layout overrides.
        config:
            Optional Plotly config overrides.
        points:
            Maximum number of displayed points after downsampling. When omitted,
            it defaults to `max(3000, duration_seconds * 16)` for timed plots and
            all points for plots without a time-based notion. Use `-1` to force
            full-resolution display.
        display:
            Display mode used by :meth:`Plot.show`.
        compress_audio:
            Whether embedded or cached audio should be MP3-compressed.
        bitrate:
            Target bitrate for compressed audio, for example `"32k"` or `64000`.
            Ignored when `compress_audio` is `False`.
        """
        return plot(
            self,
            layout=layout,
            config=config,
            points=points,
            display=display,
            compress_audio=compress_audio,
            bitrate=bitrate,
        )

    def _decoded_samples(self) -> FloatArray:
        return _read_soundfile_samples(self.path)

    def _embedded_audio(
        self,
        *,
        bitrate: int | None,
        compress_audio: bool,
    ) -> dict[str, Any]:
        if compress_audio:
            payload, encoded_sr, audio_format = _encoded_audio_bytes(
                self._decoded_samples(),
                sr=self.sr,
                bitrate=bitrate,
            )
            return _embedded_audio_info(
                self._audio_info(),
                payload=payload,
                audio_format=audio_format,
                sample_rate=encoded_sr,
            )
        return _embedded_audio_info(
            self._audio_info(),
            payload=self.path.read_bytes(),
            audio_format=self.audio_format,
            sample_rate=int(round(self.sr)),
        )

    def _notebook_asset(
        self,
        *,
        asset_dir: pathlib.Path,
        bitrate: int | None = None,
        compress_audio: bool = True,
    ) -> tuple[pathlib.Path, int | None, str]:
        if compress_audio:
            target_bitrate = bitrate if bitrate is not None else DEFAULT_EMBEDDED_MP3_BITRATE_BPS
            encoded_sr = _select_mp3_sample_rate(source_sr=self.sr, bitrate=target_bitrate)
            asset_key = _file_audio_asset_hash(
                self.path,
                target_format="mp3",
                source_sr=self.sr,
                target_sr=encoded_sr,
                bitrate=target_bitrate,
            )
            asset_path = asset_dir / f"{asset_key}.mp3"
            _ensure_audio_asset(
                asset_path,
                lambda: _encoded_audio_bytes(
                    self._decoded_samples(),
                    sr=self.sr,
                    bitrate=target_bitrate,
                )[0],
            )
            return asset_path, encoded_sr, "mp3"

        asset_key = _file_audio_asset_hash(
            self.path,
            target_format=self.audio_format,
            source_sr=self.sr,
        )
        asset_path = asset_dir / f"{asset_key}.{self.audio_format}"
        _ensure_audio_asset(asset_path, self.path.read_bytes)
        return asset_path, int(round(self.sr)), self.audio_format

    def prepared(
        self,
        *,
        points: int,
        bitrate: int | None = None,
        compress_audio: bool = True,
        embed_audio: bool = True,
    ) -> PreparedTrace:
        plotly_trace = self._display_trace(points=points)
        audio_info = self._audio_info()
        if embed_audio:
            audio_info = self._embedded_audio(
                bitrate=bitrate,
                compress_audio=compress_audio,
            )
        return PreparedTrace(plotly_trace=plotly_trace, audio_info=audio_info)

    def notebook_audio_asset(
        self,
        *,
        asset_dir: pathlib.Path,
        bitrate: int | None = None,
        compress_audio: bool = True,
        inline: bool = False,
    ) -> dict[str, Any]:
        asset_path, sample_rate, audio_format = self._notebook_asset(
            asset_dir=asset_dir,
            bitrate=bitrate,
            compress_audio=compress_audio,
        )
        return _asset_audio_info(
            self._audio_info(),
            asset_path=asset_path,
            audio_format=audio_format,
            sample_rate=sample_rate,
            inline=inline,
        )


@dataclass(slots=True)
class SeriesTrace:
    """Non-audio numeric series trace.

    Parameters
    ----------
    y:
        Series values to plot.
    time:
        Optional explicit x-axis values.
    sr:
        Optional sample rate used to derive the x-axis when `time` is omitted.
    step:
        Optional decimation factor applied before display downsampling.
    trace:
        Extra Plotly scatter arguments, such as `name`, `color`, `line`, or `opacity`.
    """
    y: FloatArray
    time: FloatArray | None = None
    sr: float | None = None
    step: int | None = None
    trace: TraceArgs = field(default_factory=dict)

    def plot(
        self,
        *,
        layout: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        points: int | None = None,
        display: str = "auto",
        compress_audio: bool = True,
        bitrate: Bitrate = None,
    ) -> "Plot":
        """Build a :class:`Plot` from this series trace.

        Parameters
        ----------
        layout:
            Optional Plotly layout overrides.
        config:
            Optional Plotly config overrides.
        points:
            Maximum number of displayed points after downsampling. When omitted,
            it defaults to `max(3000, duration_seconds * 16)` for timed plots and
            all points for plots without a time-based notion. Use `-1` to force
            full-resolution display.
        display:
            Display mode used by :meth:`Plot.show`.
        compress_audio:
            Default audio compression policy for audio traces added to the same plot.
        bitrate:
            Target bitrate for compressed audio in the resulting plot. Ignored when
            `compress_audio` is `False`.
        """
        return plot(
            self,
            layout=layout,
            config=config,
            points=points,
            display=display,
            compress_audio=compress_audio,
            bitrate=bitrate,
        )

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
    """Segment overlay trace.

    Parameters
    ----------
    items:
        Normalized segment dictionaries with `start`, `end`, `label`, and `color`.
    name:
        Hover label prefix shown for segments.
    lane:
        Whether the segment boxes are drawn in the top or bottom half of the plot.
    bg_alpha:
        Opacity used for the wide background band for each segment.
    box_alpha:
        Opacity used for the smaller label box for each segment.
    textfont:
        Optional Plotly annotation font overrides for segment labels.
    """
    items: list[dict[str, Any]]
    name: str = "Segment"
    lane: Literal["top", "bottom"] = "top"
    bg_alpha: float = 0.08
    box_alpha: float = 0.92
    textfont: dict[str, Any] = field(default_factory=dict)

    def plot(
        self,
        *,
        layout: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        points: int | None = None,
        display: str = "auto",
        compress_audio: bool = True,
        bitrate: Bitrate = None,
    ) -> "Plot":
        """Build a :class:`Plot` from this segments overlay.

        Parameters
        ----------
        layout:
            Optional Plotly layout overrides.
        config:
            Optional Plotly config overrides.
        points:
            Maximum number of displayed points after downsampling on non-segment
            traces. When omitted, it defaults to `max(3000, duration_seconds * 16)`
            for timed plots and all points for plots without a time-based notion.
            Use `-1` to force full-resolution display.
        display:
            Display mode used by :meth:`Plot.show`.
        compress_audio:
            Default audio compression policy for audio traces added to the same plot.
        bitrate:
            Target bitrate for compressed audio in the resulting plot. Ignored when
            `compress_audio` is `False`.
        """
        return plot(
            self,
            layout=layout,
            config=config,
            points=points,
            display=display,
            compress_audio=compress_audio,
            bitrate=bitrate,
        )

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
                    hover_x_array = np.unique(
                        np.concatenate(
                            (
                                np.array([start], dtype=np.float64),
                                segment_x,
                                np.array([end], dtype=np.float64),
                            )
                        )
                    )
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


AudioLikeTrace = AudioTrace | AudioFileTrace
PlotItem = AudioLikeTrace | SeriesTrace | SegmentsTrace


@dataclass(slots=True)
class Plot:
    """Interactive plot container returned by :func:`plot`.

    Parameters
    ----------
    data:
        Plotwave traces included in the figure.
    layout:
        Optional Plotly layout overrides.
    config:
        Optional Plotly config overrides.
    points:
        Maximum number of displayed points per trace after downsampling. When
        omitted, timed plots default to `max(3000, duration_seconds * 16)` and
        untimed plots default to all points. Use `-1` to force full-resolution
        display.
    display:
        Default display mode used by :meth:`show`.
    compress_audio:
        Default audio compression policy for HTML export, notebook display, and cached assets.
    bitrate:
        Default bitrate for compressed audio payloads. Ignored when
        `compress_audio` is `False`.
    """
    data: list[PlotItem]
    layout: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    points: int | None = None
    display: str = "auto"
    compress_audio: bool = True
    bitrate: Bitrate = None

    def _audio_render_options(
        self,
        *,
        compress_audio: bool | None = None,
        bitrate: Bitrate | object = _BITRATE_UNSET,
    ) -> tuple[bool, int | None]:
        effective_compress_audio = self.compress_audio if compress_audio is None else compress_audio
        bitrate_value = self.bitrate if bitrate is _BITRATE_UNSET else cast(Bitrate, bitrate)
        normalized_bitrate = (
            _normalize_bitrate(bitrate_value) if effective_compress_audio else None
        )
        return effective_compress_audio, normalized_bitrate

    def _effective_points(self) -> int:
        if self.points is not None:
            if self.points == ALL_POINTS:
                return ALL_POINTS
            if self.points <= 0:
                raise ValueError("points must be a positive integer or -1")
            return self.points

        spans = [_time_span_from_trace(item) for item in self.data]
        timed_spans = [span for span in spans if span is not None]
        if not timed_spans:
            return ALL_POINTS

        span_start = min(span[0] for span in timed_spans)
        span_end = max(span[1] for span in timed_spans)
        duration_seconds = max(0.0, span_end - span_start)
        return max(
            1,
            max(
                DEFAULT_BASE_POINTS,
                int(np.ceil(duration_seconds * DEFAULT_POINTS_PER_SECOND)),
            ),
        )

    def _prepared(
        self,
        *,
        embed_audio: bool = True,
        compress_audio: bool | None = None,
        bitrate: Bitrate | object = _BITRATE_UNSET,
    ) -> tuple[list[PreparedTrace], list[SegmentsTrace]]:
        effective_points = self._effective_points()
        effective_compress_audio, normalized_bitrate = self._audio_render_options(
            compress_audio=compress_audio,
            bitrate=bitrate,
        )
        prepared: list[PreparedTrace] = []
        overlays: list[SegmentsTrace] = []
        audio_items: list[AudioLikeTrace] = []
        pending_series: list[SeriesTrace] = []
        for item in self.data:
            if isinstance(item, SegmentsTrace):
                overlays.append(item)
            elif isinstance(item, (AudioTrace, AudioFileTrace)):
                audio_items.append(item)
                prepared.append(
                    item.prepared(
                        points=effective_points,
                        bitrate=normalized_bitrate,
                        compress_audio=effective_compress_audio,
                        embed_audio=embed_audio,
                    )
                )
            else:
                pending_series.append(item)

        if audio_items and any(item.time is None and item.sr is None for item in pending_series):
            raise ValueError(
                "SeriesTrace plotted with audio requires explicit time or sr."
            )

        for item in pending_series:
            prepared.append(item.prepared(points=effective_points))
        if not prepared and not overlays:
            raise ValueError("plot requires at least one trace")
        return prepared, overlays

    def _resolved(
        self,
        *,
        embed_audio: bool = True,
        compress_audio: bool | None = None,
        bitrate: Bitrate | object = _BITRATE_UNSET,
    ) -> tuple[list[PreparedTrace], dict[str, Any], dict[str, Any]]:
        prepared, overlays = self._prepared(
            embed_audio=embed_audio,
            compress_audio=compress_audio,
            bitrate=bitrate,
        )
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

    def _render_bundle(
        self,
        *,
        compress_audio: bool | None = None,
        bitrate: Bitrate | object = _BITRATE_UNSET,
    ) -> tuple[str, int]:
        prepared, layout, config = self._resolved(
            compress_audio=compress_audio,
            bitrate=bitrate,
        )
        frame_height = self._frame_height(layout)
        document = build_html(
            prepared,
            layout,
            config,
            frame_height=frame_height,
            segment_infos=self._segment_loop_infos(),
        )
        return document, frame_height

    def _notebook_render_bundle(
        self,
        *,
        compress_audio: bool | None = None,
        bitrate: Bitrate | object = _BITRATE_UNSET,
    ) -> tuple[str, int]:
        effective_compress_audio, normalized_bitrate = self._audio_render_options(
            compress_audio=compress_audio,
            bitrate=bitrate,
        )
        prepared, layout, config = self._resolved(
            embed_audio=False,
            compress_audio=effective_compress_audio,
            bitrate=bitrate,
        )
        asset_dir = pathlib.Path.cwd() / NOTEBOOK_ASSET_DIR
        inline_audio = _notebook_audio_mode() == "inline"
        asset_infos = [
            item.notebook_audio_asset(
                asset_dir=asset_dir,
                bitrate=normalized_bitrate,
                compress_audio=effective_compress_audio,
                inline=inline_audio,
            )
            for item in self.data
            if isinstance(item, (AudioTrace, AudioFileTrace))
        ]
        asset_iter = iter(asset_infos)
        notebook_traces: list[PreparedTrace] = []
        for trace in prepared:
            if trace.audio_info is None:
                notebook_traces.append(trace)
                continue
            audio_info = dict(trace.audio_info)
            audio_info.update(next(asset_iter))
            notebook_traces.append(
                PreparedTrace(
                    plotly_trace=trace.plotly_trace,
                    audio_info=audio_info,
                )
            )
        frame_height = self._frame_height(layout)
        document = build_html(
            notebook_traces,
            layout,
            config,
            frame_height=frame_height,
            segment_infos=self._segment_loop_infos(),
        )
        return document, frame_height

    def _segment_loop_infos(self) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        audio_count = sum(
            isinstance(item, (AudioTrace, AudioFileTrace))
            for item in self.data
        )
        overlay_index = 0
        for item in self.data:
            if not isinstance(item, SegmentsTrace):
                continue
            audio_index = min(overlay_index, audio_count - 1) if audio_count > 0 else None
            for segment in item.items:
                segments.append(
                    {
                        "start": float(segment["start"]),
                        "end": float(segment["end"]),
                        "label": str(segment["label"]),
                        "lane": item.lane,
                        "audio_index": audio_index,
                    }
                )
            overlay_index += 1
        return segments

    def _best_notebook_bundle(
        self,
        *,
        compress_audio: bool | None = None,
        bitrate: Bitrate | object = _BITRATE_UNSET,
    ) -> tuple[str, int]:
        return self._notebook_render_bundle(
            compress_audio=compress_audio,
            bitrate=bitrate,
        )

    def _document(
        self,
        *,
        compress_audio: bool | None = None,
        bitrate: Bitrate | object = _BITRATE_UNSET,
    ) -> str:
        document, _ = self._render_bundle(
            compress_audio=compress_audio,
            bitrate=bitrate,
        )
        return document

    def html(
        self,
        *,
        compress_audio: bool | None = None,
        bitrate: Bitrate | object = _BITRATE_UNSET,
    ) -> str:
        """Return a standalone HTML document for the plot.

        Parameters
        ----------
        compress_audio:
            Override the plot-level audio compression policy for this HTML export.
        bitrate:
            Override the bitrate used for compressed audio in this HTML export.
            Ignored when compression is disabled.
        """
        return self._document(
            compress_audio=compress_audio,
            bitrate=bitrate,
        )

    def save(
        self,
        path: str | pathlib.Path,
        *,
        compress_audio: bool | None = None,
        bitrate: Bitrate | object = _BITRATE_UNSET,
    ) -> pathlib.Path:
        """Write the plot HTML document to disk.

        Parameters
        ----------
        path:
            Destination HTML path.
        compress_audio:
            Override the plot-level audio compression policy for this export.
        bitrate:
            Override the bitrate used for compressed audio in this export.
            Ignored when compression is disabled.
        """
        output = pathlib.Path(path)
        output.write_text(
            self._document(
                compress_audio=compress_audio,
                bitrate=bitrate,
            ),
            encoding="utf-8",
        )
        return output

    def show(self) -> None:
        """Display the plot once using the same notebook representation as `Plot` itself."""
        if self.display == "inline" or (self.display == "auto" and _is_notebook()):
            try:
                from IPython.display import display
            except ImportError as exc:
                raise RuntimeError("Inline display requires IPython") from exc
            display(self)  # type: ignore[no-untyped-call]
            return

        if self.display == "none":
            return

        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(self._document())
            temp_path = pathlib.Path(tmp.name)
        webbrowser.open(temp_path.as_uri())
        return

    def _repr_html_(self) -> str:
        try:
            document, frame_height = self._best_notebook_bundle()
            return _iframe_html(document, height=frame_height)
        except Exception:
            document, frame_height = self._render_bundle()
            return _iframe_html(document, height=frame_height)

    def _frame_height(self, layout: dict[str, Any] | None = None) -> int:
        effective_layout = layout
        if effective_layout is None:
            _, effective_layout, _ = self._resolved()
        plot_height = int(effective_layout.get("height", 600))
        return plot_height + 45


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
    """Create an in-memory audio trace.

    Parameters
    ----------
    wav:
        Audio samples. A 2D input is averaged to mono for display and playback.
    sr:
        Sample rate in Hz.
    time:
        Optional explicit time axis for the samples. When omitted, time is
        inferred from `sr`.
    norm:
        Whether to normalize the displayed waveform to unit peak amplitude.
    clip:
        Optional symmetric quantile clipping applied only to the displayed waveform.
    step:
        Optional decimation factor applied before point-based display downsampling.
    **trace:
        Extra Plotly scatter arguments, such as `name`, `color`, `line`, `fill`,
        `mode`, or `opacity`.
    """
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


def audio_file(
    path: str | pathlib.Path,
    *,
    norm: bool = False,
    clip: float | None = None,
    step: int | None = None,
    **trace: Any,
) -> AudioFileTrace:
    """Create a file-backed audio trace.

    Parameters
    ----------
    path:
        Path to a local audio file. `soundfile` is used for metadata and decoding,
        so any format supported by the local libsndfile backend can be used.
    norm:
        Whether to normalize the displayed waveform to unit peak amplitude.
    clip:
        Optional symmetric quantile clipping applied only to the displayed waveform.
    step:
        Optional decimation factor applied before point-based display downsampling.
    **trace:
        Extra Plotly scatter arguments, such as `name`, `color`, `line`, `fill`,
        `mode`, or `opacity`.
    """
    _validate_trace_kwargs(trace)
    audio_path = pathlib.Path(path).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    try:
        info = sf.info(str(audio_path))
    except (OSError, RuntimeError, ValueError) as exc:
        _raise_soundfile_error(audio_path, action="inspect", exc=exc)
    if info.frames <= 0:
        raise ValueError("audio file cannot be empty")
    if info.samplerate <= 0:
        raise ValueError("audio file must have a positive sample rate")
    audio_format = _audio_format_from_path(audio_path, fallback=getattr(info, "format", None))
    frames = int(info.frames)
    # libsndfile can over-report MP3 frame counts in metadata; count decoded
    # PCM frames so the plot range and segment timings match the actual audio.
    if audio_format == "mp3":
        frames = _decoded_soundfile_frame_count(audio_path)
    return AudioFileTrace(
        path=audio_path,
        sr=float(info.samplerate),
        frames=frames,
        audio_format=audio_format,
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
    """Create a non-audio numeric series trace.

    Parameters
    ----------
    y:
        Series values to plot.
    time:
        Optional explicit x-axis values.
    sr:
        Optional sample rate used to derive the x-axis when `time` is omitted.
    step:
        Optional decimation factor applied before point-based display downsampling.
    **trace:
        Extra Plotly scatter arguments, such as `name`, `color`, `line`, `fill`,
        `mode`, or `opacity`.
    """
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
    if isinstance(data, (AudioTrace, AudioFileTrace, SeriesTrace, SegmentsTrace)):
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
        and all(
            isinstance(item, (AudioTrace, AudioFileTrace, SeriesTrace, SegmentsTrace))
            for item in data
        )
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
    points: int | None = None,
    display: str = "auto",
    compress_audio: bool = True,
    bitrate: Bitrate = None,
    **trace: Any,
) -> Plot:
    """Build an interactive plot from raw arrays or plotwave traces.

    Parameters
    ----------
    data:
        Raw numeric values, a single plotwave trace, or a sequence of traces.
        Raw numeric values become an audio trace when `sr` is provided and a series
        trace otherwise.
    sr:
        Sample rate in Hz for raw audio input. Not used when `data` is already a trace.
    time:
        Optional explicit x-axis for raw single-trace input.
    layout:
        Optional Plotly layout overrides.
    config:
        Optional Plotly config overrides.
    points:
        Maximum number of displayed points per trace after downsampling. When
        omitted, timed plots default to `max(3000, duration_seconds * 16)` and
        untimed plots default to all points. Use `-1` to force full-resolution
        display.
    display:
        Default display mode used by :meth:`Plot.show`.
    compress_audio:
        Whether embedded or cached audio should be MP3-compressed by default.
        When `False`, in-memory audio uses WAV and file-backed audio keeps its
        original bytes when possible.
    bitrate:
        Target bitrate for compressed audio, for example `"32k"` or `64000`.
        Ignored when `compress_audio` is `False`.
    **trace:
        Extra Plotly scatter arguments for raw single-trace input, such as `name`,
        `color`, `line`, `fill`, `mode`, or `opacity`.
    """
    if display not in {"auto", "inline", "browser", "none"}:
        raise ValueError("display must be one of: auto, inline, browser, none")
    if compress_audio:
        _normalize_bitrate(bitrate)
    items = _coerce_data(data, sr=sr, time=time, trace=dict(trace))
    return Plot(
        data=items,
        layout=layout,
        config=config,
        points=points,
        display=display,
        compress_audio=compress_audio,
        bitrate=bitrate,
    )


def audio_trace_plot(
    data: Sequence[dict[str, Any]],
    title: str | None = "Interactive Audio/Data Plot",
    max_points_display: int | None = None,
    output_mode: str = "file",
    output_path: str = "interactive_plot.html",
    iframe_height: str = "600px",
    audio_format: str = "mp3",
    audio_bitrate: str = "16k",
) -> str | Any | None:
    """Legacy helper for the pre-trace-dataclass API.

    Parameters
    ----------
    data:
        Sequence of legacy trace dictionaries. Audio entries should contain `y` and
        `sr`, and may also include `x`, `name`, `color`, `line`, `fill`, `mode`,
        `opacity`, `remove_outliers`, and `decimate_by`.
    title:
        Plot title.
    max_points_display:
        Maximum number of displayed points per trace after downsampling. When
        omitted, timed plots default to `max(3000, duration_seconds * 16)` and
        untimed plots default to all points. Use `-1` to force full-resolution
        display.
    output_mode:
        One of `"file"`, `"html_string"`, or `"jupyter"`. The Jupyter mode
        displays the plot using :meth:`Plot.show` and returns `None`.
    output_path:
        Output HTML path when `output_mode="file"`.
    iframe_height:
        Height used for the generated plot, such as `"600px"`.
    audio_format:
        Kept for backward compatibility and currently ignored.
    audio_bitrate:
        Target bitrate for compressed audio payloads.
    """
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
        plot_obj.show()
        return None
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
    """Create a segment overlay trace.

    Parameters
    ----------
    items:
        Segment dictionaries. Each segment must define `start` and `end`, and may
        also define `label`, `name`, and `color`.
    name:
        Hover label prefix shown for each segment.
    lane:
        Whether the segment boxes are drawn in the top or bottom half of the plot.
    color_map:
        Optional fallback mapping from segment label to color.
    bg_alpha:
        Opacity used for the wide background band for each segment.
    box_alpha:
        Opacity used for the smaller label box for each segment.
    textfont:
        Optional Plotly annotation font overrides for segment labels.
    """
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
