from __future__ import annotations

import html
import json
import textwrap
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PreparedTrace:
    plotly_trace: dict[str, Any]
    audio_info: dict[str, Any] | None = None


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def build_html(
    traces: list[PreparedTrace],
    layout: dict[str, Any],
    config: dict[str, Any],
    frame_height: int,
) -> str:
    root_id = f"plotwave-{uuid.uuid4().hex}"
    plot_id = f"{root_id}-plot"
    controls_id = f"{root_id}-controls"
    toggle_id = f"{root_id}-toggle"
    channel_id = f"{root_id}-channel"
    speed_id = f"{root_id}-speed"

    plotly_traces = [trace.plotly_trace for trace in traces]
    audio_infos = [trace.audio_info for trace in traces if trace.audio_info is not None]

    channel_options_html = "".join(
        f"<option value='{index}'>🎵 {html.escape(info['name'])}</option>"
        for index, info in enumerate(audio_infos)
    )
    audio_elements_html = "".join(
        (
            f"<audio id='{root_id}-audio-{index}' "
            f"src='data:audio/wav;base64,{info['b64_data']}'></audio>"
        )
        for index, info in enumerate(audio_infos)
    )
    speed_options_html = "".join(
        [
            f"<option value='{speed}' {'selected' if speed == 1.0 else ''}>{speed}x</option>"
            for speed in [0.5, 0.75, 1.0, 1.5, 2.0]
        ]
    )

    controls_disabled = "disabled" if not audio_infos else ""
    controls_style = "" if audio_infos else "display: none;"
    plot_height = max(frame_height - 45, 240)
    x_range = layout.get("xaxis", {}).get("range", [0.0, 1.0])
    y_range = layout.get("yaxis", {}).get("range", [-1.0, 1.0])
    audio_element_ids = [f"{root_id}-audio-{index}" for index in range(len(audio_infos))]
    title = layout.get("title", "plotwave")
    if isinstance(title, dict):
        title_text = str(title.get("text", "plotwave"))
    else:
        title_text = str(title)
    duration = float(x_range[1]) - float(x_range[0])

    script = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title_text)}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      overflow: hidden;
    }}
    body {{
      font-family: sans-serif;
      background: #fff;
    }}
    #{root_id} {{
      width: 100%;
      height: {frame_height}px;
    }}
    #{controls_id} {{
      display: flex;
      gap: 8px;
      align-items: center;
      background: #f0f0f0;
      padding: 5px 10px;
      border-bottom: 1px solid #ccc;
    }}
    #{plot_id} {{
      width: 100%;
      height: {plot_height}px;
    }}
    select, button {{
      font: inherit;
    }}
  </style>
</head>
<body>
  <div id="{root_id}">
    <div id="{controls_id}">
      <button id="{toggle_id}" {controls_disabled}>▶ Play</button>
      <select id="{channel_id}" {controls_disabled} style="{controls_style}">
        {channel_options_html}
      </select>
      <select id="{speed_id}" {controls_disabled} style="{controls_style}">
        {speed_options_html}
      </select>
      <span style="font-size:12px;">Total Timespan: {duration:.2f}s</span>
    </div>
    <div id="{plot_id}"></div>
    {audio_elements_html}
  </div>
  <script>
    const plotDiv = document.getElementById({json.dumps(plot_id)});
    const plotData = {json.dumps(plotly_traces)};
    const layout = {json.dumps(layout)};
    const config = {json.dumps(config)};
    const audioInfos = {json.dumps(audio_infos)};
    const audioIds = {json.dumps(audio_element_ids)};

    if (audioInfos.length > 0) {{
      const initialTime = audioInfos[0].start_time;
      plotData.push({{
        x: [initialTime, initialTime],
        y: [{json.dumps(y_range[0])}, {json.dumps(y_range[1])}],
        mode: "lines",
        type: "scatter",
        name: "cursor",
        showlegend: false,
        line: {{ color: "red", width: 2.5 }}
      }});
    }}

    Plotly.newPlot(plotDiv, plotData, layout, config);

    if (audioInfos.length > 0) {{
      const toggleBtn = document.getElementById({json.dumps(toggle_id)});
      const channelSelect = document.getElementById({json.dumps(channel_id)});
      const speedSelect = document.getElementById({json.dumps(speed_id)});
      const audioElements = audioIds.map((audioId) => document.getElementById(audioId));
      let currentAudioIndex = 0;
      let currentAudio = audioElements[currentAudioIndex];
      let animationFrameId = null;

      const cursorTraceIndex = () => plotData.length - 1;
      const drawCursor = (time) => (
        Plotly.restyle(plotDiv, {{ x: [[time, time]] }}, [cursorTraceIndex()])
      );

      const updatePlayback = () => {{
        if (!currentAudio || currentAudio.paused) return;
        const info = audioInfos[currentAudioIndex];
        drawCursor(info.start_time + currentAudio.currentTime);
        animationFrameId = requestAnimationFrame(updatePlayback);
      }};

      toggleBtn.onclick = async () => {{
        if (currentAudio.paused) {{
          await currentAudio.play();
          toggleBtn.textContent = "⏸ Pause";
          animationFrameId = requestAnimationFrame(updatePlayback);
        }} else {{
          currentAudio.pause();
          toggleBtn.textContent = "▶ Play";
          if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
        }}
      }};

      channelSelect.onchange = async () => {{
        const previousInfo = audioInfos[currentAudioIndex];
        const globalTime = previousInfo.start_time + currentAudio.currentTime;
        const wasPlaying = !currentAudio.paused;
        currentAudio.pause();
        if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
        currentAudioIndex = Number(channelSelect.value);
        currentAudio = audioElements[currentAudioIndex];
        currentAudio.playbackRate = Number(speedSelect.value);
        const nextInfo = audioInfos[currentAudioIndex];
        const nextTime = globalTime - nextInfo.start_time;
        const isInsideNextAudio = nextTime >= 0 && nextTime <= nextInfo.duration;

        if (isInsideNextAudio) {{
          currentAudio.currentTime = nextTime;
        }} else {{
          currentAudio.currentTime = 0;
        }}

        drawCursor(globalTime);

        if (wasPlaying && isInsideNextAudio) {{
          await currentAudio.play();
          animationFrameId = requestAnimationFrame(updatePlayback);
        }}
        if (!isInsideNextAudio) {{
          toggleBtn.textContent = "▶ Play";
          return;
        }}
        toggleBtn.textContent = currentAudio.paused ? "▶ Play" : "⏸ Pause";
      }};

      speedSelect.onchange = () => {{
        const rate = Number(speedSelect.value);
        audioElements.forEach((audioElement) => {{
          audioElement.playbackRate = rate;
        }});
      }};

      audioElements.forEach((audioElement, index) => {{
        audioElement.onended = () => {{
          if (index !== currentAudioIndex) return;
          toggleBtn.textContent = "▶ Play";
          if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
          const finalTime = audioInfos[index].start_time + audioInfos[index].duration;
          drawCursor(finalTime);
        }};
      }});

      plotDiv.addEventListener("click", (event) => {{
        const fullLayout = plotDiv._fullLayout;
        if (!fullLayout || !fullLayout.xaxis) return;
        const plotRect = plotDiv.getBoundingClientRect();
        const clickXPixel = event.clientX - plotRect.left - fullLayout.margin.l;
        const globalClickTime = fullLayout.xaxis.p2c(clickXPixel);
        const info = audioInfos[currentAudioIndex];
        const targetTime = globalClickTime - info.start_time;
        if (targetTime >= 0 && targetTime <= info.duration) {{
          currentAudio.currentTime = targetTime;
          drawCursor(globalClickTime);
          if (!currentAudio.paused) {{
            if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
            animationFrameId = requestAnimationFrame(updatePlayback);
          }}
        }}
      }});
    }}

    window.addEventListener("resize", () => Plotly.Plots.resize(plotDiv));
  </script>
</body>
</html>
"""
    return textwrap.dedent(script).strip()
