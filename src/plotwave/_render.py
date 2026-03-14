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
    controls_group_id = f"{root_id}-controls-group"
    timing_group_id = f"{root_id}-timing-group"
    toggle_id = f"{root_id}-toggle"
    toggle_icon_id = f"{root_id}-toggle-icon"
    toggle_label_id = f"{root_id}-toggle-label"
    channel_id = f"{root_id}-channel"
    speed_id = f"{root_id}-speed"
    current_time_id = f"{root_id}-current-time"
    current_time_value_id = f"{root_id}-current-time-value"
    total_time_id = f"{root_id}-total-time"

    plotly_traces = [trace.plotly_trace for trace in traces]
    audio_infos = [trace.audio_info for trace in traces if trace.audio_info is not None]

    channel_options_html = "".join(
        f"<option value='{index}'>{html.escape(info['name'])}</option>"
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
      font-family: "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background: #fcfcfb;
    }}
    #{root_id} {{
      width: 100%;
      height: {frame_height}px;
      display: flex;
      flex-direction: column;
      color: #161616;
    }}
    #{controls_id} {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex: 0 0 auto;
      flex-wrap: wrap;
      gap: 12px;
      padding: 10px 14px;
      background: rgba(252, 252, 251, 0.92);
      border-bottom: 1px solid #e4e1dc;
    }}
    #{controls_group_id},
    #{timing_group_id} {{
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }}
    #{timing_group_id} {{
      justify-content: flex-end;
      flex-wrap: wrap;
      color: #5c5852;
      font-size: 12px;
      letter-spacing: 0.01em;
    }}
    #{plot_id} {{
      width: 100%;
      flex: 1 1 auto;
      min-height: 240px;
    }}
    select, button {{
      font: inherit;
      font-size: 13px;
      border-radius: 10px;
      transition: border-color 120ms ease, background 120ms ease, box-shadow 120ms ease;
    }}
    #{toggle_id} {{
      width: 94px;
      height: 36px;
      box-sizing: border-box;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      padding: 0 14px;
      border: 1px solid #d8d3cc;
      background: #fff;
      color: #151515;
      text-align: center;
      line-height: 1;
      white-space: nowrap;
      box-shadow: 0 1px 0 rgba(20, 20, 20, 0.03);
      cursor: pointer;
    }}
    #{toggle_id}:hover {{
      background: #f7f4ef;
      border-color: #cfc8bf;
    }}
    #{toggle_icon_id} {{
      display: inline-flex;
      width: 12px;
      flex: 0 0 12px;
      justify-content: center;
      font-size: 11px;
    }}
    #{toggle_label_id} {{
      min-width: 34px;
    }}
    #{toggle_id}:focus-visible,
    select:focus-visible {{
      outline: none;
      box-shadow: 0 0 0 3px rgba(34, 34, 34, 0.08);
      border-color: #bfb7ad;
    }}
    select {{
      height: 36px;
      box-sizing: border-box;
      padding: 0 32px 0 12px;
      border: 1px solid #d8d3cc;
      background: #fff;
      color: #1f1d1a;
      appearance: none;
      line-height: 1.2;
      background-image:
        linear-gradient(45deg, transparent 50%, #6a645c 50%),
        linear-gradient(135deg, #6a645c 50%, transparent 50%);
      background-position:
        calc(100% - 18px) 15px,
        calc(100% - 12px) 15px;
      background-size: 6px 6px, 6px 6px;
      background-repeat: no-repeat;
    }}
    select:hover {{
      background-color: #f7f4ef;
      border-color: #cfc8bf;
    }}
    select:disabled,
    #{toggle_id}:disabled {{
      opacity: 0.55;
      cursor: default;
    }}
    .plotwave-time-label {{
      display: inline-flex;
      align-items: center;
      height: 36px;
      padding: 0 12px;
      border-radius: 10px;
      background: #f5f2ec;
      color: #514b44;
      white-space: nowrap;
    }}
    .plotwave-time-label strong {{
      font-weight: 600;
      color: #191714;
    }}
    @media (max-width: 720px) {{
      #{controls_id} {{
        flex-wrap: wrap;
        justify-content: flex-start;
      }}
      #{timing_group_id} {{
        width: 100%;
        justify-content: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <div id="{root_id}" tabindex="0">
    <div id="{controls_id}">
      <div id="{controls_group_id}">
        <button id="{toggle_id}" {controls_disabled}>
          <span id="{toggle_icon_id}">▶</span><span id="{toggle_label_id}">Play</span>
        </button>
        <select id="{channel_id}" {controls_disabled} style="{controls_style}">
          {channel_options_html}
        </select>
        <select id="{speed_id}" {controls_disabled} style="{controls_style}">
          {speed_options_html}
        </select>
      </div>
      <div id="{timing_group_id}">
        <span id="{current_time_id}" class="plotwave-time-label">
          Current Time: <strong id="{current_time_value_id}">{float(x_range[0]):.2f}s</strong>
        </span>
        <span id="{total_time_id}" class="plotwave-time-label">
          Total Timespan: <strong>{duration:.2f}s</strong>
        </span>
      </div>
    </div>
    <div id="{plot_id}"></div>
    {audio_elements_html}
  </div>
  <script>
    const rootElement = document.getElementById({json.dumps(root_id)});
    const controlsDiv = document.getElementById({json.dumps(controls_id)});
    const plotDiv = document.getElementById({json.dumps(plot_id)});
    const plotData = {json.dumps(plotly_traces)};
    const layout = {json.dumps(layout)};
    const config = {json.dumps(config)};
    const audioInfos = {json.dumps(audio_infos)};
    const audioIds = {json.dumps(audio_element_ids)};
    const resizePlot = () => Plotly.Plots.resize(plotDiv);

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

    Plotly.newPlot(plotDiv, plotData, layout, config).then(() => {{
      requestAnimationFrame(() => requestAnimationFrame(resizePlot));
    }});

    if (typeof ResizeObserver !== "undefined") {{
      const resizeObserver = new ResizeObserver(() => resizePlot());
      resizeObserver.observe(rootElement);
      resizeObserver.observe(controlsDiv);
    }}

    if (audioInfos.length > 0) {{
      const toggleBtn = document.getElementById({json.dumps(toggle_id)});
      const toggleIcon = document.getElementById({json.dumps(toggle_icon_id)});
      const toggleLabel = document.getElementById({json.dumps(toggle_label_id)});
      const channelSelect = document.getElementById({json.dumps(channel_id)});
      const speedSelect = document.getElementById({json.dumps(speed_id)});
      const currentTimeValue = document.getElementById({json.dumps(current_time_value_id)});
      const audioElements = audioIds.map((audioId) => document.getElementById(audioId));
      let currentAudioIndex = 0;
      let currentAudio = audioElements[currentAudioIndex];
      let animationFrameId = null;

      const cursorTraceIndex = () => plotData.length - 1;
      const currentGlobalTime = () => {{
        const info = audioInfos[currentAudioIndex];
        return info.start_time + currentAudio.currentTime;
      }};
      const setToggleState = (isPlaying) => {{
        toggleIcon.textContent = isPlaying ? "⏸" : "▶";
        toggleLabel.textContent = isPlaying ? "Pause" : "Play";
      }};
      const updateCurrentTimeLabel = (time) => {{
        currentTimeValue.textContent = `${{time.toFixed(2)}}s`;
      }};
      const drawCursor = (time) => (
        Plotly.restyle(plotDiv, {{ x: [[time, time]] }}, [cursorTraceIndex()])
      );
      const syncTimeline = (time) => {{
        updateCurrentTimeLabel(time);
        drawCursor(time);
      }};
      const seekBy = (deltaSeconds) => {{
        const info = audioInfos[currentAudioIndex];
        const nextTime = Math.max(
          0,
          Math.min(info.duration, currentAudio.currentTime + deltaSeconds)
        );
        currentAudio.currentTime = nextTime;
        syncTimeline(info.start_time + nextTime);
        if (!currentAudio.paused) {{
          if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
          animationFrameId = requestAnimationFrame(updatePlayback);
        }}
      }};
      const togglePlayback = async () => {{
        if (currentAudio.paused) {{
          await currentAudio.play();
          setToggleState(true);
          animationFrameId = requestAnimationFrame(updatePlayback);
        }} else {{
          currentAudio.pause();
          setToggleState(false);
          if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
        }}
      }};

      const updatePlayback = () => {{
        if (!currentAudio || currentAudio.paused) return;
        syncTimeline(currentGlobalTime());
        animationFrameId = requestAnimationFrame(updatePlayback);
      }};

      toggleBtn.onclick = togglePlayback;

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

        syncTimeline(isInsideNextAudio ? globalTime : nextInfo.start_time);

        if (wasPlaying && isInsideNextAudio) {{
          await currentAudio.play();
          animationFrameId = requestAnimationFrame(updatePlayback);
        }}
        if (!isInsideNextAudio) {{
          setToggleState(false);
          return;
        }}
        setToggleState(!currentAudio.paused);
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
          setToggleState(false);
          if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
          const finalTime = audioInfos[index].start_time + audioInfos[index].duration;
          syncTimeline(finalTime);
        }};
      }});

      plotDiv.addEventListener("click", (event) => {{
        rootElement.focus();
        const fullLayout = plotDiv._fullLayout;
        if (!fullLayout || !fullLayout.xaxis) return;
        const plotRect = plotDiv.getBoundingClientRect();
        const clickXPixel = event.clientX - plotRect.left - fullLayout.margin.l;
        const globalClickTime = fullLayout.xaxis.p2c(clickXPixel);
        const info = audioInfos[currentAudioIndex];
        const targetTime = globalClickTime - info.start_time;
        if (targetTime >= 0 && targetTime <= info.duration) {{
          currentAudio.currentTime = targetTime;
          syncTimeline(globalClickTime);
          if (!currentAudio.paused) {{
            if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
            animationFrameId = requestAnimationFrame(updatePlayback);
          }}
        }}
      }});

      rootElement.addEventListener("keydown", async (event) => {{
        if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
        const activeElement = document.activeElement;
        const tagName = activeElement ? activeElement.tagName : "";
        if (
          tagName === "BUTTON" ||
          tagName === "SELECT" ||
          tagName === "INPUT" ||
          tagName === "TEXTAREA" ||
          (activeElement && activeElement.isContentEditable)
        ) {{
          return;
        }}
        if (event.code === "Space") {{
          if (event.repeat) return;
          event.preventDefault();
          await togglePlayback();
          return;
        }}
        if (event.key === "j" || event.key === "J") {{
          event.preventDefault();
          seekBy(-5);
          return;
        }}
        if (event.key === "l" || event.key === "L") {{
          event.preventDefault();
          seekBy(5);
        }}
      }});
      setToggleState(false);
      syncTimeline(audioInfos[currentAudioIndex].start_time);
    }}

    window.addEventListener("resize", () => Plotly.Plots.resize(plotDiv));
  </script>
</body>
</html>
"""
    return textwrap.dedent(script).strip()
