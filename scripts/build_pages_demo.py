from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import plotwave

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_URL = "https://github.com/camilziane/plotwave"
PAGES_URL = "https://camilziane.github.io/plotwave/"


def main() -> None:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    signal_helpers = import_module("examples.signal_helpers")
    build_progression = signal_helpers.build_progression
    segment_items = signal_helpers.segment_items
    smooth_envelope = signal_helpers.smooth_envelope

    sr = 16_000
    chord_names = ["Bm", "G", "D", "A"]
    time, progression, segments = build_progression(chord_names, sr=sr)
    envelope = smooth_envelope(progression, window=1000)

    pred_segments = [
        {"start": 0.0, "end": 0.55, "label": "Bm"},
        {"start": 2.95, "end": 4.85, "label": "G"},
        {"start": 6.1, "end": 8.05, "label": "D"},
        {"start": 8.15, "end": 8.35, "label": "A"},
        {"start": 9.15, "end": 11.0, "label": "A"},
    ]
    gt_segments = segment_items(segments)
    color_map = {
        "Bm": "#4f46e5",
        "G": "#059669",
        "D": "#d97706",
        "A": "#ef4444",
    }

    demo = plotwave.plot(
        [
            plotwave.audio(
                progression,
                sr,
                name="Progression",
                color="#5b6c8f",
                line={"width": 1.3},
                hovertemplate="t=%{x:.2f}s<br>sample=%{y:.3f}<extra>Audio</extra>",
            ),
            plotwave.series(
                envelope,
                time=time,
                name="Envelope",
                color="rgba(249, 115, 22, 0.96)",
                line={"width": 3},
                fill="tozeroy",
                opacity=0.18,
                hovertemplate="t=%{x:.2f}s<br>env=%{y:.3f}<extra>Envelope</extra>",
            ),
            plotwave.segments(pred_segments, name="Pred", lane="top", color_map=color_map),
            plotwave.segments(gt_segments, name="Ground truth", lane="bottom", color_map=color_map),
        ],
        layout={
            "title": {"text": "Interactive plotwave demo"},
            "height": 640,
            "plot_bgcolor": "#fffdf8",
            "paper_bgcolor": "#ffffff",
            "legend": {
                "bgcolor": "rgba(255,255,255,0.86)",
                "bordercolor": "#e2e8f0",
                "borderwidth": 1,
            },
            "xaxis": {
                "title": {"text": "Time (seconds)"},
                "showgrid": True,
                "gridcolor": "#e7edf5",
                "zeroline": False,
                "range": [0, time[-1] * 1.08],
            },
            "yaxis": {
                "title": {"text": "Amplitude"},
                "showgrid": True,
                "gridcolor": "#e7edf5",
                "zeroline": False,
            },
        },
    )
    (docs_dir / "demo.html").write_text(demo.html(), encoding="utf-8")

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>plotwave live demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f4ee;
      --paper: rgba(255, 255, 255, 0.82);
      --line: #dfd6c7;
      --text: #1f2937;
      --muted: #5b6472;
      --accent: #d97706;
      --accent-2: #4f46e5;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--text);
      background:
        radial-gradient(circle at top right, rgba(79, 70, 229, 0.08), transparent 32%),
        radial-gradient(circle at left 20%, rgba(217, 119, 6, 0.1), transparent 28%),
        linear-gradient(180deg, #f9f6f1 0%, #f4efe7 100%);
    }}
    main {{
      width: min(1160px, calc(100% - 32px));
      margin: 0 auto;
      padding: 40px 0 56px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      margin-bottom: 24px;
    }}
    .eyebrow {{
      width: fit-content;
      padding: 6px 10px;
      border-radius: 999px;
      font: 600 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--accent-2);
      background: rgba(79, 70, 229, 0.1);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2.4rem, 6vw, 4.4rem);
      line-height: 0.96;
      letter-spacing: -0.04em;
      max-width: 10ch;
    }}
    .lede {{
      max-width: 70ch;
      margin: 0;
      font-size: 1.1rem;
      line-height: 1.6;
      color: var(--muted);
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 4px;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 12px 16px;
      border-radius: 999px;
      border: 1px solid transparent;
      text-decoration: none;
      font: 600 14px/1.1 ui-sans-serif, system-ui, sans-serif;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }}
    .button:hover {{
      transform: translateY(-1px);
    }}
    .button.primary {{
      color: white;
      background: linear-gradient(135deg, var(--accent-2), #7c3aed);
      box-shadow: 0 12px 28px rgba(79, 70, 229, 0.2);
    }}
    .button.secondary {{
      color: var(--text);
      background: rgba(255, 255, 255, 0.56);
      border-color: var(--line);
      backdrop-filter: blur(8px);
    }}
    .frame {{
      overflow: hidden;
      border: 1px solid rgba(223, 214, 199, 0.9);
      border-radius: 28px;
      background: var(--paper);
      box-shadow: 0 28px 80px rgba(56, 48, 37, 0.12);
      backdrop-filter: blur(8px);
    }}
    iframe {{
      display: block;
      width: 100%;
      height: 700px;
      border: 0;
      background: white;
    }}
    .notes {{
      display: grid;
      gap: 8px;
      margin-top: 16px;
      padding: 0 4px;
      color: var(--muted);
      font: 500 0.96rem/1.5 ui-sans-serif, system-ui, sans-serif;
    }}
    code {{
      padding: 2px 6px;
      border-radius: 999px;
      background: rgba(91, 108, 143, 0.1);
      font: 600 0.9em ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #38455f;
    }}
    @media (max-width: 720px) {{
      main {{
        width: min(100% - 18px, 1160px);
        padding-top: 18px;
      }}
      .frame {{
        border-radius: 20px;
      }}
      iframe {{
        height: 760px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="eyebrow">plotwave live demo</div>
      <h1>Hear the waveform while you inspect it.</h1>
      <p class="lede">
        <strong>plotwave</strong> turns Plotly signal views into interactive,
        playable audio plots. This demo overlays waveform, envelope, and
        annotated segments in one shareable HTML view.
      </p>
      <div class="actions">
        <a class="button primary" href="demo.html">Open demo only</a>
        <a class="button secondary" href="{REPO_URL}">View repository</a>
      </div>
    </section>

    <section class="frame">
      <iframe
        title="plotwave interactive demo"
        src="./demo.html"
        loading="eager"
      ></iframe>
    </section>

    <section class="notes">
      <div>
        Try clicking directly on the waveform to seek, then switch between
        segment lanes while listening.
      </div>
      <div>Live page URL: <a href="{PAGES_URL}">{PAGES_URL}</a></div>
      <div>Rebuild locally with <code>uv run python scripts/build_pages_demo.py</code>.</div>
    </section>
  </main>
</body>
</html>
"""
    (docs_dir / "index.html").write_text(index_html, encoding="utf-8")


if __name__ == "__main__":
    main()
