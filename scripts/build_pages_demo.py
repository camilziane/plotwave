from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import plotwave

REPO_ROOT = Path(__file__).resolve().parents[1]

REPO_URL = "https://github.com/camilziane/plotwave"
PAGES_URL = "https://camilziane.github.io/plotwave/"
PAGES_PATH = "/plotwave"


def main() -> None:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    docs_assets_dir = docs_dir / "assets"
    docs_assets_dir.mkdir(exist_ok=True)
    shutil.copy2(
        REPO_ROOT / "assets" / "logo_name_black.png",
        docs_assets_dir / "logo_name_black.png",
    )
    favicon_source_dir = REPO_ROOT / "docs" / "favicon"
    for favicon_name in (
        "favicon.ico",
        "favicon.svg",
        "favicon-96x96.png",
        "apple-touch-icon.png",
        "site.webmanifest",
        "web-app-manifest-192x192.png",
        "web-app-manifest-512x512.png",
    ):
        shutil.copy2(favicon_source_dir / favicon_name, docs_dir / favicon_name)
    root_manifest = docs_dir / "site.webmanifest"
    root_manifest.write_text(
        """{
  "name": "plotwave",
  "short_name": "plotwave",
  "icons": [
    {
      "src": "/plotwave/web-app-manifest-192x192.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "maskable"
    },
    {
      "src": "/plotwave/web-app-manifest-512x512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "maskable"
    }
  ],
  "theme_color": "#0b0c10",
  "background_color": "#f7f4ee",
  "display": "standalone"
}
""",
        encoding="utf-8",
    )
    first_audio_path = REPO_ROOT / "examples" / "plotwave_C.mp3"
    second_audio_path = REPO_ROOT / "examples" / "plotwave_E5.mp3"

    demo = plotwave.plot(
        [
            plotwave.audio_file(
                first_audio_path,
                name="First Audio",
                color="#2563eb",
                line={"width": 1.3},
            ),
            plotwave.audio_file(
                second_audio_path,
                name="Second Audio",
                color="#ea580c",
                line={"width": 1.3, "dash": "dot"},
            ),
            plotwave.segments(
                [
                    {"start": 0.0, "end": 2.09, "label": "C"},
                    {"start": 2.09, "end": 4.18, "label": "Asus2"},
                    {"start": 4.18, "end": 6.27, "label": "Esus2"},
                    {"start": 6.27, "end": 8.36, "label": "B"},
                ],
                name="First Chords",
                lane="top",
                textfont={"color": "white"},
                color_map={
                    "C": "#0f62fe",
                    "Asus2": "#367af7",
                    "Esus2": "#82adfc",
                    "B": "#bcd2fb",
                },
            ),
            plotwave.segments(
                [
                    {"start": 0.0, "end": 2.09, "label": "E5"},
                    {"start": 2.09, "end": 4.18, "label": "B"},
                    {"start": 4.18, "end": 6.27, "label": "B"},
                    {"start": 6.27, "end": 8.36, "label": "F#"},
                ],
                name="Second Chords",
                lane="bottom",
                textfont={"color": "white"},
                color_map={
                    "E5": "#fe5b0f",
                    "B": "#ff7636",
                    "B": "#f79466",
                    "F#": "#fdbc9e",
                },
            ),
        ],
        layout={
            "title": {"text": "Two MP3 tracks with loopable chord labels"},
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
            },
            "yaxis": {
                "title": {"text": "Amplitude"},
                "showgrid": True,
                "gridcolor": "#e7edf5",
                "zeroline": False,
            },
        },
    )
    demo_html = demo.html()
    (docs_dir / "demo.html").write_text(demo_html, encoding="utf-8")
    demo_version = hashlib.sha256(demo_html.encode("utf-8")).hexdigest()[:10]

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>plotwave live demo</title>
  <link rel="icon" type="image/png" href="{PAGES_PATH}/favicon-96x96.png" sizes="96x96">
  <link rel="icon" type="image/svg+xml" href="{PAGES_PATH}/favicon.svg">
  <link rel="icon" href="{PAGES_PATH}/favicon.ico">
  <link rel="shortcut icon" href="{PAGES_PATH}/favicon.ico">
  <link rel="apple-touch-icon" sizes="180x180" href="{PAGES_PATH}/apple-touch-icon.png">
  <link rel="mask-icon" href="{PAGES_PATH}/favicon.svg" color="#111827">
  <link rel="manifest" href="{PAGES_PATH}/site.webmanifest">
  <meta name="apple-mobile-web-app-title" content="plotwave">
  <meta name="theme-color" content="#0b0c10">
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
      gap: 22px;
      margin-bottom: 24px;
      padding: 10px 0 2px;
    }}
    .hero-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      flex-wrap: wrap;
    }}
    .hero-copy {{
      display: grid;
      gap: 18px;
      align-content: start;
      max-width: 860px;
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
    .brand-lockup {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      min-height: 1px;
    }}
    .brand-lockup img {{
      display: block;
      width: min(190px, 34vw);
      height: auto;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2.4rem, 5.8vw, 4.9rem);
      line-height: 0.9;
      letter-spacing: -0.04em;
      max-width: 8.5ch;
    }}
    .lede {{
      max-width: 58ch;
      margin: 0;
      font-size: 1.08rem;
      line-height: 1.6;
      color: var(--muted);
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 4px;
    }}
    .hero-meta {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
      width: 100%;
    }}
    .meta-card {{
      padding: 16px 16px 18px;
      border: 1px solid rgba(224, 218, 208, 0.95);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.62);
    }}
    .meta-label {{
      margin: 0 0 8px;
      color: #4f46e5;
      font: 700 0.7rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .meta-text {{
      margin: 0;
      color: var(--text);
      font: 600 0.92rem/1.45 ui-sans-serif, system-ui, sans-serif;
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
      .hero-top {{
        align-items: flex-start;
      }}
      .brand-lockup img {{
        width: min(150px, 48vw);
      }}
      .hero-meta {{
        grid-template-columns: 1fr;
        gap: 12px;
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
      <div class="hero-top">
        <div class="eyebrow">plotwave live demo</div>
        <div class="brand-lockup">
          <img src="./assets/logo_name_black.png" alt="plotwave logo">
        </div>
      </div>
      <div class="hero-copy">
        <h1>Hear the waveform while you inspect it.</h1>
        <p class="lede">
          <strong>plotwave</strong> is a Python library that turns Plotly
          signal views into interactive, playable audio plots. This demo lets
          you switch between two real MP3 tracks while keeping both labeled
          chord lanes in view and loop a chord by clicking its label.
        </p>
        <div class="actions">
          <a class="button primary" href="demo.html?v={demo_version}">Open demo only</a>
          <a class="button secondary" href="{REPO_URL}">View repository</a>
        </div>
      </div>
      <div class="hero-meta">
        <div class="meta-card">
          <p class="meta-label">Playable</p>
          <p class="meta-text">Click anywhere in the waveform to seek and listen.</p>
        </div>
        <div class="meta-card">
          <p class="meta-label">Overlay-ready</p>
          <p class="meta-text">
            Mix multiple playable tracks and labeled segment lanes in one view.
          </p>
        </div>
        <div class="meta-card">
          <p class="meta-label">Shareable</p>
          <p class="meta-text">Export the same interaction as a standalone HTML file.</p>
        </div>
      </div>
    </section>

    <section class="frame">
      <iframe
        title="plotwave interactive demo"
        src="./demo.html?v={demo_version}"
        loading="eager"
      ></iframe>
    </section>

    <section class="notes">
      <div>
        Try clicking directly on the waveform to seek, then click a chord label
        to loop that region while switching between the two audio tracks.
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
