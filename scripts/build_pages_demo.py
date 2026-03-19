from __future__ import annotations

import html
import hashlib
import io
import keyword
import shutil
import token
import textwrap
import tokenize
from pathlib import Path

import plotwave

REPO_ROOT = Path(__file__).resolve().parents[1]

REPO_URL = "https://github.com/camilziane/plotwave"
PAGES_URL = "https://camilziane.github.io/plotwave/"
PAGES_PATH = "/plotwave"

DEMO_CODE_SNIPPET = (
    textwrap.dedent(
        """
        import plotwave

        plotwave.plot(
            [
                plotwave.audio_file("examples/plotwave_C_piano.wav", name="Piano", color="#2563eb"),
                plotwave.audio_file(
                    "examples/plotwave_C_VHS.wav",
                    name="VHS-style synth",
                    color="#ea580c",
                    line={"dash": "dot"},
                ),
                plotwave.segments(
                    [
                        {"start": 0.0, "end": 2.09, "label": "C"},
                        {"start": 2.09, "end": 4.18, "label": "Asus2"},
                        {"start": 4.18, "end": 6.27, "label": "Esus2"},
                        {"start": 6.27, "end": 8.3, "label": "B"},
                    ],
                    name="Ground Truth",
                    lane="top",
                    color_map={
                        "C": "#0f62fe",
                        "Asus2": "#367af7",
                        "Esus2": "#82adfc",
                        "B": "#bcd2fb",
                    },
                ),
                plotwave.segments(
                    [
                        {"start": 0.0, "end": 2.09, "label": "C"},
                        {"start": 2.09, "end": 4.18, "label": "D5"},
                        {"start": 4.18, "end": 6.27, "label": "Esus2"},
                        {"start": 6.27, "end": 8.3, "label": "B"},
                    ],
                    name="Prediction",
                    lane="bottom",
                    color_map={
                        "C": "#0f62fe",
                        "D5": "#ee9f0d",
                        "Esus2": "#82adfc",
                        "B": "#bcd2fb",
                    },
                ),
            ],
            layout={"title": {"text": "Style transfer: piano to VHS-style synth"}},
        )
        """
    ).strip()
    + "\n"
)


def _python_token_class(token_type: int, token_string: str) -> str | None:
    if token_type == token.COMMENT:
        return "py-comment"
    if token_type == token.STRING:
        return "py-string"
    if token_type == token.NUMBER:
        return "py-number"
    if token_type == token.OP:
        return "py-operator"
    if token_type == token.NAME and keyword.iskeyword(token_string):
        return "py-keyword"
    if token_type == token.NAME and token_string == "plotwave":
        return "py-module"
    return None


def highlight_python(code: str) -> str:
    pieces: list[str] = []
    lines = code.splitlines(keepends=True)
    current_row = 1
    current_col = 0

    def append_plain(text: str) -> None:
        if text:
            pieces.append(html.escape(text))

    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type == tokenize.ENDMARKER:
            break
        start_row, start_col = tok.start
        end_row, end_col = tok.end
        if current_row == start_row:
            append_plain(lines[current_row - 1][current_col:start_col])
        else:
            append_plain(lines[current_row - 1][current_col:])
            for row in range(current_row + 1, start_row):
                append_plain(lines[row - 1])
            append_plain(lines[start_row - 1][:start_col])
        token_html = html.escape(tok.string)
        token_class = _python_token_class(tok.type, tok.string)
        if token_class:
            pieces.append(f'<span class="{token_class}">{token_html}</span>')
        else:
            pieces.append(token_html)
        current_row = end_row
        current_col = end_col

    if current_row <= len(lines):
        append_plain(lines[current_row - 1][current_col:])
        for row in range(current_row + 1, len(lines) + 1):
            append_plain(lines[row - 1])

    return "".join(pieces)


def inject_demo_code_panel(document: str) -> str:
    head_injection = textwrap.dedent(
        """
        <style>
          html, body {
            overflow-y: auto;
            overflow-x: hidden;
          }
          .plotwave-code-panel {
            margin: 0;
            border-bottom: 1px solid #e4e1dc;
            background: linear-gradient(180deg, #fffdfa 0%, #f8f4ee 100%);
          }
          .plotwave-code-panel summary {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 10px;
            padding: 14px 18px;
            cursor: pointer;
            list-style: none;
            -webkit-user-select: none;
            user-select: none;
          }
          .plotwave-code-panel summary::-webkit-details-marker {
            display: none;
          }
          .plotwave-code-panel__title {
            color: #171717;
            font: 700 13px/1.2 "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            letter-spacing: 0.01em;
          }
          .plotwave-code-panel__icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.08);
            color: #4f46e5;
            flex: 0 0 auto;
            transition: transform 180ms ease, background 180ms ease, color 180ms ease;
          }
          .plotwave-code-panel__icon svg {
            width: 14px;
            height: 14px;
            stroke: currentColor;
            stroke-width: 1.8;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
          }
          .plotwave-code-panel[open] .plotwave-code-panel__icon {
            transform: rotate(180deg);
            background: rgba(79, 70, 229, 0.16);
            color: #352ab8;
          }
          .plotwave-code-panel__body {
            display: grid;
            gap: 12px;
            padding: 0 18px 16px;
          }
          .plotwave-code-panel__copy {
            margin: 0;
            color: #5f5a52;
            font: 500 13px/1.55 "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
          }
          .plotwave-code-panel__copy code {
            padding: 2px 6px;
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.08);
            color: #4138c2;
            font: 700 0.92em ui-monospace, SFMono-Regular, Menlo, monospace;
          }
          .plotwave-code-panel__pre {
            margin: 0;
            overflow-x: auto;
            border: 1px solid #d8d3cc;
            border-radius: 14px;
            padding: 16px 18px;
            background: #161616;
          }
          .plotwave-code-panel__pre code {
            display: block;
            min-width: max-content;
            color: #f5f3ef;
            font: 500 13px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
            tab-size: 4;
          }
          .plotwave-code-panel .py-keyword { color: #ff7b72; }
          .plotwave-code-panel .py-string { color: #a5d6ff; }
          .plotwave-code-panel .py-number { color: #79c0ff; }
          .plotwave-code-panel .py-comment { color: #8b949e; }
          .plotwave-code-panel .py-operator { color: #f2cc60; }
          .plotwave-code-panel .py-module { color: #d2a8ff; }
          @media (max-width: 720px) {
            .plotwave-code-panel summary {
              gap: 10px;
            }
          }
        </style>
        """
    ).strip()

    code_markup = textwrap.dedent(
        """
        <details class="plotwave-code-panel" id="plotwave-code-panel">
          <summary>
            <span class="plotwave-code-panel__icon" aria-hidden="true">
              <svg viewBox="0 0 16 16" focusable="false">
                <path d="M4 6l4 4 4-4"></path>
              </svg>
            </span>
            <span class="plotwave-code-panel__title" id="plotwave-code-panel-title">Show Python code</span>
          </summary>
          <div class="plotwave-code-panel__body">
            <pre class="plotwave-code-panel__pre"><code>__CODE_HTML__</code></pre>
          </div>
        </details>
        """
    ).replace("__CODE_HTML__", highlight_python(DEMO_CODE_SNIPPET))

    panel_script = textwrap.dedent(
        """
        <script>
          (() => {
            const codePanel = document.getElementById("plotwave-code-panel");
            const codePanelTitle = document.getElementById("plotwave-code-panel-title");
            if (!codePanel) {
              return;
            }

            const syncCodePanelTitle = () => {
              if (!codePanelTitle) return;
              codePanelTitle.textContent = codePanel.open
                ? "Hide Python code"
                : "Show Python code";
            };

            const measuredContentHeight = () => {
              const bodyRect = document.body.getBoundingClientRect();
              const bodyTop = bodyRect.top;
              const childBottoms = Array.from(document.body.children)
                .filter((child) => child instanceof HTMLElement)
                .map((child) => child.getBoundingClientRect().bottom - bodyTop);
              const bodyHeight = bodyRect.bottom - bodyTop;
              return Math.ceil(Math.max(bodyHeight, ...childBottoms));
            };

            const publishHeight = () => {
              if (!window.parent || window.parent === window) return;
              const height = Math.max(620, measuredContentHeight());
              window.parent.postMessage({ type: "plotwave-demo-height", height }, "*");
            };

            const publishHeightSoon = () => {
              requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                  publishHeight();
                });
              });
            };

            codePanel.addEventListener("toggle", () => {
              syncCodePanelTitle();
              publishHeight();
              publishHeightSoon();
            });

            if (typeof ResizeObserver !== "undefined") {
              const heightObserver = new ResizeObserver(() => publishHeight());
              heightObserver.observe(document.body);
              heightObserver.observe(document.documentElement);
              heightObserver.observe(codePanel);
            }

            window.addEventListener("load", publishHeight);
            window.addEventListener("resize", publishHeight);
            syncCodePanelTitle();
            publishHeight();
            publishHeightSoon();
          })();
        </script>
        """
    )

    document = document.replace("</head>", f"{head_injection}\n</head>", 1)
    document = document.replace("<body>", f"<body>\n{code_markup}", 1)
    document = document.replace("</body>", f"{panel_script}\n</body>", 1)
    return document


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
    first_audio_path = REPO_ROOT / "examples" / "plotwave_C_piano.wav"
    second_audio_path = REPO_ROOT / "examples" / "plotwave_C_VHS.wav"

    demo = plotwave.plot(
        [
            plotwave.audio_file(
                first_audio_path,
                name="Piano",
                color="#2563eb",
                line={"width": 1.3},
            ),
            plotwave.audio_file(
                second_audio_path,
                name="VHS-style synth",
                color="#ea580c",
                line={"width": 1.3, "dash": "dot"},
            ),
            plotwave.segments(
                [
                    {"start": 0.0, "end": 2.09, "label": "C"},
                    {"start": 2.09, "end": 4.18, "label": "Asus2"},
                    {"start": 4.18, "end": 6.27, "label": "Esus2"},
                    {"start": 6.27, "end": 8.3, "label": "B"},
                ],
                name="Ground Truth",
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
                    {"start": 0.0, "end": 2.09, "label": "C"},
                    {"start": 2.09, "end": 4.18, "label": "D5"},
                    {"start": 4.18, "end": 6.27, "label": "Esus2"},
                    {"start": 6.27, "end": 8.3, "label": "B"},
                ],
                name="Prediction",
                lane="bottom",
                textfont={"color": "white"},
                color_map={
                    "C": "#0f62fe",
                    "D5": "#ee9f0d",
                    "Esus2": "#82adfc",
                    "B": "#bcd2fb",
                },
            ),
        ],
        layout={
            "title": {"text": "Style transfer: piano to VHS-style synth"},
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
    demo_html = inject_demo_code_panel(demo.html())
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
      height: 620px;
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
        height: 620px;
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
          signal views into interactive, playable audio plots. This demo shows
          a simple style transfer example where a "surrogate generative model" does two things: it
          generates a VHS-style synth version of a short piano phrase and it
          predicts the chord labels in the same view.
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
          <p class="meta-label">Comparable</p>
          <p class="meta-text">
            The dropdown next to Play lets you switch between the piano and the VHS-style synth.
          </p>
        </div>
        <div class="meta-card">
          <p class="meta-label">Loopable</p>
          <p class="meta-text">Click a chord label to loop the audio for that labeled region.</p>
        </div>
      </div>
    </section>


    <section class="frame">
      <iframe
        id="plotwave-demo-frame"
        title="plotwave interactive demo"
        src="./demo.html?v={demo_version}"
        loading="eager"
      ></iframe>
    </section>

    <script>
      (() => {{
        const demoFrame = document.getElementById("plotwave-demo-frame");
        if (!(demoFrame instanceof HTMLIFrameElement)) return;
        window.addEventListener("message", (event) => {{
          const data = event.data;
          if (!data || data.type !== "plotwave-demo-height") return;
          if (typeof data.height !== "number" || !Number.isFinite(data.height)) return;
          demoFrame.style.height = `${{Math.max(620, Math.ceil(data.height))}}px`;
        }});
      }})();
    </script>

  </main>
</body>
</html>
"""
    (docs_dir / "index.html").write_text(index_html, encoding="utf-8")


if __name__ == "__main__":
    main()
