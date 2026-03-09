#!/usr/bin/env python3
"""Build documentation: Markdown -> HTML with navigation, math, and syntax highlighting.

Self-contained build script. Requires only the Python `markdown` package:
    pip install markdown

Usage:
    python scripts/build_docs.py            # Build to docs/_site/
    python scripts/build_docs.py --serve    # Build and serve on localhost:8000
"""

import argparse
import os
import re
import shutil
import sys

try:
    import markdown
    from markdown.extensions.tables import TableExtension
    from markdown.extensions.fenced_code import FencedCodeExtension
    from markdown.extensions.codehilite import CodeHiliteExtension
    from markdown.extensions.toc import TocExtension
except ImportError:
    print("Error: 'markdown' package not found. Install with: pip install markdown")
    sys.exit(1)

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
OUT_DIR = os.path.join(DOCS_DIR, "_site")
ASSETS_DIR = os.path.join(DOCS_DIR, "assets")

# Navigation structure: (title, path_relative_to_docs)
NAV = [
    ("Guide", [
        ("Home", "index.md"),
        ("Tutorial", "tutorial.md"),
        ("API Reference", "api.md"),
    ]),
]


def _protect_math(md_text):
    """Extract math blocks before Markdown processing to prevent mangling."""
    placeholders = {}
    counter = [0]

    def replace(m):
        key = f"\x00MATH{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = m.group(0)
        return key

    text = re.sub(r'\$\$(.+?)\$\$', replace, md_text, flags=re.DOTALL)
    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', replace, text)
    return text, placeholders


def _restore_math(html, placeholders):
    """Re-insert math blocks after Markdown processing."""
    for key, original in placeholders.items():
        html = html.replace(key, original)
    return html


def md_to_html(md_text):
    """Convert Markdown to HTML with extensions, preserving LaTeX math."""
    protected_text, math_placeholders = _protect_math(md_text)

    extensions = [
        TableExtension(),
        FencedCodeExtension(),
        TocExtension(permalink=True, toc_depth="2-3"),
    ]
    try:
        extensions.append(CodeHiliteExtension(guess_lang=False, css_class="highlight"))
    except Exception:
        pass

    md = markdown.Markdown(extensions=extensions, output_format="html")
    html = md.convert(protected_text)

    html = _restore_math(html, math_placeholders)
    return html


def render_sidebar(current_path):
    """Render the sidebar navigation HTML."""
    html = []
    for section_title, pages in NAV:
        html.append(f'<h3>{section_title}</h3>')
        html.append('<ul>')
        for title, md_path in pages:
            html_path = md_path.replace(".md", ".html")
            active = ' class="active"' if md_path == current_path else ""
            current_dir = os.path.dirname(current_path) or "."
            target_dir = os.path.dirname(html_path) or "."
            if current_dir == target_dir:
                href = os.path.basename(html_path)
            elif current_dir == "." and target_dir != ".":
                href = html_path
            elif current_dir != "." and target_dir == ".":
                href = "../" + os.path.basename(html_path)
            else:
                href = "../" + html_path
            html.append(f'  <li><a href="{href}"{active}>{title}</a></li>')
        html.append('</ul>')
    return "\n".join(html)


def render_page(md_text, current_path, title):
    """Render a full HTML page from Markdown content."""
    content_html = md_to_html(md_text)

    # Fix internal .md links -> .html
    content_html = re.sub(r'href="([^"]*?)\.md"', r'href="\1.html"', content_html)

    sidebar_html = render_sidebar(current_path)

    depth = current_path.count("/")
    prefix = "../" * depth if depth > 0 else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} — jaxrate</title>
  <link rel="stylesheet" href="{prefix}assets/style.css">
  <!-- KaTeX for math rendering -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{
      delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '$', right: '$', display: false}}
      ]
    }});"></script>
  <!-- highlight.js for syntax highlighting -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css">
  <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
  <script>hljs.highlightAll();</script>
</head>
<body>
  <header class="site-header">
    <a href="{prefix}index.html">jaxrate</a>
    <nav>
      <a href="{prefix}index.html">Home</a>
      <a href="{prefix}tutorial.html">Tutorial</a>
      <a href="{prefix}api.html">API</a>
      <a href="https://github.com/ihh/jaxrate">GitHub</a>
    </nav>
  </header>
  <div class="page-wrapper">
    <aside class="sidebar">
      {sidebar_html}
    </aside>
    <main class="content">
      {content_html}
    </main>
  </div>
  <footer class="site-footer">
    jaxrate &mdash; Stochastic MCFGs with phylogenetic terminal weights &mdash;
    <a href="https://github.com/ihh/jaxrate">ihh/jaxrate</a>
  </footer>
</body>
</html>
"""


def build():
    """Build all documentation pages."""
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    assets_out = os.path.join(OUT_DIR, "assets")
    if os.path.exists(ASSETS_DIR):
        shutil.copytree(ASSETS_DIR, assets_out)

    page_count = 0
    for _, pages in NAV:
        for title, md_path in pages:
            src = os.path.join(DOCS_DIR, md_path)
            if not os.path.exists(src):
                print(f"  SKIP  {md_path} (not found)")
                continue

            with open(src, "r") as f:
                md_text = f.read()

            html = render_page(md_text, md_path, title)
            out_path = os.path.join(OUT_DIR, md_path.replace(".md", ".html"))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                f.write(html)
            page_count += 1
            print(f"  BUILD {md_path} -> {md_path.replace('.md', '.html')}")

    print(f"\nBuilt {page_count} pages to {OUT_DIR}/")
    return page_count


def serve(port=8000):
    """Serve the built docs on localhost."""
    import http.server
    import functools

    os.chdir(OUT_DIR)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=OUT_DIR)
    with http.server.HTTPServer(("", port), handler) as httpd:
        print(f"Serving docs at http://localhost:{port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="Build jaxrate documentation")
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument("--port", type=int, default=8000, help="Port for local server")
    args = parser.parse_args()

    print("Building documentation...")
    count = build()

    if count == 0:
        print("No pages built!")
        sys.exit(1)

    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()
