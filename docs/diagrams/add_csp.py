"""
Adds a Content-Security-Policy to a rendered Archify diagram before it is published.

    python docs/diagrams/add_csp.py aegis-architecture.html [more.html ...]

The rendered pages are self-contained: fonts and images are embedded as data
URLs, nothing is fetched, and the viewer never builds HTML from strings. A CSP
keeps it that way if a later Archify release changes: the page may run only
the inline scripts it shipped with (allowed by their SHA-256), load only
embedded fonts and images, and connect nowhere.

GitHub Pages cannot send headers, so the policy is a <meta> tag placed right
after <meta charset>, ahead of every script. Running the script again replaces
the tag it added, so it can follow every re-render. tests/unit/test_diagram_csp.py
checks the published policy against the scripts with the same functions.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
from pathlib import Path

# Executable inline scripts: <script> with no type, or a JavaScript type. Data
# blocks (type="application/json") are never run and need no hash.
_SCRIPT = re.compile(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.DOTALL)
_TYPE = re.compile(r"""\btype\s*=\s*["']([^"']+)["']""")
_EXECUTABLE_TYPES = {"text/javascript", "application/javascript", "module"}
_CSP_META = re.compile(r'\s*<meta http-equiv="Content-Security-Policy"[^>]*>')
_CHARSET = re.compile(r"<meta charset=[^>]*>", re.IGNORECASE)

# Everything the viewer needs and nothing more: embedded fonts and images,
# blob: URLs for the PNG/SVG/WebM exports it builds in the page, inline styles
# (the viewer sets style attributes), and no network at all.
_POLICY = (
    "default-src 'none'; script-src {scripts}; style-src 'unsafe-inline'; "
    "img-src data: blob:; font-src data:; media-src blob:; connect-src 'none'; "
    "base-uri 'none'; form-action 'none'"
)


def executable_scripts(html: str) -> list[str]:
    """The bodies of the inline scripts a browser would run."""
    bodies = []
    for match in _SCRIPT.finditer(html):
        attrs = match.group("attrs")
        if "src=" in attrs:
            raise ValueError("the page loads an external script; it is no longer self-contained")
        declared = _TYPE.search(attrs)
        if declared is None or declared.group(1).lower() in _EXECUTABLE_TYPES:
            bodies.append(match.group("body"))
    return bodies


def script_hash(body: str) -> str:
    digest = hashlib.sha256(body.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode() + "'"


def policy_for(html: str) -> str:
    hashes = sorted({script_hash(body) for body in executable_scripts(html)})
    return _POLICY.format(scripts=" ".join(hashes) or "'none'")


def with_csp(html: str) -> str:
    """The page with its policy tag added, or replaced if it already has one."""
    html = _CSP_META.sub("", html)
    charset = _CHARSET.search(html)
    if charset is None:
        raise ValueError("no <meta charset> to place the policy after")
    tag = f'\n    <meta http-equiv="Content-Security-Policy" content="{policy_for(html)}" />'
    return html[: charset.end()] + tag + html[charset.end() :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("pages", nargs="+", type=Path)
    for page in parser.parse_args().pages:
        html = page.read_text(encoding="utf-8")
        page.write_text(with_csp(html), encoding="utf-8", newline="")
        print(f"{page}: {len(executable_scripts(html))} inline script(s) allowed by hash")


if __name__ == "__main__":
    main()
