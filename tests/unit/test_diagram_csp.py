"""
docs/diagrams/add_csp.py, which gives each rendered diagram a CSP before it is
published to GitHub Pages.

The rendered pages are build artifacts that live only on the gh-pages branch,
so these tests exercise the script on a small page with the same shape: inline
viewer scripts, JSON data blocks, a charset tag.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "docs" / "diagrams" / "add_csp.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("add_csp", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


add_csp = _load()

_PAGE = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <script>
      document.documentElement.dataset.theme = "dark";
    </script>
    <script id="data" type="application/json">{"views": []}</script>
  </head>
  <body>
    <script>console.log("viewer");</script>
  </body>
</html>
"""


def _sha256(body: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(body.encode()).digest()).decode() + "'"


def test_only_executable_scripts_are_allowed_by_their_exact_hash() -> None:
    policy = add_csp.policy_for(_PAGE)
    assert _sha256('\n      document.documentElement.dataset.theme = "dark";\n    ') in policy
    assert _sha256('console.log("viewer");') in policy
    assert _sha256('{"views": []}') not in policy  # data, never run
    assert policy.count("'sha256-") == 2


def test_the_policy_allows_no_network_and_no_other_origin() -> None:
    policy = add_csp.policy_for(_PAGE)
    assert "default-src 'none'" in policy
    assert "connect-src 'none'" in policy
    assert "http" not in policy
    assert "'unsafe-inline'" not in policy.split("style-src")[0]  # scripts: hashes only
    assert "unsafe-eval" not in policy


def test_the_tag_precedes_every_script() -> None:
    html = add_csp.with_csp(_PAGE)
    assert html.index("Content-Security-Policy") < html.index("<script")


def test_running_it_again_replaces_the_tag() -> None:
    once = add_csp.with_csp(_PAGE)
    assert add_csp.with_csp(once) == once
    assert once.count("Content-Security-Policy") == 1


def test_a_changed_script_gets_a_new_hash() -> None:
    changed = _PAGE.replace('console.log("viewer");', 'console.log("viewer 2");')
    assert add_csp.policy_for(changed) != add_csp.policy_for(_PAGE)


def test_an_external_script_is_refused() -> None:
    page = _PAGE.replace("<script>console", '<script src="https://cdn.example/x.js">console')
    with pytest.raises(ValueError, match="external script"):
        add_csp.policy_for(page)
