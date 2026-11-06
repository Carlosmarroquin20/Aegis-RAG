"""
The gateway playground (deploy/gateway-playground) must run the real gateway,
and only code this site serves.

The page loads exactly the files build.py lists in its manifest, on top of a
Python that has pydantic and nothing else from this project's dependencies. A
module the gateway starts importing but the manifest misses would only fail in
a visitor's browser, so the build's --verify step imports the built tree the
same way: from the built directory alone, with the structlog stand-in.

The runtime itself is downloaded and hash-checked by the publishing workflow;
here the build skips it (--no-runtime) and only the checks are exercised.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DIR = _ROOT / "deploy" / "gateway-playground"
_BUILD = _DIR / "build.py"


def _load_build() -> ModuleType:
    spec = importlib.util.spec_from_file_location("playground_build", _BUILD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build()


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("playground")
    subprocess.run(
        [sys.executable, str(_BUILD), "--out", str(out), "--no-runtime", "--verify"],
        check=True,
    )
    return out


def test_the_page_and_its_modules_are_built(built: Path) -> None:
    for name in (*build.PAGE_FILES, "py/manifest.json"):
        assert (built / name).is_file(), name
    manifest = json.loads((built / "py" / "manifest.json").read_text(encoding="utf-8"))
    assert "aegis/infrastructure/security/security_gateway.py" in manifest
    assert "structlog.py" in manifest
    assert all((built / "py" / entry).is_file() for entry in manifest)


class TestOnlyThisSiteServesCode:
    def test_the_csp_allows_no_other_origin(self) -> None:
        page = (_DIR / "index.html").read_text(encoding="utf-8")
        csp = re.search(r'Content-Security-Policy"\s+content="([^"]+)"', page)
        assert csp is not None
        assert "http" not in csp.group(1), "the policy names an external origin"
        assert "'unsafe-eval'" not in csp.group(1)
        assert "'unsafe-inline'" not in csp.group(1)

    def test_no_script_or_stylesheet_is_loaded_from_elsewhere(self) -> None:
        page = (_DIR / "index.html").read_text(encoding="utf-8")
        for tag in re.findall(r"<(?:script|link)\b[^>]*>", page):
            assert "http" not in tag, tag

    def test_the_script_does_not_point_the_runtime_elsewhere(self) -> None:
        script = (_DIR / "playground.js").read_text(encoding="utf-8")
        assert "http" not in script.split("const HARNESS")[0]
        assert "indexURL" not in script


class TestRuntimeIntegrity:
    def test_a_matching_sri_hash_passes(self) -> None:
        data = b"runtime"
        digest = base64.b64encode(hashlib.sha512(data).digest()).decode()
        build.check_sri(data, f"sha512-{digest}")

    def test_a_tampered_file_fails_the_sri_check(self) -> None:
        digest = base64.b64encode(hashlib.sha512(b"runtime").digest()).decode()
        with pytest.raises(build.IntegrityError):
            build.check_sri(b"runtime, modified", f"sha512-{digest}")

    def test_a_weak_algorithm_is_refused(self) -> None:
        with pytest.raises(build.IntegrityError):
            build.check_sri(b"x", "md5-" + base64.b64encode(hashlib.md5(b"x").digest()).decode())

    def test_a_wheel_must_match_the_lock_file(self) -> None:
        build.check_sha256(b"wheel", hashlib.sha256(b"wheel").hexdigest(), "w.whl")
        with pytest.raises(build.IntegrityError):
            build.check_sha256(b"other", hashlib.sha256(b"wheel").hexdigest(), "w.whl")

    def test_dependencies_are_resolved_from_the_lock_file(self) -> None:
        lock = {
            "packages": {
                "pydantic": {"name": "pydantic", "depends": ["pydantic_core", "typing-extensions"]},
                "pydantic_core": {"name": "pydantic_core", "depends": ["typing-extensions"]},
                "typing-extensions": {"name": "typing-extensions", "depends": []},
                "numpy": {"name": "numpy", "depends": []},
            }
        }
        names = [entry["name"] for entry in build.package_closure(lock, ("pydantic",))]
        # Names are matched across "-" and "_", as the lock file spells them both ways.
        assert names == ["pydantic", "pydantic_core", "typing-extensions"]

    def test_the_pinned_integrity_is_sha512(self) -> None:
        assert build.PYODIDE_INTEGRITY.startswith("sha512-")
        assert build.PYODIDE_VERSION in build.PYODIDE_TARBALL
        assert f"v{build.PYODIDE_VERSION}/" in build.PYODIDE_PACKAGES
