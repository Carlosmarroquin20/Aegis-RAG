"""
Builds the gateway playground: a static page that runs the real SecurityGateway
in the visitor's browser, on Pyodide.

    python deploy/gateway-playground/build.py --out build/playground

The page is a demo of the input guard only, and it must be the guard: not a
port to JavaScript, not a copy that can drift. So the Python it loads is taken
from src/ at build time, by following the gateway's imports. Whatever the
gateway imports from the project is copied, every package __init__ on the way
included; anything else it imports must be the standard library or one of the
packages the browser provides, or the build stops.

Output: the page (PAGE_FILES) beside py/, which holds the copied modules and
manifest.json, the list of files the page writes into Pyodide's file system.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"
_ENTRY = "aegis.infrastructure.security.security_gateway"

# Third-party imports the browser can satisfy: pydantic ships with Pyodide, and
# structlog is replaced by a no-op module (logging has nowhere to go in a page).
BROWSER_PROVIDED = frozenset({"pydantic", "structlog"})

PAGE_FILES = ("index.html", "playground.js", "playground.css")


def module_path(module: str) -> Path:
    base = _SRC.joinpath(*module.split("."))
    package = base / "__init__.py"
    return package if package.exists() else base.with_suffix(".py")


def project_imports(path: Path) -> tuple[set[str], set[str]]:
    """(aegis modules, other top-level packages) imported anywhere in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    project: set[str] = set()
    external: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            if name.split(".")[0] == "aegis":
                project.add(name)
            elif name != "__future__":
                external.add(name.split(".")[0])
    return project, external


def closure(entry: str) -> tuple[list[Path], set[str]]:
    """Every project file the entry module needs, and the external packages they use."""
    files: set[Path] = set()
    external: set[str] = set()
    pending = [entry]
    seen: set[str] = set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        parts = module.split(".")
        # Importing a.b.c runs a/__init__.py and a/b/__init__.py first.
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        path = module_path(module)
        if not path.exists():
            raise SystemExit(f"cannot find {module} at {path}")
        files.add(path)
        project, other = project_imports(path)
        pending.extend(project)
        external |= other
    return sorted(files), external


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, required=True)
    out: Path = parser.parse_args().out

    files, external = closure(_ENTRY)
    unsupported = {
        name for name in external - BROWSER_PROVIDED if name not in sys.stdlib_module_names
    }
    if unsupported:
        raise SystemExit(
            f"the gateway now imports {sorted(unsupported)}, which the page cannot load"
        )

    if out.exists():
        shutil.rmtree(out)
    (out / "py").mkdir(parents=True)
    for page_file in PAGE_FILES:
        shutil.copy2(_HERE / page_file, out / page_file)

    manifest = []
    for path in files:
        relative = path.relative_to(_SRC).as_posix()
        target = out / "py" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest.append(relative)
    shutil.copy2(_HERE / "structlog_stub.py", out / "py" / "structlog.py")
    manifest.append("structlog.py")
    (out / "py" / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(manifest)} files into {out}")


if __name__ == "__main__":
    main()
