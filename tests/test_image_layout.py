"""The app must import from what the Dockerfile copies into the image, and from nothing else.

No Docker needed. The runtime stage's ``COPY`` lines are replayed into a temp folder laid out
like ``/app``, then ``import api`` runs in a fresh interpreter with the image's ``PYTHONPATH``.
A file the app reads at import but the Dockerfile forgets to copy fails here, in seconds,
instead of in a published image.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Runs inside the temp layout. The second check matters on a machine where the project is
# installed in editable mode: the checkout's src/ is then on sys.path of every interpreter
# and would quietly supply a module the image does not have.
IMPORT_CHECK = """
import sys
import api
leaked = sorted(n for n, m in sys.modules.items() if (getattr(m, "__file__", None) or "").startswith(sys.argv[1]))
if leaked:
    sys.exit(f"loaded from the checkout instead of the image layout: {leaked}")
"""


def context_copies(dockerfile: Path) -> list[tuple[str, str]]:
    """``(source, destination)`` for every COPY from the build context in the last stage."""
    copies: list[tuple[str, str]] = []
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        keyword = words[0].upper()
        if keyword == "FROM":
            copies = []  # a new stage starts; only the last one ships
        elif keyword == "COPY" and not any(w.startswith("--from=") for w in words):
            *sources, dest = [w for w in words[1:] if not w.startswith("--")]
            copies += [(src, dest) for src in sources]
    return copies


def build_layout(app: Path) -> None:
    """Recreate under ``app`` what the image has under its WORKDIR."""
    copies = context_copies(ROOT / "Dockerfile")
    assert copies, "found no COPY from the build context in the Dockerfile's last stage"
    for src, dest in copies:
        assert not Path(dest).is_absolute(), f"COPY destination {dest!r}: this test only handles paths relative to WORKDIR"
        source, target = ROOT / src, app / dest
        assert source.exists(), f"the Dockerfile copies {src!r}, which is not in the repository"
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        else:
            target = target / source.name if dest.endswith("/") else target
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def test_app_imports_from_the_image_layout(tmp_path: Path) -> None:
    app = tmp_path / "app"
    build_layout(app)
    env = {**os.environ, "PYTHONPATH": str(app / "src"), "RUNS_DIR": str(tmp_path / "runs")}
    proc = subprocess.run(
        [sys.executable, "-c", IMPORT_CHECK, str(ROOT / "src")],
        cwd=app, env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"`import api` failed in the image layout:\n{proc.stderr[-2000:]}"
