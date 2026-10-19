"""Forward-looking evaluation harness for the synthetic-data agent.

Importing this package puts ``src/`` on ``sys.path`` so that the agent modules,
which import each other by top-level name, resolve from anywhere in the repo.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
