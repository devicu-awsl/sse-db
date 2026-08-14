"""Make ``src/`` importable when a script is run directly from a checkout.

Installing the package (``uv sync`` or ``pip install -e .``) makes this
unnecessary, but a script that only works after an editable install is a script
that will fail in cron at 18:30 on a Friday.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
