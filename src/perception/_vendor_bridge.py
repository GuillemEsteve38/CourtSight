"""Puts vendor/TennisProject on sys.path so its flat (non-package) modules import cleanly.

vendor/TennisProject is a vendored reference (see CLAUDE.md), not installed as a
package -- its files use flat imports like `from tracknet import BallTrackerNet` that
only resolve if the vendor directory itself is on sys.path. Call `ensure()` once
before importing any vendor module.
"""

import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "TennisProject"

_ensured = False


def ensure():
    global _ensured
    if _ensured:
        return
    if not VENDOR_DIR.is_dir():
        raise FileNotFoundError(
            f"vendor/TennisProject not found at {VENDOR_DIR} -- see SETUP.md step 1"
        )
    sys.path.insert(0, str(VENDOR_DIR))
    _ensured = True
