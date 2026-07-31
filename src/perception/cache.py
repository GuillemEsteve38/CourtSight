"""JSON caching helpers shared by the perception wrapper modules.

Per CLAUDE.md, GPU inference runs once per clip and results are persisted to data/
as JSON so every downstream stage reads from cache instead of re-running models.
"""

import json
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, cls=_NumpyEncoder)


def load_json(path):
    with open(path) as f:
        return json.load(f)
