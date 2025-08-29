from __future__ import annotations
import json
from pathlib import Path

def save_feature_list(feature_names, path: str | Path):
    path = Path(path)
    path.write_text(json.dumps(list(feature_names), indent=2))

def load_feature_list(path: str | Path):
    return json.loads(Path(path).read_text())

def label_mapping():
    return {0: "Not Divorced / Not Stable", 1: "Divorced / Stable"}