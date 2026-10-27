from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROCESSES_DIR = Path(__file__).resolve().parent.parent / "processes"


def load_process(process_id: str) -> dict[str, Any]:
    path = PROCESSES_DIR / f"{process_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Process definition not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or data.get("process_id") != process_id:
        raise ValueError(f"Invalid process file for id={process_id}")
    return data


def list_processes() -> list[str]:
    return sorted(p.stem for p in PROCESSES_DIR.glob("*.yaml"))
