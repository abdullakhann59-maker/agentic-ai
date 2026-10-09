"""Safe file paths: every tool may only touch files inside the workspace folder."""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings


def resolve(path: str) -> Path:
    ws = Path(get_settings().workspace_dir).resolve()
    p = Path(path)
    full = (p if p.is_absolute() else ws / p).resolve()
    if ws not in full.parents and full != ws:
        raise PermissionError("Access denied: files must be inside the workspace folder.")
    if not full.exists():
        # be forgiving: the user may give just the file name of an upload
        alt = ws / "uploads" / p.name
        if alt.exists():
            return alt
        raise FileNotFoundError(f"File not found: {path}")
    return full


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(get_settings().workspace_dir).resolve()))
