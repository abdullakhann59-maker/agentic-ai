"""Load the starter knowledge into the vector DB and copy demo files into the workspace.

Run once after installing (Ollama must be running):
    python -m scripts.load_seed
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.agent import ensure_tool_index
from app.config import get_settings
from app.knowledge import get_kb
from app.storage import init_db

SEED = Path(__file__).resolve().parent.parent / "data" / "seed"


def main() -> None:
    init_db()
    kb = get_kb()
    for t in json.loads((SEED / "task_playbooks.json").read_text()):
        kb.add_task(t["title"], t["description"], t["tools"], t["examples"])
    for c in json.loads((SEED / "chat_examples.json").read_text()):
        kb.add_chat(c["q"], c["a"])
    for doc in (SEED / "docs").iterdir():
        kb.add_document(doc)
    ensure_tool_index()

    uploads = Path(get_settings().workspace_dir) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    for f in (SEED / "files").iterdir():
        shutil.copy(f, uploads / f.name)
    print("Knowledge loaded:", kb.counts())
    print("Demo files copied to", uploads)


if __name__ == "__main__":
    main()
