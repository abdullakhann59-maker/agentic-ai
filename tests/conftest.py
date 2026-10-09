"""Test setup: temporary data folders + fake models (no Ollama needed)."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="nexus_test_"))
os.environ.update({
    "DATA_DIR": str(TMP), "CHROMA_DIR": str(TMP / "chroma"), "SQLITE_PATH": str(TMP / "nexus.db"),
    "WORKSPACE_DIR": str(TMP / "workspace"), "SCRAPE_DELAY_SECONDS": "0", "OLLAMA_BASE_URL": "http://127.0.0.1:9",
})
# Optional: point Selenium at a specific Chrome/driver (CI or sandbox)
for var in ("CHROME_BINARY", "CHROMEDRIVER_PATH", "CHROMEDRIVER_ARGS"):
    if os.environ.get("TEST_" + var):
        os.environ[var] = os.environ["TEST_" + var]

from app import llm, storage  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.knowledge import reset_kb  # noqa: E402
from tests.fakes import HashEmbeddings, ScriptedChatModel  # noqa: E402

DEMO_PORT = 8011
_setup_counter = [0]


@pytest.fixture(scope="session", autouse=True)
def _setup():
    get_settings.cache_clear()
    storage.init_db()
    uploads = Path(get_settings().workspace_dir) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "data" / "seed" / "files").iterdir():
        shutil.copy(f, uploads / f.name)
    yield
    from app.browser import browser
    browser.close()
    shutil.rmtree(TMP, ignore_errors=True)


@pytest.fixture
def fake_llm():
    """A fresh scripted model + fresh knowledge base for each test."""
    model = ScriptedChatModel(replies=[], calls=[], bound_tools=[])
    llm.set_overrides(chat=model, embeddings=HashEmbeddings())
    # a fresh vector DB folder per test (deleting an open Chroma folder breaks the client)
    _setup_counter[0] += 1
    get_settings().chroma_dir = TMP / f"chroma_{_setup_counter[0]}"
    reset_kb()
    from app.agent import ensure_tool_index
    ensure_tool_index()
    yield model
    reset_kb()


@pytest.fixture(scope="session")
def demo_server():
    from demo_sites.server import start
    server = start(DEMO_PORT)
    yield f"http://localhost:{DEMO_PORT}"
    server.shutdown()


def chrome_available() -> bool:
    if os.environ.get("CHROME_BINARY"):
        return Path(os.environ["CHROME_BINARY"]).exists()
    return any(shutil.which(b) for b in ("google-chrome", "chromium", "chromium-browser", "chrome"))
