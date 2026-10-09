"""Per-request context shared by the agent and its tools.

Tools need to know which chat session they run in (to save files, store scraped
datasets, or register an action that needs the user's confirmation) and they must
report LLM tokens they use internally. A ContextVar keeps this per request.
"""
from __future__ import annotations

import contextvars
import itertools
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunContext:
    session_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    files: list[str] = field(default_factory=list)
    pending_action: dict | None = None

    def add_tokens(self, inp: int, out: int) -> None:
        self.input_tokens += inp
        self.output_tokens += out

    def add_file(self, path: str) -> None:
        if path not in self.files:
            self.files.append(path)


_current: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar("run_ctx", default=None)


def set_context(ctx: RunContext) -> contextvars.Token:
    return _current.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    _current.reset(token)


def ctx() -> RunContext:
    c = _current.get()
    if c is None:  # tools called outside the agent (tests, scripts)
        c = RunContext(session_id="default")
        _current.set(c)
    return c


# ---------------------------------------------------------------------------
# Session data that must survive between /chat calls (in memory, per session):
# scraped datasets and site profiles. Large data stays here; the LLM only sees
# short summaries and ids, which keeps prompts small and fast.
# ---------------------------------------------------------------------------
class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._ids = itertools.count(1)

    def _bucket(self, session_id: str) -> dict[str, Any]:
        return self._data.setdefault(session_id, {"datasets": {}, "profiles": {}})

    def save_dataset(self, records: list[dict], meta: dict) -> str:
        with self._lock:
            ds_id = f"ds_{next(self._ids)}"
            self._bucket(ctx().session_id)["datasets"][ds_id] = {"records": records, "meta": meta}
            return ds_id

    def get_dataset(self, ds_id: str) -> dict | None:
        return self._bucket(ctx().session_id)["datasets"].get(ds_id)

    def save_profile(self, url: str, profile: dict) -> None:
        self._bucket(ctx().session_id)["profiles"][url] = profile

    def get_profile(self, url: str) -> dict | None:
        return self._bucket(ctx().session_id)["profiles"].get(url)


session_store = SessionStore()
